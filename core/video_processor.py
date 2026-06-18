"""
core/video_processor.py
Video processing backend: handles video import metadata extraction, thumbnailing,
and asynchronous frame-by-frame export with audio preservation using FFmpeg.
"""

from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.params import VideoRecord, AdjustmentParams
from core.pipeline import process_frame
from core.filters import apply_filter

import shutil

# Find FFmpeg in project root
ROOT_DIR = Path(__file__).parent.parent.resolve()

def find_ffmpeg() -> str:
    # 1. Check in ROOT_DIR/ffmpeg/ffmpeg.exe
    local_pkg = ROOT_DIR / "ffmpeg" / "ffmpeg.exe"
    if local_pkg.exists():
        return str(local_pkg)
    # 2. Check in ROOT_DIR/ffmpeg.exe
    local_root = ROOT_DIR / "ffmpeg.exe"
    if local_root.exists():
        return str(local_root)
    # 3. Check in system PATH
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
    # Fallback default path
    return str(local_pkg)

FFMPEG_EXE = find_ffmpeg()


def get_video_info(path: Path) -> Tuple[dict, Optional[np.ndarray]]:
    """
    Extracts video metadata and generates a thumbnail at 10% of duration.
    Returns: (metadata_dict, thumbnail_array_bgr)
    """
    metadata = {
        "duration": 0.0,
        "fps": 0.0,
        "frame_count": 0,
        "width": 0,
        "height": 0
    }
    thumbnail: Optional[np.ndarray] = None

    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        metadata.update({
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        })

        # Capture thumbnail at 10% of length to avoid initial black frames
        target_frame = min(max(0, int(frame_count * 0.1)), frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret and frame is not None:
            # Scale down to thumbnail size
            h, w = frame.shape[:2]
            scale = min(160 / w, 160 / h)
            tw, th = max(1, int(w * scale)), max(1, int(h * scale))
            thumbnail = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

        cap.release()

    return metadata, thumbnail


class VideoExportWorker(QThread):
    """
    Processes video frames in a background thread and compiles them with audio.
    Emits progress(status_text, current_frame, total_frames) and finished(success, message).
    """
    progress = Signal(str, int, int, object)  # (status, current, total, preview_frame)
    finished = Signal(bool, str)

    def __init__(self, record: VideoRecord, params: AdjustmentParams, out_path: str):
        super().__init__()
        self.record = record
        self.params = params
        self.out_path = out_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        input_file = str(self.record.path.resolve())
        temp_dir = Path(tempfile.gettempdir())
        
        temp_audio = temp_dir / f"pixclip_temp_audio_{self.record.id}.aac"
        temp_video = temp_dir / f"pixclip_temp_video_{self.record.id}.mp4"

        has_audio = False

        # ── Step 1: Extract Audio via FFmpeg ──────────────────────────────────
        if os.path.exists(FFMPEG_EXE):
            self.progress.emit("Extracting audio...", 0, 100, None)
            
            # Extract to AAC format (transcodes if necessary to avoid copy issues)
            cmd_audio = [
                FFMPEG_EXE, "-y",
                "-i", input_file,
                "-vn",
                "-acodec", "aac",
                "-ab", "192k",
                str(temp_audio)
            ]
            try:
                # Startupinfo to hide cmd window on Windows
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
                result = subprocess.run(
                    cmd_audio,
                    startupinfo=startupinfo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0 and temp_audio.exists() and temp_audio.stat().st_size > 500:
                    has_audio = True
            except Exception:
                has_audio = False
        
        # ── Step 2: Render Processed Video Frames ──────────────────────────────
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            self.finished.emit(False, "Could not open source video.")
            return

        total_frames = self.record.frame_count
        fps = self.record.fps
        width = self.record.width
        height = self.record.height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            cap.release()
            self.finished.emit(False, "Could not initialize video writer.")
            return

        frame_idx = 0
        try:
            while True:
                if self._cancelled:
                    break
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Apply stateless process_frame pipeline at full resolution
                processed = process_frame(frame, self.params, filter_lut_fn=apply_filter)
                writer.write(processed)
                
                frame_idx += 1
                
                # Emit progress and every 10 frames pass a small preview of the output
                if frame_idx % 10 == 0 or frame_idx == total_frames:
                    prev_small = cv2.resize(processed, (160, 100), interpolation=cv2.INTER_AREA)
                    self.progress.emit(f"Filtering frames...", frame_idx, total_frames, prev_small)
                else:
                    self.progress.emit(f"Filtering frames...", frame_idx, total_frames, None)

        except Exception as e:
            cap.release()
            writer.release()
            self._cleanup_temp_files(temp_audio, temp_video)
            self.finished.emit(False, f"Render error: {str(e)}")
            return

        cap.release()
        writer.release()

        if self._cancelled:
            self._cleanup_temp_files(temp_audio, temp_video)
            self.finished.emit(False, "Export cancelled.")
            return

        # ── Step 3: Mux Audio and Video together using FFmpeg ─────────────────
        if os.path.exists(FFMPEG_EXE):
            self.progress.emit("Compiling final video...", total_frames, total_frames, None)
            
            # Mux command using standard H.264 H.264 video (libx264) for maximum platform compatibility
            if has_audio:
                cmd_mux = [
                    FFMPEG_EXE, "-y",
                    "-i", str(temp_video),
                    "-i", str(temp_audio),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-shortest",
                    self.out_path
                ]
            else:
                cmd_mux = [
                    FFMPEG_EXE, "-y",
                    "-i", str(temp_video),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    self.out_path
                ]

            try:
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                result = subprocess.run(
                    cmd_mux,
                    startupinfo=startupinfo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    self._cleanup_temp_files(temp_audio, temp_video)
                    self.finished.emit(True, "Export completed successfully.")
                else:
                    self._cleanup_temp_files(temp_audio, temp_video)
                    self.finished.emit(False, f"Muxing failed: {result.stderr}")
            except Exception as e:
                self._cleanup_temp_files(temp_audio, temp_video)
                self.finished.emit(False, f"Muxing error: {str(e)}")
        else:
            # Fallback to copy the raw temporary video if FFmpeg is missing
            try:
                import shutil
                shutil.copy2(str(temp_video), self.out_path)
                self._cleanup_temp_files(temp_audio, temp_video)
                self.finished.emit(True, "Export completed (no audio compression fallback).")
            except Exception as e:
                self._cleanup_temp_files(temp_audio, temp_video)
                self.finished.emit(False, f"Copy error: {str(e)}")

    def _cleanup_temp_files(self, a_path: Path, v_path: Path):
        for p in [a_path, v_path]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
