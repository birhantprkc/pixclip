"""
core/batch.py
Batch processing engine for PixClip.

Processes multiple images concurrently using a ThreadPoolExecutor.
Keeps the UI responsive by emitting progress through a callback.
The same process_frame() pipeline is used — ensuring 100% consistency
between interactive edits and batch exports.
"""

from __future__ import annotations
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from core.params import AdjustmentParams, ProjectState, ImageRecord
from core.pipeline import process_frame
from core.filters import apply_filter


class ExportFormat(Enum):
    JPEG = "jpg"
    PNG = "png"
    WEBP = "webp"
    TIFF = "tif"


@dataclass
class ExportOptions:
    output_dir: Path
    format: ExportFormat = ExportFormat.JPEG
    jpeg_quality: int = 95
    webp_quality: int = 90
    overwrite: bool = True
    suffix: str = ""  # appended to filename before extension, e.g. "_edited"
    max_workers: int = 4


@dataclass
class BatchResult:
    image_id: str
    source_path: Path
    output_path: Optional[Path]
    success: bool
    error: Optional[str] = None


ProgressCallback = Callable[[int, int, BatchResult], None]


def _export_single(
    record: ImageRecord,
    params: AdjustmentParams,
    options: ExportOptions,
) -> BatchResult:
    """Process and export a single image. Runs in a worker thread."""
    try:
        if record.original is None:
            # Load from disk if not already in memory
            img = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
            if img is None:
                return BatchResult(record.id, record.path, None, False, "Failed to read image")
        else:
            img = record.original.copy()

        # Process through the full pipeline
        result = process_frame(img, params, filter_lut_fn=apply_filter)

        # Build output path
        stem = record.path.stem + options.suffix
        ext = options.format.value
        out_path = options.output_dir / f"{stem}.{ext}"

        # Write output
        if options.format == ExportFormat.JPEG:
            cv2.imwrite(str(out_path), result,
                        [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality])
        elif options.format == ExportFormat.PNG:
            cv2.imwrite(str(out_path), result,
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
        elif options.format == ExportFormat.WEBP:
            cv2.imwrite(str(out_path), result,
                        [cv2.IMWRITE_WEBP_QUALITY, options.webp_quality])
        elif options.format == ExportFormat.TIFF:
            cv2.imwrite(str(out_path), result)

        return BatchResult(record.id, record.path, out_path, True)

    except Exception as e:
        return BatchResult(record.id, record.path, None, False, str(e))


class BatchProcessor:
    """
    Batch processor for exporting images from a ProjectState.
    Supports progress callbacks for UI integration.
    """

    def __init__(self, state: ProjectState, options: ExportOptions):
        self.state = state
        self.options = options
        self._cancelled = False

    def cancel(self) -> None:
        """Signal the batch operation to stop after the current image."""
        self._cancelled = True

    def run(
        self,
        image_ids: Optional[List[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[BatchResult]:
        """
        Run batch export for specified images (or all if image_ids is None).

        Args:
            image_ids: IDs of images to export. None = all images.
            progress_callback: Called after each image with (completed, total, result).

        Returns:
            List of BatchResult for each processed image.
        """
        self._cancelled = False
        options = self.options
        options.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve which images to process
        if image_ids is None:
            records = self.state.images
        else:
            records = [img for img in self.state.images if img.id in image_ids]

        total = len(records)
        results: List[BatchResult] = []

        # Build (record, params) pairs with resolved params
        tasks: List[Tuple[ImageRecord, AdjustmentParams]] = [
            (rec, self.state.resolved_params(rec.id)) for rec in records
        ]

        completed = 0
        with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
            future_to_record: Dict[Future, ImageRecord] = {
                executor.submit(_export_single, rec, params, options): rec
                for rec, params in tasks
            }

            for future in as_completed(future_to_record):
                if self._cancelled:
                    # Cancel remaining futures
                    for f in future_to_record:
                        f.cancel()
                    break

                result = future.result()
                results.append(result)
                completed += 1

                if progress_callback:
                    progress_callback(completed, total, result)

        return results


def load_image_record(record: ImageRecord) -> bool:
    """
    Load the original image data into an ImageRecord.
    Returns True if successful. Sets record.original and record.thumbnail.
    """
    try:
        img = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if img is None:
            return False
        record.original = img

        # Build thumbnail (160x160 max, aspect-preserving)
        h, w = img.shape[:2]
        scale = min(160 / w, 160 / h)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        record.thumbnail = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        return True
    except Exception:
        return False
