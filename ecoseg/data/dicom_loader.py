"""DICOM data loading for LNQ2023 dataset.

Loads CT volumes and SEG masks from local DICOM files.
The LNQ2023 dataset on TCIA/IDC provides:
- CT image series
- DICOM SEG objects with Series Description indicating
  "fully annotated" or "partially annotated"

This module handles loading these into numpy arrays for training
and inference.
"""

import numpy as np
import pydicom
from pydicom.uid import UID
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StudyData:
    """Loaded CT volume and segmentation for one study."""
    study_id: str
    patient_id: str
    volume: np.ndarray  # (D, H, W) in HU
    spacing: tuple[float, float, float]  # (slice_spacing, row_spacing, col_spacing) in mm
    seg_mask: Optional[np.ndarray] = None  # (D, H, W) binary lymph node mask
    annotation_type: str = "unknown"  # "fully_annotated" or "partially_annotated"
    ct_series_uid: str = ""
    seg_series_uid: str = ""


def load_ct_volume(dicom_dir: Path) -> tuple[np.ndarray, tuple[float, float, float], list[pydicom.Dataset]]:
    """Load a CT volume from a directory of DICOM slices.

    Args:
        dicom_dir: directory containing .dcm files for one CT series

    Returns:
        volume: (D, H, W) numpy array in Hounsfield Units
        spacing: (slice_spacing, row_spacing, col_spacing) in mm
        datasets: sorted list of pydicom datasets
    """
    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        # Try without extension
        dcm_files = [f for f in sorted(dicom_dir.iterdir()) if f.is_file()]

    datasets = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f)
            if hasattr(ds, "ImagePositionPatient"):
                datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise ValueError(f"No valid CT DICOM files found in {dicom_dir}")

    # Sort by slice position (z-axis)
    datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

    # Extract pixel data and apply rescale
    slices = []
    for ds in datasets:
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(pixels * slope + intercept)

    volume = np.stack(slices, axis=0)  # (D, H, W)

    # Spacing
    pixel_spacing = [float(x) for x in datasets[0].PixelSpacing]
    if len(datasets) > 1:
        slice_spacing = abs(
            float(datasets[1].ImagePositionPatient[2])
            - float(datasets[0].ImagePositionPatient[2])
        )
    else:
        slice_spacing = float(getattr(datasets[0], "SliceThickness", 1.0))

    spacing = (slice_spacing, pixel_spacing[0], pixel_spacing[1])

    return volume, spacing, datasets


def load_seg_mask(
    seg_path: Path,
    reference_shape: tuple[int, int, int],
) -> tuple[np.ndarray, str]:
    """Load a DICOM SEG object and extract the binary mask.

    Args:
        seg_path: path to DICOM SEG file
        reference_shape: (D, H, W) shape of the CT volume to match

    Returns:
        mask: (D, H, W) binary numpy array
        annotation_type: "fully_annotated" or "partially_annotated"
    """
    ds = pydicom.dcmread(seg_path)

    # Determine annotation type from Series Description
    series_desc = str(getattr(ds, "SeriesDescription", "")).lower()
    if "full" in series_desc:
        annotation_type = "fully_annotated"
    elif "partial" in series_desc:
        annotation_type = "partially_annotated"
    else:
        annotation_type = "unknown"

    # Extract pixel data from SEG
    # DICOM SEG can be binary or labelmap format
    try:
        import highdicom
        seg = highdicom.seg.Segmentation.from_dataset(ds)
        # Get the first segment (lymph node)
        mask_frames = seg.get_pixels_by_dimension(
            segment_numbers=[1],
        )
        # Reshape to volume
        if mask_frames.ndim == 4:
            mask = mask_frames.squeeze(-1)  # Remove segment dim
        else:
            mask = mask_frames

        # Ensure correct shape
        if mask.shape != reference_shape:
            logger.warning(
                f"SEG shape {mask.shape} != reference {reference_shape}, "
                "attempting to match..."
            )
            # Pad or crop to match
            result = np.zeros(reference_shape, dtype=np.uint8)
            slices = tuple(
                slice(0, min(m, r))
                for m, r in zip(mask.shape, reference_shape)
            )
            result[slices] = mask[slices]
            mask = result

    except Exception as e:
        logger.warning(f"highdicom loading failed ({e}), falling back to raw pixel data")
        pixel_data = ds.pixel_array
        if pixel_data.ndim == 3:
            mask = (pixel_data > 0).astype(np.uint8)
        elif pixel_data.ndim == 4:
            mask = (pixel_data.max(axis=-1) > 0).astype(np.uint8)
        else:
            mask = (pixel_data > 0).astype(np.uint8).reshape(reference_shape)

    return mask.astype(np.uint8), annotation_type


class LNQDataset:
    """Manager for the LNQ2023 dataset stored as local DICOM files.

    Expected directory structure (TCIA/IDC layout):
        data_root/
            case_XXXX/
                <StudyInstanceUID>/
                    CT_<SeriesInstanceUID>/
                        <SOPInstanceUID>.dcm  (CT slices)
                    SEG_<SeriesInstanceUID>/
                        <SOPInstanceUID>.dcm  (DICOM SEG)
    """

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self._studies: dict[str, StudyData] = {}
        self._index: Optional[dict] = None

    def discover_studies(self) -> dict[str, dict]:
        """Scan the data directory and build an index of available studies.

        Handles the TCIA/IDC layout where series directories are prefixed
        with CT_ or SEG_ followed by their SeriesInstanceUID.

        Returns:
            Dict mapping patient_id (e.g. case_0002) to metadata and paths.
        """
        index = {}
        logger.info(f"Scanning {self.data_root} for DICOM studies...")

        for patient_dir in sorted(self.data_root.iterdir()):
            if not patient_dir.is_dir():
                continue

            # Find the study directory (there should be one per patient)
            study_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
            if not study_dirs:
                continue

            for study_dir in study_dirs:
                # Find CT and SEG series by prefix
                ct_dir = None
                seg_dir = None
                for series_dir in sorted(study_dir.iterdir()):
                    if not series_dir.is_dir():
                        continue
                    name = series_dir.name
                    if name.startswith("CT"):
                        ct_dir = series_dir
                    elif name.startswith("SEG"):
                        seg_dir = series_dir

                if ct_dir is not None and seg_dir is not None:
                    # SEG directory may contain one file (the SEG object)
                    seg_files = list(seg_dir.glob("*.dcm"))
                    if not seg_files:
                        seg_files = [f for f in seg_dir.iterdir() if f.is_file()]

                    # Use patient_dir name (e.g. case_0002) as study_id
                    study_id = patient_dir.name
                    index[study_id] = {
                        "patient_id": patient_dir.name,
                        "study_uid": study_dir.name,
                        "ct_dir": ct_dir,
                        "seg_path": seg_files[0] if seg_files else None,
                    }

        self._index = index
        logger.info(f"Found {len(index)} studies")
        return index

    def load_study(self, study_id: str) -> StudyData:
        """Load a single study's CT volume and segmentation."""
        if study_id in self._studies:
            return self._studies[study_id]

        if self._index is None:
            self.discover_studies()

        if study_id not in self._index:
            raise KeyError(f"Study {study_id} not found in dataset")

        info = self._index[study_id]

        # Load CT
        volume, spacing, ct_datasets = load_ct_volume(info["ct_dir"])

        # Load SEG if available
        seg_mask = None
        annotation_type = "unknown"
        if info["seg_path"] is not None:
            seg_mask, annotation_type = load_seg_mask(
                info["seg_path"],
                reference_shape=volume.shape,
            )

        study = StudyData(
            study_id=study_id,
            patient_id=info["patient_id"],
            volume=volume,
            spacing=spacing,
            seg_mask=seg_mask,
            annotation_type=annotation_type,
            ct_series_uid=str(getattr(ct_datasets[0], "SeriesInstanceUID", "")),
        )
        self._studies[study_id] = study
        return study

    def get_fully_annotated_ids(self) -> list[str]:
        """Return study IDs with full annotations."""
        if self._index is None:
            self.discover_studies()
        # We need to peek at the SEG to know annotation type
        # For now return all — filtering happens after loading
        return list(self._index.keys())

    def get_validation_ids(self, count: int = 20) -> list[str]:
        """Return the first `count` fully-annotated study IDs (LNQ validation set)."""
        fully = [
            sid for sid in self._index
            if self._is_fully_annotated(sid)
        ]
        return fully[:count]

    def get_test_ids(self, validation_count: int = 20) -> list[str]:
        """Return fully-annotated study IDs not in the validation set."""
        fully = [
            sid for sid in self._index
            if self._is_fully_annotated(sid)
        ]
        return fully[validation_count:]

    def _is_fully_annotated(self, study_id: str) -> bool:
        """Check if a study has full annotation by peeking at the SEG."""
        info = self._index.get(study_id)
        if info is None or info["seg_path"] is None:
            return False
        try:
            ds = pydicom.dcmread(info["seg_path"], stop_before_pixels=True)
            series_desc = str(getattr(ds, "SeriesDescription", "")).lower()
            return "full" in series_desc
        except Exception:
            return False
