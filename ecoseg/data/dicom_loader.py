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
    ct_datasets: Optional[list[pydicom.Dataset]] = None,
) -> tuple[np.ndarray, str]:
    """Load a DICOM SEG object and extract the binary mask.

    DICOM SEGs only contain frames for slices with annotations, so we
    need to map each SEG frame back to the correct CT slice position
    using the PerFrameFunctionalGroupsSequence.

    Args:
        seg_path: path to DICOM SEG file
        reference_shape: (D, H, W) shape of the CT volume to match
        ct_datasets: sorted list of CT pydicom datasets (for z-position mapping)

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

    mask = np.zeros(reference_shape, dtype=np.uint8)

    # Build a z-position to slice index lookup from the CT datasets
    z_to_slice = {}
    if ct_datasets:
        for i, ct_ds in enumerate(ct_datasets):
            z_pos = round(float(ct_ds.ImagePositionPatient[2]), 3)
            z_to_slice[z_pos] = i

    # Extract frames from the SEG and map to CT slices
    pixel_data = ds.pixel_array  # (num_frames, rows, cols) or with extra dim

    if pixel_data.ndim == 4:
        pixel_data = pixel_data.squeeze(-1)

    num_frames = pixel_data.shape[0]
    per_frame_groups = getattr(ds, "PerFrameFunctionalGroupsSequence", None)

    if per_frame_groups and ct_datasets:
        # Map each SEG frame to the correct CT slice using z-position
        for frame_idx in range(num_frames):
            frame_group = per_frame_groups[frame_idx]

            # Get the z-position of this frame
            plane_pos = getattr(frame_group, "PlanePositionSequence", None)
            if plane_pos:
                z_pos = round(float(plane_pos[0].ImagePositionPatient[2]), 3)
                slice_idx = z_to_slice.get(z_pos)
                if slice_idx is not None:
                    frame_data = pixel_data[frame_idx]
                    mask[slice_idx, :frame_data.shape[0], :frame_data.shape[1]] = (
                        frame_data > 0
                    ).astype(np.uint8)
                else:
                    # Try nearest z-position
                    z_positions = np.array(list(z_to_slice.keys()))
                    nearest_idx = np.argmin(np.abs(z_positions - z_pos))
                    slice_idx = z_to_slice[z_positions[nearest_idx]]
                    frame_data = pixel_data[frame_idx]
                    mask[slice_idx, :frame_data.shape[0], :frame_data.shape[1]] = (
                        frame_data > 0
                    ).astype(np.uint8)
    else:
        # Fallback: if no per-frame groups or no CT datasets,
        # assume frames are consecutive starting at slice 0
        logger.warning("No per-frame position info; placing SEG frames sequentially")
        n = min(num_frames, reference_shape[0])
        for i in range(n):
            frame_data = pixel_data[i]
            mask[i, :frame_data.shape[0], :frame_data.shape[1]] = (
                frame_data > 0
            ).astype(np.uint8)

    return mask.astype(np.uint8), annotation_type


def _load_and_cache_worker(info: dict, study_id: str, cache_path: Optional[str]) -> str:
    """Load a study from DICOM and write .npz cache to disk.

    Returns the study_id on success. The main process then reads the
    .npz from local disk, avoiding pickling large arrays through pipes.
    """
    volume, spacing, ct_datasets = load_ct_volume(info["ct_dir"])

    seg_mask = None
    annotation_type = "unknown"
    if info["seg_path"] is not None:
        seg_mask, annotation_type = load_seg_mask(
            info["seg_path"],
            reference_shape=volume.shape,
            ct_datasets=ct_datasets,
        )

    ct_series_uid = str(getattr(ct_datasets[0], "SeriesInstanceUID", ""))

    if cache_path is not None:
        out = Path(cache_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            volume=volume,
            seg_mask=seg_mask if seg_mask is not None else np.array([]),
            spacing=np.array(spacing),
            meta=np.array([study_id, info["patient_id"], annotation_type, ct_series_uid]),
        )

    return study_id


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

    def __init__(self, data_root: Path, cache_dir: Optional[Path] = None):
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
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

    def _load_single_study(self, study_id: str) -> StudyData:
        """Load a single study from disk (no caching). Used by parallel loader."""
        info = self._index[study_id]

        volume, spacing, ct_datasets = load_ct_volume(info["ct_dir"])

        seg_mask = None
        annotation_type = "unknown"
        if info["seg_path"] is not None:
            seg_mask, annotation_type = load_seg_mask(
                info["seg_path"],
                reference_shape=volume.shape,
                ct_datasets=ct_datasets,
            )

        return StudyData(
            study_id=study_id,
            patient_id=info["patient_id"],
            volume=volume,
            spacing=spacing,
            seg_mask=seg_mask,
            annotation_type=annotation_type,
            ct_series_uid=str(getattr(ct_datasets[0], "SeriesInstanceUID", "")),
        )

    def load_study(self, study_id: str) -> StudyData:
        """Load a single study's CT volume and segmentation."""
        if study_id in self._studies:
            return self._studies[study_id]

        if self._index is None:
            self.discover_studies()

        if study_id not in self._index:
            raise KeyError(f"Study {study_id} not found in dataset")

        study = self._load_single_study(study_id)
        self._studies[study_id] = study
        return study

    def _cache_path(self, study_id: str) -> Optional[Path]:
        """Return the .npz cache file path for a study, or None if caching is disabled."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{study_id}.npz"

    def _save_to_cache(self, study: StudyData) -> None:
        """Save a study to the local disk cache as .npz."""
        path = self._cache_path(study.study_id)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                volume=study.volume,
                seg_mask=study.seg_mask if study.seg_mask is not None else np.array([]),
                spacing=np.array(study.spacing),
                meta=np.array([
                    study.study_id, study.patient_id,
                    study.annotation_type, study.ct_series_uid,
                ]),
            )
        except Exception as e:
            logger.warning(f"Failed to cache {study.study_id}: {e}")

    def _load_from_cache(self, study_id: str) -> Optional[StudyData]:
        """Load a study from the local disk cache. Returns None on miss."""
        path = self._cache_path(study_id)
        if path is None or not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            seg_mask = data["seg_mask"]
            if seg_mask.ndim == 0 or seg_mask.size == 0:
                seg_mask = None
            meta = data["meta"]
            return StudyData(
                study_id=str(meta[0]),
                patient_id=str(meta[1]),
                volume=data["volume"],
                spacing=tuple(data["spacing"].tolist()),
                seg_mask=seg_mask,
                annotation_type=str(meta[2]),
                ct_series_uid=str(meta[3]),
            )
        except Exception as e:
            logger.warning(f"Cache read failed for {study_id}: {e}")
            return None

    def preload_studies(self, study_ids: list[str], num_workers: int = -1) -> None:
        """Pre-load multiple studies into memory.

        Phase 1: Load from local .npz cache (fast, local disk reads).
        Phase 2: For cache misses, spawn worker processes that read DICOM
                 and write .npz files directly to disk (no pickle through
                 pipes). Then read the .npz files from local disk.

        Args:
            study_ids: list of study IDs to pre-load
            num_workers: number of parallel worker processes (-1 = auto)
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if self._index is None:
            self.discover_studies()

        to_load = [sid for sid in study_ids if sid not in self._studies]
        if not to_load:
            logger.info("All requested studies already in memory")
            return

        # Phase 1: load from local cache in parallel (I/O bound, threads are fine)
        from concurrent.futures import ThreadPoolExecutor

        if num_workers < 0:
            from ecoseg import available_workers
            num_workers = available_workers()

        cache_hits = 0
        still_need = []

        # Check which ones have cache files
        cached_sids = []
        for sid in to_load:
            path = self._cache_path(sid)
            if path is not None and path.exists():
                cached_sids.append(sid)
            else:
                still_need.append(sid)

        if cached_sids:
            logger.info(f"Loading {len(cached_sids)} studies from cache using {num_workers} threads...")
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(self._load_from_cache, sid): sid
                    for sid in cached_sids
                }
                for future in futures:
                    sid = futures[future]
                    try:
                        study = future.result()
                        if study is not None:
                            self._studies[sid] = study
                            cache_hits += 1
                        else:
                            still_need.append(sid)
                    except Exception:
                        still_need.append(sid)

            logger.info(f"Loaded {cache_hits}/{len(cached_sids)} studies from cache")

        if not still_need:
            return

        # Phase 2: workers write .npz to disk, main process reads them back
        if self.cache_dir is None:
            # Need a cache dir for the worker strategy to work
            self.cache_dir = Path.home() / ".ecoseg" / "cache"

        logger.info(
            f"Loading {len(still_need)} studies from DICOM using "
            f"{num_workers} workers (cache: {self.cache_dir})"
        )

        loaded = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_sid = {
                executor.submit(
                    _load_and_cache_worker,
                    self._index[sid],
                    sid,
                    str(self._cache_path(sid)),
                ): sid
                for sid in still_need
            }

            for future in as_completed(future_to_sid):
                sid = future_to_sid[future]
                try:
                    future.result()  # Just the study_id string, not data
                    # Read back from local cache — fast local disk read
                    study = self._load_from_cache(sid)
                    if study is not None:
                        self._studies[sid] = study
                        loaded += 1
                    else:
                        logger.warning(f"  Cache file missing after write for {sid}")
                        failed += 1
                    if loaded % 20 == 0 or loaded == len(still_need):
                        logger.info(f"  Loaded {loaded}/{len(still_need)} from DICOM")
                except Exception as e:
                    logger.warning(f"  Failed to load {sid}: {e}")
                    failed += 1

        logger.info(f"Pre-loading complete: {cache_hits} cached, {loaded} from DICOM, {failed} failed")

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
