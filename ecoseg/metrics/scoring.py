"""Segmentation scoring metrics matching LNQ2023 evaluation.

LNQ2023 uses:
- Dice coefficient via SimpleITK LabelOverlapMeasuresImageFilter
- Average Symmetric Surface Distance (ASSD) via SimpleITK SignedMaurerDistanceMap

We implement equivalent metrics using scipy/numpy to avoid a SimpleITK
dependency, but produce the same results (physical-space distances using
voxel spacing).
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass


@dataclass
class SegmentationScore:
    study_id: str
    dice: float
    assd: float  # Average Symmetric Surface Distance in mm


def dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    pred_bool = prediction.astype(bool)
    gt_bool = ground_truth.astype(bool)

    intersection = np.sum(pred_bool & gt_bool)
    sum_vols = np.sum(pred_bool) + np.sum(gt_bool)

    if sum_vols == 0:
        return 1.0  # Both empty = perfect match
    return 2.0 * intersection / sum_vols


def _surface_distances(
    mask: np.ndarray,
    reference: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Compute distances from surface of mask to surface of reference.

    Matches LNQ2023 approach: extract surface via erosion, compute
    distance transform of reference surface, sample at mask surface points.

    Args:
        mask: binary mask whose surface we measure from
        reference: binary mask whose surface we measure to
        spacing: voxel spacing in mm (D, H, W)

    Returns:
        Array of distances from each mask surface voxel to nearest
        reference surface voxel, in mm.
    """
    # Extract surfaces (equivalent to sitk.LabelContour)
    struct = ndimage.generate_binary_structure(3, 1)
    mask_eroded = ndimage.binary_erosion(mask.astype(bool), struct)
    mask_surface = mask.astype(bool) & ~mask_eroded

    ref_eroded = ndimage.binary_erosion(reference.astype(bool), struct)
    ref_surface = reference.astype(bool) & ~ref_eroded

    if not np.any(mask_surface) or not np.any(ref_surface):
        return np.array([0.0])

    # Distance transform from reference surface (in physical units)
    # distance_transform_edt computes distance from each 0-voxel to nearest 1-voxel
    # We want distance from each point to nearest ref_surface point
    ref_distance = ndimage.distance_transform_edt(~ref_surface, sampling=spacing)

    # Sample distances at mask surface locations
    distances = ref_distance[mask_surface]
    return distances


def average_symmetric_surface_distance(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """Compute ASSD matching LNQ2023 evaluation.

    ASSD = (mean(d(pred_surface, gt_surface)) + mean(d(gt_surface, pred_surface))) / 2

    Args:
        prediction: binary prediction mask
        ground_truth: binary ground truth mask
        spacing: voxel spacing in mm (D, H, W)

    Returns:
        ASSD in mm. Returns 0.0 if both masks are empty.
    """
    pred_bool = prediction.astype(bool)
    gt_bool = ground_truth.astype(bool)

    if not np.any(pred_bool) and not np.any(gt_bool):
        return 0.0

    if not np.any(pred_bool) or not np.any(gt_bool):
        # One is empty, the other is not — return a large distance
        # LNQ2023 handles this with zero-padding which gives misleading 0.0
        # We return inf to make it clear this case failed
        return float("inf")

    pred_to_gt = _surface_distances(pred_bool, gt_bool, spacing)
    gt_to_pred = _surface_distances(gt_bool, pred_bool, spacing)

    return (np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2.0


def score_segmentation(
    study_id: str,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> SegmentationScore:
    """Score a single segmentation against ground truth.

    Args:
        study_id: identifier for the study
        prediction: binary prediction mask (D, H, W)
        ground_truth: binary ground truth mask (D, H, W)
        spacing: voxel spacing in mm (D, H, W)

    Returns:
        SegmentationScore with Dice and ASSD
    """
    return SegmentationScore(
        study_id=study_id,
        dice=dice_score(prediction, ground_truth),
        assd=average_symmetric_surface_distance(prediction, ground_truth, spacing),
    )
