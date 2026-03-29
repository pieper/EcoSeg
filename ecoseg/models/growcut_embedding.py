"""GrowCut on embedding vectors.

Two variants:
1. growcut_intensity: Classic GrowCut using 1 - |intensity_diff| as fitness
2. growcut_embedding: Same algorithm but using cosine similarity of
   per-voxel embedding vectors as fitness

Both label every voxel. The hypothesis: embedding similarity respects
tissue boundaries better than intensity similarity, producing more
accurate segmentations from the same seed points.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GrowCutConfig:
    max_iterations: int = 1000
    convergence_threshold: float = 0.0
    stop_after_no_change: int = 2


def _growcut_core(
    fitness_fn,
    labels: torch.Tensor,
    strength: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig,
    method_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core GrowCut iteration shared by intensity and embedding variants.

    Args:
        fitness_fn: callable(dz, dy, dx) -> (D, H, W) fitness tensor
        labels: (D, H, W) initial labels (cloned from seeds)
        strength: (D, H, W) initial strength
        seeds: (D, H, W) original seeds (preserved each iteration)
        config: iteration config
        method_name: for logging

    Returns:
        labels, strength
    """
    device = labels.device
    D, H, W = labels.shape
    seed_mask = seeds > 0
    no_change_count = 0

    offsets = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

    for iteration in range(config.max_iterations):
        changed = 0
        new_labels = labels.clone()
        new_strength = strength.clone()

        for dz, dy, dx in offsets:
            neighbor_labels = torch.roll(labels, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))
            neighbor_strength = torch.roll(strength, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))

            # Boundary mask: prevent wrap-around from torch.roll
            boundary_mask = torch.ones(D, H, W, device=device, dtype=torch.bool)
            if dz == -1: boundary_mask[0, :, :] = False
            elif dz == 1: boundary_mask[-1, :, :] = False
            if dy == -1: boundary_mask[:, 0, :] = False
            elif dy == 1: boundary_mask[:, -1, :] = False
            if dx == -1: boundary_mask[:, :, 0] = False
            elif dx == 1: boundary_mask[:, :, -1] = False

            fitness = fitness_fn(dz, dy, dx)
            attack = fitness * neighbor_strength

            wins = (
                boundary_mask
                & (neighbor_labels > 0)
                & ((attack > new_strength) | ((new_labels == 0) & (attack > 0)))
            )

            new_labels[wins] = neighbor_labels[wins]
            new_strength[wins] = attack[wins]
            changed += wins.sum().item()

        labels = new_labels
        strength = new_strength

        labels[seed_mask] = seeds[seed_mask]
        strength[seed_mask] = 1.0

        total_voxels = D * H * W

        if changed == 0:
            no_change_count += 1
            if no_change_count >= config.stop_after_no_change:
                n_labeled = (labels > 0).sum().item()
                logger.info(
                    f"GrowCut ({method_name}) converged at iteration {iteration + 1}, "
                    f"labeled={n_labeled}/{total_voxels}"
                )
                break
        else:
            no_change_count = 0

        if (iteration + 1) % 100 == 0:
            n_labeled = (labels > 0).sum().item()
            logger.info(
                f"GrowCut ({method_name}) iter {iteration+1}: "
                f"changed={changed}, labeled={n_labeled}/{total_voxels}"
            )

    return labels, strength


def growcut_intensity(
    volume: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig = GrowCutConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classic GrowCut using intensity similarity.

    Fitness = 1 - |intensity_center - intensity_neighbor|
    """
    def fitness_fn(dz, dy, dx):
        neighbor_vol = torch.roll(volume, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))
        return (1.0 - torch.abs(volume - neighbor_vol)).clamp(0, 1)

    labels = seeds.clone()
    strength = (seeds > 0).float()
    return _growcut_core(fitness_fn, labels, strength, seeds, config, "intensity")


def growcut_embedding(
    embeddings: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig = GrowCutConfig(),
    fitness_fn: Optional[callable] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GrowCut using embedding cosine similarity as fitness.

    Exact same algorithm as intensity GrowCut, but fitness is the
    cosine similarity between neighboring embedding vectors instead
    of 1 - |intensity_difference|.

    Labels propagate through regions with similar embeddings and stop
    at boundaries where embeddings change.
    """
    emb_norm = F.normalize(embeddings, dim=0)

    def emb_fitness_fn(dz, dy, dx):
        neighbor_emb = torch.roll(emb_norm, shifts=(-dz, -dy, -dx), dims=(1, 2, 3))
        return (emb_norm * neighbor_emb).sum(dim=0).clamp(0, 1)

    labels = seeds.clone()
    strength = (seeds > 0).float()
    return _growcut_core(
        fitness_fn or emb_fitness_fn,
        labels, strength, seeds, config, "embedding",
    )


def simulate_paint_strokes(
    ground_truth: np.ndarray,
    num_positive: int = 50,
    num_negative: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate user paint strokes by sampling voxels from ground truth.

    Args:
        ground_truth: (D, H, W) binary mask
        num_positive: number of foreground seed voxels
        num_negative: number of background seed voxels
        rng: random number generator

    Returns:
        seeds: (D, H, W) int array -- 0 = unlabeled, 1 = foreground, 2 = background
    """
    if rng is None:
        rng = np.random.default_rng()

    seeds = np.zeros_like(ground_truth, dtype=np.int32)

    pos_coords = np.argwhere(ground_truth > 0)
    neg_coords = np.argwhere(ground_truth == 0)

    if len(pos_coords) > 0:
        n = min(num_positive, len(pos_coords))
        idx = rng.choice(len(pos_coords), size=n, replace=False)
        for z, y, x in pos_coords[idx]:
            seeds[z, y, x] = 1

    if len(neg_coords) > 0:
        n = min(num_negative, len(neg_coords))
        idx = rng.choice(len(neg_coords), size=n, replace=False)
        for z, y, x in neg_coords[idx]:
            seeds[z, y, x] = 2

    return seeds
