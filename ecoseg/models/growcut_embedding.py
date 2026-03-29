"""GrowCut on embedding vectors.

Instead of using image intensity as the fitness term for GrowCut
propagation, this uses the similarity of per-voxel embedding vectors
from a pre-trained encoder (SwinUNETR).

The key insight: if two neighboring voxels have similar embeddings,
a label should propagate between them. If their embeddings are very
different (e.g., tissue boundary), propagation should stop.

Phase 1: Cosine similarity of embeddings as fitness
Phase 2: Learned discriminator as fitness (uses all paint strokes
         to train a separator in embedding space)
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
    max_iterations: int = 200
    convergence_threshold: float = 0.001  # Fraction of voxels still changing


def growcut_embedding(
    embeddings: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig = GrowCutConfig(),
    fitness_fn: Optional[callable] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GrowCut using embedding similarity for propagation.

    Each voxel has a label (0 = unlabeled, 1..N = species) and a strength
    (0-1, how confident the assignment is). Seeds start at strength 1.0.

    At each iteration, for each voxel, check all 6-connected neighbors.
    If a neighbor's label could "attack" this voxel:
      - Compute fitness = similarity between neighbor's embedding and this voxel's embedding
      - If fitness * neighbor_strength > this_voxel_strength, the neighbor wins
      - This voxel gets the neighbor's label with strength = fitness * neighbor_strength

    Args:
        embeddings: (C, D, H, W) per-voxel embedding vectors (on GPU)
        seeds: (D, H, W) int tensor — 0 = unlabeled, 1+ = species label (on GPU)
        config: iteration parameters
        fitness_fn: optional learned fitness function. If None, uses cosine similarity.
                    Signature: fitness_fn(emb_center, emb_neighbor) -> similarity (0-1)

    Returns:
        labels: (D, H, W) int tensor — final species assignment
        strength: (D, H, W) float tensor — confidence of assignment
    """
    device = embeddings.device
    C, D, H, W = embeddings.shape

    # Normalize embeddings for cosine similarity
    emb_norm = F.normalize(embeddings, dim=0)  # (C, D, H, W)

    # Initialize labels and strength from seeds
    labels = seeds.clone()  # (D, H, W)
    strength = (seeds > 0).float()  # Seeds have strength 1.0

    # 6-connected neighbor offsets
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
            # Shift embeddings, labels, and strength to represent this neighbor
            # Using roll + masking at boundaries
            neighbor_emb = torch.roll(emb_norm, shifts=(-dz, -dy, -dx), dims=(1, 2, 3))
            neighbor_labels = torch.roll(labels, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))
            neighbor_strength = torch.roll(strength, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))

            # Mask out boundary wrapping artifacts
            boundary_mask = torch.ones(D, H, W, device=device, dtype=torch.bool)
            if dz == -1:
                boundary_mask[-1, :, :] = False
            elif dz == 1:
                boundary_mask[0, :, :] = False
            if dy == -1:
                boundary_mask[:, -1, :] = False
            elif dy == 1:
                boundary_mask[:, 0, :] = False
            if dx == -1:
                boundary_mask[:, :, -1] = False
            elif dx == 1:
                boundary_mask[:, :, 0] = False

            # Compute fitness: cosine similarity between center and neighbor
            if fitness_fn is not None:
                fitness = fitness_fn(emb_norm, neighbor_emb)
            else:
                # Cosine similarity (embeddings already normalized)
                fitness = (emb_norm * neighbor_emb).sum(dim=0)  # (D, H, W)
                fitness = fitness.clamp(0, 1)  # Only positive similarity propagates

            # Attack strength = fitness * neighbor's strength
            attack = fitness * neighbor_strength

            # Neighbor wins if it has a label, attack > current strength,
            # and we're not at a boundary
            wins = (
                boundary_mask
                & (neighbor_labels > 0)
                & (attack > new_strength)
            )

            # Update labels and strength where neighbor wins
            new_labels[wins] = neighbor_labels[wins]
            new_strength[wins] = attack[wins]

            changed += wins.sum().item()

        labels = new_labels
        strength = new_strength

        # Preserve original seeds (never overwrite user input)
        seed_mask = seeds > 0
        labels[seed_mask] = seeds[seed_mask]
        strength[seed_mask] = 1.0

        # Check convergence
        total_voxels = D * H * W
        change_fraction = changed / total_voxels
        if change_fraction < config.convergence_threshold:
            logger.debug(f"GrowCut converged at iteration {iteration + 1} "
                         f"({changed} voxels changed, {change_fraction:.4f})")
            break

        if (iteration + 1) % 50 == 0:
            logger.debug(f"GrowCut iteration {iteration + 1}: "
                         f"{changed} voxels changed ({change_fraction:.4f})")

    return labels, strength


def growcut_intensity(
    volume: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig = GrowCutConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Traditional GrowCut using image intensity similarity.

    Fitness = 1 - |intensity_center - intensity_neighbor| / max_range.
    This is the classic GrowCut algorithm as used in 3D Slicer's
    "Grow from seeds" effect.

    Args:
        volume: (D, H, W) normalized CT volume in [0, 1] (on GPU)
        seeds: (D, H, W) int tensor — 0 = unlabeled, 1+ = species label
        config: iteration parameters

    Returns:
        labels: (D, H, W) int tensor — final species assignment
        strength: (D, H, W) float tensor — confidence
    """
    device = volume.device
    D, H, W = volume.shape

    labels = seeds.clone()
    strength = (seeds > 0).float()

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
            neighbor_vol = torch.roll(volume, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))
            neighbor_labels = torch.roll(labels, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))
            neighbor_strength = torch.roll(strength, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))

            boundary_mask = torch.ones(D, H, W, device=device, dtype=torch.bool)
            if dz == -1:
                boundary_mask[-1, :, :] = False
            elif dz == 1:
                boundary_mask[0, :, :] = False
            if dy == -1:
                boundary_mask[:, -1, :] = False
            elif dy == 1:
                boundary_mask[:, 0, :] = False
            if dx == -1:
                boundary_mask[:, :, -1] = False
            elif dx == 1:
                boundary_mask[:, :, 0] = False

            # Fitness: 1 - normalized intensity difference
            fitness = 1.0 - torch.abs(volume - neighbor_vol)
            fitness = fitness.clamp(0, 1)

            attack = fitness * neighbor_strength

            wins = (
                boundary_mask
                & (neighbor_labels > 0)
                & (attack > new_strength)
            )

            new_labels[wins] = neighbor_labels[wins]
            new_strength[wins] = attack[wins]
            changed += wins.sum().item()

        labels = new_labels
        strength = new_strength

        seed_mask = seeds > 0
        labels[seed_mask] = seeds[seed_mask]
        strength[seed_mask] = 1.0

        total_voxels = D * H * W
        change_fraction = changed / total_voxels
        if change_fraction < config.convergence_threshold:
            logger.debug(f"GrowCut (intensity) converged at iteration {iteration + 1}")
            break

    return labels, strength


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
        seeds: (D, H, W) int array — 0 = unlabeled, 1 = foreground, 2 = background
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
