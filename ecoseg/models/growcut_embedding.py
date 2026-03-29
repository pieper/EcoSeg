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
    max_iterations: int = 500
    convergence_threshold: float = 0.0  # 0 = run until no changes at all


def growcut_embedding(
    embeddings: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig = GrowCutConfig(),
    fitness_fn: Optional[callable] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GrowCut using embedding similarity, preserving per-seed competition.

    Stays true to the GrowCut algorithm: each individual seed voxel
    competes independently, and the winner at each voxel is the seed
    with the strongest path through similar tissue. Different seeds
    from the same class compete — the one with the best path wins,
    providing natural variability in the "DNA" that conquers each region.

    Fitness at each step combines two factors:
      - Local similarity: cosine similarity between the attacking neighbor's
        embedding and the target voxel's embedding (boundary detection —
        propagation attenuates at tissue boundaries)
      - Prototype affinity: cosine similarity between the target voxel's
        embedding and the attacking species' prototype (species selectivity —
        prevents propagation into wrong tissue types)

    fitness = local_similarity * prototype_affinity

    This means: a label propagates easily through homogeneous tissue
    that matches the species prototype, but stops at boundaries (low
    local similarity) AND stops in wrong tissue (low prototype affinity).

    Args:
        embeddings: (C, D, H, W) per-voxel embedding vectors (on GPU)
        seeds: (D, H, W) int tensor — 0 = unlabeled, 1+ = species label
        config: iteration parameters
        fitness_fn: optional override fitness function

    Returns:
        labels: (D, H, W) int tensor — final species assignment
        strength: (D, H, W) float tensor — confidence of assignment
    """
    device = embeddings.device
    C, D, H, W = embeddings.shape

    # Normalize embeddings for cosine similarity
    emb_norm = F.normalize(embeddings, dim=0)  # (C, D, H, W)

    # Compute species prototypes (mean embedding of each species' seeds)
    unique_labels = seeds.unique()
    unique_labels = unique_labels[unique_labels > 0]

    prototypes = {}
    for lbl in unique_labels:
        lbl_val = lbl.item()
        mask = (seeds == lbl_val)
        seed_embs = emb_norm[:, mask]  # (C, num_seeds)
        proto = seed_embs.mean(dim=1)  # (C,)
        prototypes[lbl_val] = F.normalize(proto, dim=0)

    # Pre-compute prototype affinity for each species at every voxel
    # This is the "does this voxel match this species?" component
    proto_affinity = {}
    for lbl_val, proto in prototypes.items():
        sim = (proto.view(C, 1, 1, 1) * emb_norm).sum(dim=0)  # (D, H, W)
        proto_affinity[lbl_val] = sim.clamp(0, 1)

    # Initialize labels and strength from seeds
    labels = seeds.clone()
    strength = torch.zeros(D, H, W, device=device)
    seed_mask = seeds > 0
    strength[seed_mask] = 1.0

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
            # Shifted neighbor data
            neighbor_emb = torch.roll(emb_norm, shifts=(-dz, -dy, -dx), dims=(1, 2, 3))
            neighbor_labels = torch.roll(labels, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))
            neighbor_strength = torch.roll(strength, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))

            # Boundary mask
            # Mask out boundary voxels where the rolled neighbor wraps around.
            # For offset dz=-1 (neighbor at z-1): z=0 has no z-1 neighbor,
            # but roll brings z=-1 (=last slice) into position 0, so mask z=0.
            # For offset dz=+1 (neighbor at z+1): z=last has no z+1 neighbor,
            # but roll brings z=0 into position last, so mask z=last.
            boundary_mask = torch.ones(D, H, W, device=device, dtype=torch.bool)
            if dz == -1: boundary_mask[0, :, :] = False
            elif dz == 1: boundary_mask[-1, :, :] = False
            if dy == -1: boundary_mask[:, 0, :] = False
            elif dy == 1: boundary_mask[:, -1, :] = False
            if dx == -1: boundary_mask[:, :, 0] = False
            elif dx == 1: boundary_mask[:, :, -1] = False

            # Local similarity: how similar is the center voxel to the
            # attacking neighbor? (boundary detection, just like classic GrowCut)
            local_sim = (emb_norm * neighbor_emb).sum(dim=0).clamp(0, 1)  # (D, H, W)

            for lbl_val in prototypes:
                # Combined fitness = local boundary preservation × species affinity
                fitness = local_sim * proto_affinity[lbl_val]

                # Attack strength = fitness × neighbor's accumulated strength
                # This preserves the classic GrowCut strength decay along paths
                attack = fitness * neighbor_strength

                can_attack = (
                    boundary_mask
                    & (neighbor_labels == lbl_val)
                    & ((attack > new_strength) | ((new_labels == 0) & (attack > 0)))
                )

                new_labels[can_attack] = lbl_val
                new_strength[can_attack] = attack[can_attack]
                changed += can_attack.sum().item()

        labels = new_labels
        strength = new_strength

        # Preserve original seeds (never overwrite user input)
        labels[seed_mask] = seeds[seed_mask]
        strength[seed_mask] = 1.0

        total_voxels = D * H * W
        change_fraction = changed / total_voxels
        if change_fraction < config.convergence_threshold:
            n_labeled = (labels > 0).sum().item()
            logger.info(
                f"GrowCut (embedding) converged at iteration {iteration + 1}, "
                f"labeled={n_labeled}/{total_voxels}"
            )
            break

        if (iteration + 1) % 100 == 0:
            n_labeled = (labels > 0).sum().item()
            logger.info(
                f"GrowCut (embedding) iter {iteration+1}: "
                f"changed={changed}, labeled={n_labeled}/{total_voxels}"
            )

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

            # Mask out boundary voxels where the rolled neighbor wraps around
            boundary_mask = torch.ones(D, H, W, device=device, dtype=torch.bool)
            if dz == -1: boundary_mask[0, :, :] = False
            elif dz == 1: boundary_mask[-1, :, :] = False
            if dy == -1: boundary_mask[:, 0, :] = False
            elif dy == 1: boundary_mask[:, -1, :] = False
            if dx == -1: boundary_mask[:, :, 0] = False
            elif dx == 1: boundary_mask[:, :, -1] = False

            # Fitness: 1 - normalized intensity difference
            fitness = 1.0 - torch.abs(volume - neighbor_vol)
            fitness = fitness.clamp(0, 1)

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

        seed_mask = seeds > 0
        labels[seed_mask] = seeds[seed_mask]
        strength[seed_mask] = 1.0

        total_voxels = D * H * W
        change_fraction = changed / total_voxels
        if change_fraction < config.convergence_threshold:
            n_labeled = (labels > 0).sum().item()
            logger.info(
                f"GrowCut (intensity) converged at iteration {iteration + 1}, "
                f"labeled={n_labeled}/{total_voxels}"
            )
            break

        if (iteration + 1) % 100 == 0:
            n_labeled = (labels > 0).sum().item()
            logger.info(
                f"GrowCut (intensity) iter {iteration+1}: "
                f"changed={changed}, labeled={n_labeled}/{total_voxels}"
            )

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
