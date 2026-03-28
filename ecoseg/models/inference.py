"""Inference mode: apply trained species models to segment full volumes.

Uses torch.unfold for GPU-side patch extraction. Since each species model
returns one scalar score per patch, the fold-back step maps per-patch scores
to per-voxel scores using torch.nn.functional.fold.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from ecoseg.models.species import SpeciesRegistry


@dataclass
class InferenceConfig:
    patch_size: int = 32
    stride: int = 16
    batch_size: int = 256


def infer_volume(
    registry: SpeciesRegistry,
    volume: np.ndarray,
    config: InferenceConfig = InferenceConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference mode on a full volume.

    Args:
        registry: species registry with trained models
        volume: (D, H, W) normalized CT volume in [0, 1]
        config: inference parameters

    Returns:
        labels: (D, H, W) int array - species index per voxel (0-based)
        fitness_map: (num_species, D, H, W) float array - fitness scores
    """
    device = registry.device
    ps = config.patch_size
    stride = config.stride
    d, h, w = volume.shape
    num_species = len(registry.species)

    if num_species == 0:
        return np.zeros((d, h, w), dtype=np.int32), np.zeros((0, d, h, w), dtype=np.float32)

    # --- Extract patches on GPU using unfold ---
    vol_t = torch.tensor(volume, dtype=torch.float32, device=device)

    # (D, H, W) -> (nz, ny, nx, ps, ps, ps)
    patches_unfolded = vol_t.unfold(0, ps, stride).unfold(1, ps, stride).unfold(2, ps, stride)
    nz, ny, nx = patches_unfolded.shape[:3]
    n_patches = nz * ny * nx

    # (N, 1, ps, ps, ps) for model input
    patches_flat = patches_unfolded.contiguous().view(n_patches, 1, ps, ps, ps)

    # --- Run batched inference for all species on GPU ---
    all_scores = []
    for sp_name in registry.species_names:
        species = registry.species[sp_name]
        scores_list = []
        for batch_start in range(0, n_patches, config.batch_size):
            batch = patches_flat[batch_start:batch_start + config.batch_size]
            with torch.no_grad():
                scores_list.append(species.network(batch).squeeze(-1))
        all_scores.append(torch.cat(scores_list, dim=0))

    # (num_species, nz, ny, nx)
    scores_grid = torch.stack(all_scores, dim=0).view(num_species, nz, ny, nx)

    # --- Fold per-patch scores back to per-voxel fitness ---
    # Each patch has one scalar score that applies to all ps^3 voxels in it.
    # Use F.fold on 2D slabs (H, W) for each z-slab, processing all species
    # at once. This replaces the Python triple loop with nz iterations only.

    fitness_sum = torch.zeros((num_species, d, h, w), device=device)
    count = torch.zeros((1, 1, d, h, w), device=device)

    # Precompute the 2D fold for (ny, nx) -> (H, W)
    # F.fold expects input shape (batch, C * kH * kW, L)
    # We'll process each z-position as a batch element

    # For the count: fold ones to get overlap count per voxel
    # Process z-axis with a loop (nz is small, typically <20 for stride 16)
    ones_2d = torch.ones((1, ps * ps, ny * nx), device=device)
    count_2d = F.fold(ones_2d, output_size=(h, w), kernel_size=ps, stride=stride)  # (1, 1, H, W)

    z_starts = list(range(0, d - ps + 1, stride))

    # Z overlap count
    z_count = torch.zeros(d, device=device)
    for zs in z_starts:
        z_count[zs:zs + ps] += 1.0
    z_count = z_count.clamp(min=1.0)

    for iz, zs in enumerate(z_starts):
        # scores for this z: (num_species, ny, nx)
        # Expand each score to fill a ps x ps patch, then fold
        z_scores = scores_grid[:, iz, :, :]  # (S, ny, nx)

        # Expand to (S, ps*ps, ny*nx): each score repeated ps*ps times
        z_expanded = z_scores.reshape(num_species, ny * nx).unsqueeze(1).expand(-1, ps * ps, -1)

        # Fold to (S, 1, H, W)
        folded = F.fold(z_expanded, output_size=(h, w), kernel_size=ps, stride=stride)

        # Accumulate across z-slices of the patch
        fitness_sum[:, zs:zs + ps, :, :] += folded.squeeze(1).unsqueeze(1).expand(-1, ps, -1, -1)

    # Divide by total overlap count (separable: z_count * count_2d)
    total_count = z_count.view(1, -1, 1, 1) * count_2d.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(0)
    total_count = total_count.clamp(min=1.0)

    fitness_map = fitness_sum / total_count

    labels = fitness_map.argmax(dim=0).int()

    return labels.cpu().numpy(), fitness_map.cpu().numpy()
