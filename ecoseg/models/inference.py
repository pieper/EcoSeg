"""Inference mode: apply trained species models to segment full volumes.

Slides a patch window across the volume, computes fitness for all species
at each position, and assigns each voxel to the species with maximum fitness.
Overlapping patch contributions are averaged.
"""

import torch
import numpy as np
from dataclasses import dataclass

from ecoseg.models.species import SpeciesRegistry


@dataclass
class InferenceConfig:
    patch_size: int = 32
    stride: int = 16
    batch_size: int = 128


def infer_volume(
    registry: SpeciesRegistry,
    volume: np.ndarray,
    config: InferenceConfig = InferenceConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference mode on a full volume.

    Computes per-voxel fitness for all species using a sliding window
    approach, then assigns each voxel to the species with maximum fitness.

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

    # Accumulation buffers for averaging overlapping patches
    fitness_sum = np.zeros((num_species, d, h, w), dtype=np.float64)
    count = np.zeros((d, h, w), dtype=np.float64)

    # Collect all patch center positions
    positions = []
    for z in range(0, max(d - ps + 1, 1), stride):
        for y in range(0, max(h - ps + 1, 1), stride):
            for x in range(0, max(w - ps + 1, 1), stride):
                positions.append((z, y, x))

    # Process in batches
    vol_tensor = torch.tensor(volume, dtype=torch.float32)

    for batch_start in range(0, len(positions), config.batch_size):
        batch_positions = positions[batch_start:batch_start + config.batch_size]

        patches = torch.stack([
            vol_tensor[z:z + ps, y:y + ps, x:x + ps]
            for z, y, x in batch_positions
        ]).unsqueeze(1).to(device)  # (B, 1, ps, ps, ps)

        # Get fitness from all species
        all_fitness = registry.fitness_all(patches)

        # Accumulate into output buffers
        for i, name in enumerate(registry.species_names):
            scores = all_fitness[name].cpu().numpy()
            for j, (z, y, x) in enumerate(batch_positions):
                fitness_sum[i, z:z + ps, y:y + ps, x:x + ps] += scores[j]

        for z, y, x in batch_positions:
            count[z:z + ps, y:y + ps, x:x + ps] += 1.0

    # Average overlapping contributions
    count = np.maximum(count, 1.0)
    fitness_map = (fitness_sum / count[np.newaxis]).astype(np.float32)

    # Argmax across species
    labels = fitness_map.argmax(axis=0).astype(np.int32)

    return labels, fitness_map
