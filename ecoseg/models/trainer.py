"""Training loop for species models.

Extracts patches from labeled volumes and trains each species
as an independent binary classifier using BCE loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ecoseg.models.species import SpeciesModel


@dataclass
class TrainingConfig:
    patch_size: int = 32
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    positive_patches_per_scan: int = 500
    negative_patches_per_scan: int = 500
    num_workers: int = 0


def augment_patch(patch: torch.Tensor) -> torch.Tensor:
    """Apply random augmentations to a (1, D, H, W) patch.

    Augmentations:
    - Random flips along each spatial axis
    - Random intensity shift and scale
    """
    # Random flips along each spatial axis
    for dim in (1, 2, 3):
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, [dim])
    # Random intensity jitter: shift +/- 0.05, scale 0.9-1.1
    shift = (torch.rand(1).item() - 0.5) * 0.1
    scale = 0.9 + torch.rand(1).item() * 0.2
    patch = patch * scale + shift
    return patch.clamp(0.0, 1.0)


class PatchDataset(Dataset):
    """Dataset of 3D patches extracted from labeled volumes.

    Each item is a (patch, label) pair where:
    - patch: (1, 32, 32, 32) float tensor (CT intensities, normalized)
    - label: 0.0 or 1.0 (negative or positive for this species)
    """

    def __init__(
        self,
        patches: torch.Tensor,
        labels: torch.Tensor,
        augment: bool = True,
    ):
        self.patches = patches
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx]
        if self.augment:
            patch = augment_patch(patch)
        return patch, self.labels[idx]


def extract_patches(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 32,
    num_positive: int = 200,
    num_negative: int = 200,
    rng: Optional[np.random.Generator] = None,
    hard_negatives: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract positive and negative patches from a labeled volume.

    Args:
        volume: (D, H, W) CT volume, expected to be normalized to [0, 1]
        mask: (D, H, W) binary mask, 1 = positive for this species
        patch_size: side length of cubic patches
        num_positive: number of positive patches to extract
        num_negative: number of negative patches to extract
        rng: random number generator for reproducibility
        hard_negatives: if True, sample negatives from tissue regions with
            similar intensity to the positive class, forcing the model to
            learn shape/texture rather than just "tissue vs air"

    Returns:
        patches: (N, 1, ps, ps, ps) float tensor
        labels: (N,) float tensor of 0.0/1.0
    """
    if rng is None:
        rng = np.random.default_rng()

    half = patch_size // 2
    d, h, w = volume.shape

    # Valid region for patch centers (must fit full patch)
    valid_mask = np.zeros_like(mask, dtype=bool)
    valid_mask[half:d - half, half:h - half, half:w - half] = True

    pos_coords = np.argwhere(valid_mask & (mask > 0))

    if hard_negatives and len(pos_coords) > 0:
        # Sample negatives from tissue with similar intensity to positives.
        # This prevents the model from learning "tissue vs air" and forces
        # it to learn the actual distinguishing features of the target structure.
        pos_values = volume[mask > 0]
        intensity_low = np.percentile(pos_values, 10)
        intensity_high = np.percentile(pos_values, 90)

        # Tissue mask: voxels with similar intensity but NOT positive
        tissue_mask = (
            valid_mask
            & (mask == 0)
            & (volume >= intensity_low)
            & (volume <= intensity_high)
        )
        hard_neg_coords = np.argwhere(tissue_mask)

        if len(hard_neg_coords) > 0:
            neg_coords = hard_neg_coords
        else:
            neg_coords = np.argwhere(valid_mask & (mask == 0))
    else:
        neg_coords = np.argwhere(valid_mask & (mask == 0))

    def sample_patches(coords: np.ndarray, count: int) -> list[np.ndarray]:
        if len(coords) == 0:
            return []
        indices = rng.choice(len(coords), size=min(count, len(coords)), replace=len(coords) < count)
        patches = []
        for idx in indices:
            cz, cy, cx = coords[idx]
            patch = volume[
                cz - half:cz + half,
                cy - half:cy + half,
                cx - half:cx + half,
            ]
            patches.append(patch)
        return patches

    pos_patches = sample_patches(pos_coords, num_positive)
    neg_patches = sample_patches(neg_coords, num_negative)

    all_patches = pos_patches + neg_patches
    all_labels = [1.0] * len(pos_patches) + [0.0] * len(neg_patches)

    if len(all_patches) == 0:
        return torch.empty(0, 1, patch_size, patch_size, patch_size), torch.empty(0)

    patches_tensor = torch.tensor(
        np.stack(all_patches), dtype=torch.float32
    ).unsqueeze(1)  # (N, 1, D, H, W)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)

    return patches_tensor, labels_tensor


def normalize_ct(volume: np.ndarray, window_center: float = 40.0, window_width: float = 400.0) -> np.ndarray:
    """Normalize CT volume using a soft-tissue window to [0, 1]."""
    low = window_center - window_width / 2
    high = window_center + window_width / 2
    normalized = (volume.astype(np.float32) - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def train_species(
    species: SpeciesModel,
    volumes: list[np.ndarray],
    masks: list[np.ndarray],
    config: TrainingConfig = TrainingConfig(),
    device: torch.device | str = "cpu",
    rng: Optional[np.random.Generator] = None,
) -> list[float]:
    """Train a species model on labeled volumes.

    Args:
        species: the species to train
        volumes: list of (D, H, W) normalized CT volumes
        masks: list of (D, H, W) binary masks (1 = positive for this species)
        config: training hyperparameters
        device: compute device
        rng: random number generator

    Returns:
        List of per-epoch average loss values
    """
    if rng is None:
        rng = np.random.default_rng(42)

    device = torch.device(device)

    # Extract patches from all volumes
    all_patches = []
    all_labels = []
    for vol, mask in zip(volumes, masks):
        patches, labels = extract_patches(
            vol, mask,
            patch_size=config.patch_size,
            num_positive=config.positive_patches_per_scan,
            num_negative=config.negative_patches_per_scan,
            rng=rng,
        )
        if len(patches) > 0:
            all_patches.append(patches)
            all_labels.append(labels)

    if not all_patches:
        return []

    dataset = PatchDataset(
        patches=torch.cat(all_patches, dim=0),
        labels=torch.cat(all_labels, dim=0),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    species.network.to(device)
    species.network.train()
    optimizer = optim.AdamW(species.network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCELoss()

    epoch_losses = []
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        count = 0
        for patches_batch, labels_batch in loader:
            patches_batch = patches_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            predictions = species.network(patches_batch).squeeze(-1)
            loss = criterion(predictions, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels_batch)
            count += len(labels_batch)

        avg_loss = total_loss / count if count > 0 else 0.0
        epoch_losses.append(avg_loss)

    species.network.eval()
    species.generation += 1
    return epoch_losses
