"""Learned fitness from paint strokes on raw CT data.

Trains a small patch-based CNN directly on the user's seed voxels
to learn what each species' "habitat" looks like in the raw image.
No pre-trained encoder needed — the model learns from scratch what
distinguishes foreground from background in THIS scan.

The trained model produces a per-voxel fitness map that drives
GrowCut propagation.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class PatchClassifier(nn.Module):
    """Tiny CNN that classifies small 3D patches.

    Input: (batch, 1, ps, ps, ps) — raw CT patch
    Output: (batch,) — probability of belonging to species in [0, 1]

    ~30K parameters, trains in <1 second on GPU.
    """

    def __init__(self, patch_size: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_patch_classifier(
    volume: torch.Tensor,
    seeds: torch.Tensor,
    patch_size: int = 16,
    num_epochs: int = 100,
    device: torch.device = None,
) -> PatchClassifier:
    """Train a patch classifier from seed voxels on raw CT.

    Extracts patches centered on each seed voxel, trains a binary
    classifier (foreground vs background), returns the trained model.

    Args:
        volume: (D, H, W) normalized CT volume on GPU
        seeds: (D, H, W) int tensor — 1=foreground, 2=background
        patch_size: side length of cubic patches
        num_epochs: training epochs
        device: compute device

    Returns:
        Trained PatchClassifier
    """
    if device is None:
        device = volume.device

    half = patch_size // 2
    D, H, W = volume.shape

    # Collect seed coordinates with valid patch bounds
    fg_coords = torch.argwhere(seeds == 1)
    bg_coords = torch.argwhere(seeds == 2)

    def extract_valid_patches(coords):
        patches = []
        for z, y, x in coords:
            z, y, x = z.item(), y.item(), x.item()
            if (z >= half and z < D - half and
                y >= half and y < H - half and
                x >= half and x < W - half):
                patch = volume[z-half:z+half, y-half:y+half, x-half:x+half]
                patches.append(patch)
        if not patches:
            return torch.empty(0, 1, patch_size, patch_size, patch_size, device=device)
        return torch.stack(patches).unsqueeze(1)  # (N, 1, ps, ps, ps)

    fg_patches = extract_valid_patches(fg_coords)
    bg_patches = extract_valid_patches(bg_coords)

    if len(fg_patches) == 0 or len(bg_patches) == 0:
        logger.warning("Not enough valid seed patches for training")
        model = PatchClassifier(patch_size).to(device)
        model.eval()
        return model

    X = torch.cat([fg_patches, bg_patches], dim=0)
    y = torch.cat([
        torch.ones(len(fg_patches), device=device),
        torch.zeros(len(bg_patches), device=device),
    ])

    model = PatchClassifier(patch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    n = len(y)
    batch_size = min(256, n)

    for epoch in range(num_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(X[idx])
            loss = criterion(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # Log accuracy
    with torch.no_grad():
        pred_all = model(X)
        acc = ((pred_all > 0.5) == (y > 0.5)).float().mean()
        logger.debug(f"Patch classifier: {len(fg_patches)} fg, {len(bg_patches)} bg, "
                     f"accuracy={acc:.3f}")

    return model


def compute_fitness_map(
    model: PatchClassifier,
    volume: torch.Tensor,
    patch_size: int = 16,
    stride: int = 4,
) -> torch.Tensor:
    """Compute per-voxel fitness map using the trained classifier.

    Slides the classifier over the volume and averages overlapping
    predictions. Returns a smooth fitness map in [0, 1].

    Args:
        model: trained PatchClassifier
        volume: (D, H, W) normalized CT on GPU
        patch_size: must match training patch size
        stride: sliding window stride (smaller = smoother but slower)

    Returns:
        (D, H, W) fitness map in [0, 1] on GPU
    """
    device = volume.device
    D, H, W = volume.shape
    half = patch_size // 2

    fitness_sum = torch.zeros(D, H, W, device=device)
    count = torch.zeros(D, H, W, device=device)

    # Collect all valid patch positions
    z_starts = range(half, D - half, stride)
    y_starts = range(half, H - half, stride)
    x_starts = range(half, W - half, stride)

    # Process in batches for GPU efficiency
    positions = []
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                positions.append((z, y, x))

    batch_size = 512
    with torch.no_grad():
        for batch_start in range(0, len(positions), batch_size):
            batch_pos = positions[batch_start:batch_start + batch_size]

            patches = torch.stack([
                volume[z-half:z+half, y-half:y+half, x-half:x+half]
                for z, y, x in batch_pos
            ]).unsqueeze(1)  # (B, 1, ps, ps, ps)

            scores = model(patches)  # (B,)

            for j, (z, y, x) in enumerate(batch_pos):
                fitness_sum[z, y, x] += scores[j]
                count[z, y, x] += 1.0

    # For voxels not covered by any patch center, use nearest value
    covered = count > 0
    if covered.all():
        return fitness_sum / count

    # Fill uncovered edges by extending the nearest covered value
    fitness = torch.where(covered, fitness_sum / count, torch.zeros_like(fitness_sum))

    # Simple fill: dilate covered values into uncovered regions
    for _ in range(patch_size):
        for dz, dy, dx in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            shifted = torch.roll(fitness, shifts=(dz, dy, dx), dims=(0, 1, 2))
            shifted_covered = torch.roll(covered, shifts=(dz, dy, dx), dims=(0, 1, 2))
            fill_mask = ~covered & shifted_covered
            fitness[fill_mask] = shifted[fill_mask]
            covered = covered | fill_mask
        if covered.all():
            break

    return fitness
