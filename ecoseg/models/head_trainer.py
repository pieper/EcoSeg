"""Trainer for species decoder heads on frozen embeddings.

Since the encoder is frozen and embeddings are pre-computed, training
a species head is extremely fast — it only backpropagates through a
few 1x1x1 conv layers operating on cached feature vectors.

Training extracts embedding vectors at labeled voxel positions and
trains the head as a per-voxel binary classifier.
"""

import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ecoseg.models.ecosegnet import SpeciesHead

logger = logging.getLogger(__name__)


@dataclass
class HeadTrainingConfig:
    num_epochs: int = 30
    batch_size: int = 4096  # Can be large since we're training on feature vectors
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    samples_per_scan: int = 2000  # Positive + negative voxels sampled per scan
    hard_negatives: bool = True


def extract_labeled_features(
    embeddings: np.ndarray,
    mask: np.ndarray,
    num_samples: int = 2000,
    hard_negatives: bool = True,
    volume: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract feature vectors at labeled voxel positions.

    Instead of extracting 32^3 patches, we just sample individual
    voxels' embedding vectors — each is a (feature_dim,) vector.

    Args:
        embeddings: (feature_dim, D, H, W) float16/32 array
        mask: (D, H, W) binary mask (1 = positive for this species)
        num_samples: total samples (split 50/50 positive/negative)
        hard_negatives: if True, sample negatives from tissue regions
        volume: (D, H, W) volume needed for hard negative mining
        rng: random number generator

    Returns:
        features: (N, feature_dim) float32 tensor
        labels: (N,) float32 tensor of 0.0/1.0
    """
    if rng is None:
        rng = np.random.default_rng()

    feature_dim = embeddings.shape[0]
    num_pos = num_samples // 2
    num_neg = num_samples - num_pos

    pos_coords = np.argwhere(mask > 0)
    neg_candidates = mask == 0

    if hard_negatives and volume is not None and len(pos_coords) > 0:
        pos_values = volume[mask > 0]
        lo = np.percentile(pos_values, 10)
        hi = np.percentile(pos_values, 90)
        neg_candidates = neg_candidates & (volume >= lo) & (volume <= hi)

    neg_coords = np.argwhere(neg_candidates)

    def _sample(coords, count):
        if len(coords) == 0:
            return np.empty((0, feature_dim), dtype=np.float32)
        idx = rng.choice(len(coords), size=min(count, len(coords)),
                         replace=len(coords) < count)
        sampled = coords[idx]
        # Extract feature vectors at these positions
        feats = embeddings[:, sampled[:, 0], sampled[:, 1], sampled[:, 2]]
        return feats.T.astype(np.float32)  # (N, feature_dim)

    pos_feats = _sample(pos_coords, num_pos)
    neg_feats = _sample(neg_coords, num_neg)

    if len(pos_feats) == 0 and len(neg_feats) == 0:
        return torch.empty(0, feature_dim), torch.empty(0)

    all_feats = np.concatenate([pos_feats, neg_feats], axis=0)
    all_labels = np.concatenate([
        np.ones(len(pos_feats), dtype=np.float32),
        np.zeros(len(neg_feats), dtype=np.float32),
    ])

    return torch.tensor(all_feats), torch.tensor(all_labels)


def train_species_head(
    head: SpeciesHead,
    embeddings_list: list[np.ndarray],
    masks: list[np.ndarray],
    config: HeadTrainingConfig = HeadTrainingConfig(),
    device: torch.device | str = "cpu",
    volumes: Optional[list[np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[float]:
    """Train a species head on pre-computed embeddings.

    Args:
        head: the species decoder head to train
        embeddings_list: list of (feature_dim, D, H, W) arrays
        masks: list of (D, H, W) binary masks
        config: training hyperparameters
        device: compute device
        volumes: optional list of volumes for hard negative mining
        rng: random number generator

    Returns:
        List of per-epoch average loss values
    """
    if rng is None:
        rng = np.random.default_rng(42)

    device = torch.device(device)
    t0 = time.time()

    # Extract feature vectors at labeled positions from all scans
    all_feats = []
    all_labels = []
    for i, (emb, mask) in enumerate(zip(embeddings_list, masks)):
        vol = volumes[i] if volumes is not None else None
        feats, labels = extract_labeled_features(
            emb, mask,
            num_samples=config.samples_per_scan,
            hard_negatives=config.hard_negatives,
            volume=vol,
            rng=rng,
        )
        if len(feats) > 0:
            all_feats.append(feats)
            all_labels.append(labels)

    if not all_feats:
        return []

    # Concatenate and move to GPU
    feats_t = torch.cat(all_feats, dim=0).to(device)
    labels_t = torch.cat(all_labels, dim=0).to(device)
    n_samples = len(labels_t)

    logger.info(
        f"Training head on {n_samples} feature vectors "
        f"({feats_t.shape[1]}-dim, {(labels_t == 1).sum()} pos, "
        f"{(labels_t == 0).sum()} neg)"
    )

    # The head expects (batch, feature_dim, D, H, W) but we have
    # individual voxels as (N, feature_dim). Reshape to (N, feature_dim, 1, 1, 1)
    feats_5d = feats_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # Output will be (N, 1, 1, 1, 1) → squeeze to (N,)

    head.to(device)
    head.train()
    optimizer = optim.AdamW(head.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    criterion = nn.BCELoss()

    epoch_losses = []
    for epoch in range(config.num_epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        count = 0

        for start in range(0, n_samples, config.batch_size):
            idx = perm[start:start + config.batch_size]
            batch_feats = feats_5d[idx]
            batch_labels = labels_t[idx]

            optimizer.zero_grad()
            preds = head(batch_feats).squeeze()
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            count += len(batch_labels)

        avg_loss = total_loss / count if count > 0 else 0.0
        epoch_losses.append(avg_loss)

    head.eval()
    elapsed = time.time() - t0
    logger.info(f"Head training: {elapsed:.1f}s, final loss={epoch_losses[-1]:.4f}")

    return epoch_losses


def train_species_head_from_features(
    head: SpeciesHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    config: HeadTrainingConfig = HeadTrainingConfig(),
    device: torch.device | str = "cpu",
) -> list[float]:
    """Train a species head directly from pre-extracted feature vectors.

    This is the memory-efficient path: features have already been extracted
    at labeled voxel positions from the zarr cache, so no full embedding
    volumes are in memory.

    Args:
        head: the species decoder head to train
        features: (N, feature_dim) float32 tensor
        labels: (N,) float32 tensor of 0.0/1.0
        config: training hyperparameters
        device: compute device

    Returns:
        List of per-epoch average loss values
    """
    device = torch.device(device)
    t0 = time.time()

    feats_t = features.to(device)
    labels_t = labels.to(device)
    n_samples = len(labels_t)

    # Reshape for conv1x1x1: (N, feature_dim) -> (N, feature_dim, 1, 1, 1)
    feats_5d = feats_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    head.to(device)
    head.train()
    optimizer = optim.AdamW(head.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    criterion = nn.BCELoss()

    epoch_losses = []
    for epoch in range(config.num_epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        count = 0

        for start in range(0, n_samples, config.batch_size):
            idx = perm[start:start + config.batch_size]
            batch_feats = feats_5d[idx]
            batch_labels = labels_t[idx]

            optimizer.zero_grad()
            preds = head(batch_feats).squeeze()
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            count += len(batch_labels)

        avg_loss = total_loss / count if count > 0 else 0.0
        epoch_losses.append(avg_loss)

    head.eval()
    elapsed = time.time() - t0
    logger.info(f"Head training: {elapsed:.1f}s, final loss={epoch_losses[-1]:.4f}")

    return epoch_losses
