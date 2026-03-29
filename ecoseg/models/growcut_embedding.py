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
                logger.debug(
                    f"GrowCut ({method_name}) converged at iteration {iteration + 1}"
                )
                break
        else:
            no_change_count = 0

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


def growcut_learned_fitness(
    volume: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig = GrowCutConfig(),
    patch_size: int = 16,
    stride: int = 4,
    num_epochs: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GrowCut using a learned fitness from raw CT patches.

    1. Trains a small CNN on patches at seed locations (foreground vs background)
    2. Computes a per-voxel fitness map by sliding the classifier over the volume
    3. Uses fitness_map as the GrowCut fitness: species propagate into voxels
       that the classifier scores highly for them

    The fitness combines the learned score with local intensity similarity
    so that propagation still respects intensity boundaries.

    Args:
        volume: (D, H, W) normalized CT on GPU
        seeds: (D, H, W) int tensor — 1=foreground, 2=background
        config: GrowCut iteration config
        patch_size: CNN patch size
        stride: fitness map stride
        num_epochs: classifier training epochs
    """
    from ecoseg.models.learned_fitness import train_patch_classifier, compute_fitness_map

    device = volume.device

    # Train classifier from seed patches
    model = train_patch_classifier(volume, seeds, patch_size, num_epochs, device)

    # Compute per-voxel fitness map (P(foreground) for each voxel)
    fg_fitness = compute_fitness_map(model, volume, patch_size, stride)
    # Background fitness = 1 - foreground fitness
    bg_fitness = 1.0 - fg_fitness

    # Build per-species fitness maps
    species_fitness = {1: fg_fitness, 2: bg_fitness}

    # GrowCut fitness = species_score * local_intensity_similarity
    # This combines the learned "what tissue type is this?" with
    # the classic "is there a smooth boundary?" signal
    def learned_fitness_fn(dz, dy, dx):
        neighbor_vol = torch.roll(volume, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))
        local_sim = (1.0 - torch.abs(volume - neighbor_vol)).clamp(0, 1)
        # We need to know which species is attacking, but _growcut_core
        # doesn't pass that info. Use a combined fitness that favors
        # the locally-dominant species.
        # For each voxel: max species fitness * local similarity
        max_species = torch.max(fg_fitness, bg_fitness)
        return local_sim * (0.3 + 0.7 * max_species)

    labels = seeds.clone()
    strength = (seeds > 0).float()
    return _growcut_core(learned_fitness_fn, labels, strength, seeds, config, "learned_fitness")


def growcut_learned_fitness_per_species(
    volume: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig = GrowCutConfig(),
    patch_size: int = 16,
    stride: int = 4,
    num_epochs: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GrowCut where each species uses its own learned fitness.

    Each species gets a fitness map from the CNN — foreground seeds
    use fg_fitness, background seeds use bg_fitness. A species can
    only conquer voxels where its classifier scores them highly.

    This is the core "species adapted to its habitat" concept:
    the paint strokes define the species' DNA, the CNN learns what
    habitat it thrives in, and GrowCut simulates the competition.
    """
    from ecoseg.models.learned_fitness import train_patch_classifier, compute_fitness_map

    device = volume.device
    D, H, W = volume.shape

    # Train and compute fitness maps
    model = train_patch_classifier(volume, seeds, patch_size, num_epochs, device)
    fg_fitness = compute_fitness_map(model, volume, patch_size, stride)
    bg_fitness = 1.0 - fg_fitness

    species_fitness = {1: fg_fitness, 2: bg_fitness}

    # Per-species GrowCut
    labels = seeds.clone()
    strength = (seeds > 0).float()
    seed_mask = seeds > 0
    no_change_count = 0

    offsets = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

    for iteration in range(config.max_iterations):
        changed = 0
        new_labels = labels.clone()
        new_strength = strength.clone()

        for dz, dy, dx in offsets:
            neighbor_vol = torch.roll(volume, shifts=(-dz,-dy,-dx), dims=(0,1,2))
            neighbor_labels = torch.roll(labels, shifts=(-dz,-dy,-dx), dims=(0,1,2))
            neighbor_strength = torch.roll(strength, shifts=(-dz,-dy,-dx), dims=(0,1,2))

            boundary_mask = torch.ones(D, H, W, device=device, dtype=torch.bool)
            if dz == -1: boundary_mask[0,:,:] = False
            elif dz == 1: boundary_mask[-1,:,:] = False
            if dy == -1: boundary_mask[:,0,:] = False
            elif dy == 1: boundary_mask[:,-1,:] = False
            if dx == -1: boundary_mask[:,:,0] = False
            elif dx == 1: boundary_mask[:,:,-1] = False

            # Local intensity similarity (boundary detection)
            local_sim = (1.0 - torch.abs(volume - neighbor_vol)).clamp(0, 1)

            for lbl_val, sp_fitness in species_fitness.items():
                # Fitness = local similarity * species-specific learned score
                fitness = local_sim * (0.3 + 0.7 * sp_fitness)
                attack = fitness * neighbor_strength

                wins = (
                    boundary_mask
                    & (neighbor_labels == lbl_val)
                    & ((attack > new_strength) | ((new_labels == 0) & (attack > 0)))
                )

                new_labels[wins] = lbl_val
                new_strength[wins] = attack[wins]
                changed += wins.sum().item()

        labels = new_labels
        strength = new_strength
        labels[seed_mask] = seeds[seed_mask]
        strength[seed_mask] = 1.0

        if changed == 0:
            no_change_count += 1
            if no_change_count >= config.stop_after_no_change:
                logger.debug(f"GrowCut (learned_fitness) converged at iteration {iteration + 1}")
                break
        else:
            no_change_count = 0

    return labels, strength


def train_fitness_classifier(
    embeddings: torch.Tensor,
    seeds: torch.Tensor,
    num_epochs: int = 50,
) -> 'FitnessClassifier':
    """Train a per-species fitness classifier from seed voxel embeddings.

    For each species, trains a small MLP that maps an embedding vector
    to a fitness score (0-1). The classifier learns the nonlinear
    boundary in embedding space that separates this species from others.

    Args:
        embeddings: (C, D, H, W) per-voxel embeddings (on GPU)
        seeds: (D, H, W) int tensor -- 0=unlabeled, 1+=species label

    Returns:
        FitnessClassifier that can score any voxel's embedding for each species
    """
    device = embeddings.device
    C = embeddings.shape[0]

    unique_labels = seeds.unique()
    unique_labels = unique_labels[unique_labels > 0]

    # Collect training data: embeddings at all seed positions
    all_coords = {}
    for lbl in unique_labels:
        lbl_val = lbl.item()
        mask = seeds == lbl_val
        all_coords[lbl_val] = torch.argwhere(mask)  # (N, 3)

    # Train one classifier per species (one-vs-rest)
    classifiers = {}
    for lbl_val in [l.item() for l in unique_labels]:
        # Positive: this species' seeds. Negative: all other species' seeds.
        pos_coords = all_coords[lbl_val]
        neg_coords = torch.cat([all_coords[l] for l in all_coords if l != lbl_val])

        pos_embs = embeddings[:, pos_coords[:, 0], pos_coords[:, 1], pos_coords[:, 2]].T  # (Npos, C)
        neg_embs = embeddings[:, neg_coords[:, 0], neg_coords[:, 1], neg_coords[:, 2]].T  # (Nneg, C)

        X = torch.cat([pos_embs, neg_embs], dim=0)
        y = torch.cat([
            torch.ones(len(pos_embs), device=device),
            torch.zeros(len(neg_embs), device=device),
        ])

        # Small MLP: C -> 64 -> 32 -> 1
        mlp = torch.nn.Sequential(
            torch.nn.Linear(C, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        ).to(device)

        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        for epoch in range(num_epochs):
            perm = torch.randperm(len(X), device=device)
            for start in range(0, len(X), 256):
                idx = perm[start:start + 256]
                pred = mlp(X[idx]).squeeze(-1)
                loss = criterion(pred, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        mlp.eval()
        classifiers[lbl_val] = mlp

        # Log accuracy
        with torch.no_grad():
            pred_all = mlp(X).squeeze(-1)
            acc = ((pred_all > 0.5) == (y > 0.5)).float().mean()
            logger.debug(f"Species {lbl_val} classifier: {len(pos_embs)} pos, "
                         f"{len(neg_embs)} neg, accuracy={acc:.3f}")

    return FitnessClassifier(classifiers, C)


class FitnessClassifier:
    """Learned fitness function for GrowCut.

    Contains per-species MLPs that score each voxel's embedding.
    Used as the fitness_fn in growcut_embedding.
    """

    def __init__(self, classifiers: dict, feature_dim: int):
        self.classifiers = classifiers  # {label_int: nn.Module}
        self.feature_dim = feature_dim
        self._score_cache = {}

    def precompute_scores(self, embeddings: torch.Tensor) -> dict[int, torch.Tensor]:
        """Pre-compute fitness scores for all voxels for each species.

        Args:
            embeddings: (C, D, H, W) on GPU

        Returns:
            Dict mapping species label to (D, H, W) fitness scores in [0, 1]
        """
        C, D, H, W = embeddings.shape
        flat = embeddings.reshape(C, -1).T  # (N, C)

        scores = {}
        for lbl_val, mlp in self.classifiers.items():
            with torch.no_grad():
                # Process in chunks to avoid OOM
                chunk_scores = []
                for start in range(0, len(flat), 100000):
                    chunk = flat[start:start + 100000]
                    chunk_scores.append(mlp(chunk).squeeze(-1))
                s = torch.cat(chunk_scores).reshape(D, H, W)
                scores[lbl_val] = s

        return scores


def growcut_learned(
    embeddings: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig = GrowCutConfig(),
    num_classifier_epochs: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GrowCut using a learned fitness classifier.

    1. Trains a small MLP per species on seed voxel embeddings
    2. Pre-computes per-voxel fitness scores for each species
    3. Runs GrowCut where fitness = classifier_score(center_voxel)
       for the attacking species, modulated by local embedding similarity

    The classifier learns what each species "looks like" in embedding
    space from the paint strokes, then the GrowCut propagates labels
    using that learned fitness.
    """
    device = embeddings.device
    emb_norm = F.normalize(embeddings, dim=0)

    # Step 1: Train classifiers from seed embeddings
    classifier = train_fitness_classifier(embeddings, seeds, num_classifier_epochs)

    # Step 2: Pre-compute per-voxel fitness for each species
    species_scores = classifier.precompute_scores(embeddings)

    # Step 3: GrowCut with learned fitness
    # Fitness = learned_score * local_similarity
    # learned_score: how much does this voxel match the attacking species?
    # local_similarity: is there a smooth path from neighbor to center?
    def learned_fitness_fn(dz, dy, dx):
        # This gets called inside _growcut_core, but we need to know
        # which species is attacking. Since _growcut_core doesn't pass
        # species info, we use a different approach: pre-combine the
        # scores into a single fitness map based on the attacking neighbor's label.
        # We'll compute fitness as local_sim * max_species_score, which
        # favors propagation toward voxels that match ANY species well.
        neighbor_emb = torch.roll(emb_norm, shifts=(-dz, -dy, -dx), dims=(1, 2, 3))
        local_sim = (emb_norm * neighbor_emb).sum(dim=0).clamp(0, 1)

        # Use the max species score as fitness boost
        all_scores = torch.stack(list(species_scores.values()), dim=0)
        max_score = all_scores.max(dim=0).values  # (D, H, W)

        return local_sim * (0.3 + 0.7 * max_score)

    labels = seeds.clone()
    strength = (seeds > 0).float()
    return _growcut_core(learned_fitness_fn, labels, strength, seeds, config, "learned")


def growcut_learned_per_species(
    embeddings: torch.Tensor,
    seeds: torch.Tensor,
    config: GrowCutConfig = GrowCutConfig(),
    num_classifier_epochs: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GrowCut where each species uses its own learned fitness.

    Unlike the basic growcut_learned, this version gives each species
    its own fitness function — the classifier score for THAT species.
    A lymph node seed can only propagate to voxels that the LN classifier
    scores highly, while a background seed uses the BG classifier.

    This requires a modified GrowCut core that handles per-species fitness.
    """
    device = embeddings.device
    C, D, H, W = embeddings.shape
    emb_norm = F.normalize(embeddings, dim=0)

    # Train classifiers
    classifier = train_fitness_classifier(embeddings, seeds, num_classifier_epochs)
    species_scores = classifier.precompute_scores(embeddings)

    unique_labels = seeds.unique()
    unique_labels = unique_labels[unique_labels > 0]

    # Initialize
    labels = seeds.clone()
    strength = (seeds > 0).float()
    seed_mask = seeds > 0
    no_change_count = 0

    offsets = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

    for iteration in range(config.max_iterations):
        changed = 0
        new_labels = labels.clone()
        new_strength = strength.clone()

        for dz, dy, dx in offsets:
            neighbor_emb = torch.roll(emb_norm, shifts=(-dz,-dy,-dx), dims=(1,2,3))
            neighbor_labels = torch.roll(labels, shifts=(-dz,-dy,-dx), dims=(0,1,2))
            neighbor_strength = torch.roll(strength, shifts=(-dz,-dy,-dx), dims=(0,1,2))

            boundary_mask = torch.ones(D, H, W, device=device, dtype=torch.bool)
            if dz == -1: boundary_mask[0,:,:] = False
            elif dz == 1: boundary_mask[-1,:,:] = False
            if dy == -1: boundary_mask[:,0,:] = False
            elif dy == 1: boundary_mask[:,-1,:] = False
            if dx == -1: boundary_mask[:,:,0] = False
            elif dx == 1: boundary_mask[:,:,-1] = False

            # Local embedding similarity (boundary detection)
            local_sim = (emb_norm * neighbor_emb).sum(dim=0).clamp(0, 1)

            for lbl in unique_labels:
                lbl_val = lbl.item()
                # This species' learned fitness at the TARGET voxel
                species_fit = species_scores[lbl_val]
                # Combined: smooth path AND target matches this species
                fitness = local_sim * (0.3 + 0.7 * species_fit)

                attack = fitness * neighbor_strength

                wins = (
                    boundary_mask
                    & (neighbor_labels == lbl_val)
                    & ((attack > new_strength) | ((new_labels == 0) & (attack > 0)))
                )

                new_labels[wins] = lbl_val
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
                    f"GrowCut (learned) converged at iteration {iteration + 1}, "
                    f"labeled={n_labeled}/{total_voxels}"
                )
                break
        else:
            no_change_count = 0

        if (iteration + 1) % 100 == 0:
            n_labeled = (labels > 0).sum().item()
            logger.info(
                f"GrowCut (learned) iter {iteration+1}: "
                f"changed={changed}, labeled={n_labeled}/{total_voxels}"
            )

    return labels, strength


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
