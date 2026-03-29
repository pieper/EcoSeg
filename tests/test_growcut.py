"""Test and debug GrowCut algorithms on small synthetic data."""

import numpy as np
import torch
import torch.nn.functional as F

from ecoseg.models.growcut_embedding import (
    growcut_intensity, growcut_embedding,
    simulate_paint_strokes, GrowCutConfig,
)


def make_simple_test():
    """Create a small 2-class volume where intensity and embedding
    should give different results.

    Volume: 32x32x32
    - Left half: intensity ~0.3 (class 1 / foreground)
    - Right half: intensity ~0.7 (class 2 / background)
    - But with a "confusing" region in the foreground that has
      background-like intensity (0.7) — intensity GrowCut should fail here,
      embedding GrowCut should succeed if embeddings are discriminative.
    """
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    gt = np.zeros((32, 32, 32), dtype=np.uint8)

    # Left half = foreground (class 1), intensity 0.3
    vol[:, :, :16] = 0.3
    gt[:, :, :16] = 1

    # Right half = background (class 2), intensity 0.7
    vol[:, :, 16:] = 0.7

    # Add noise
    rng = np.random.default_rng(42)
    vol += rng.normal(0, 0.05, vol.shape).astype(np.float32)
    vol = np.clip(vol, 0, 1)

    return vol, gt


def make_embedding_test():
    """Create synthetic embeddings that ARE discriminative even where
    intensity isn't.

    Returns volume, ground truth, and embeddings where:
    - Foreground voxels have embedding [1, 0, 0, 0] (plus noise)
    - Background voxels have embedding [0, 1, 0, 0] (plus noise)
    """
    vol, gt = make_simple_test()
    C = 4
    D, H, W = vol.shape

    emb = np.zeros((C, D, H, W), dtype=np.float32)
    rng = np.random.default_rng(42)

    # Foreground embedding: [1, 0, 0, 0]
    emb[0, gt == 1] = 1.0
    # Background embedding: [0, 1, 0, 0]
    emb[1, gt == 0] = 1.0

    # Add noise
    emb += rng.normal(0, 0.1, emb.shape).astype(np.float32)

    return vol, gt, emb


def test_basic_growcut():
    """Test that intensity GrowCut runs and produces non-trivial output."""
    vol, gt = make_simple_test()
    seeds = simulate_paint_strokes(gt, num_positive=10, num_negative=10,
                                    rng=np.random.default_rng(42))

    vol_t = torch.tensor(vol)
    seeds_t = torch.tensor(seeds, dtype=torch.int32)

    config = GrowCutConfig(max_iterations=100)
    labels, strength = growcut_intensity(vol_t, seeds_t, config)

    labels_np = labels.numpy()
    pred = (labels_np == 1).astype(np.uint8)

    from ecoseg.metrics.scoring import dice_score
    d = dice_score(pred, gt)
    print(f"Intensity GrowCut Dice: {d:.3f}")
    print(f"  Labels unique: {np.unique(labels_np)}")
    print(f"  Pred sum: {pred.sum()}, GT sum: {gt.sum()}")
    assert d > 0.5, f"Intensity GrowCut should work on easy synthetic data, got {d}"


def test_embedding_growcut_differs():
    """Test that embedding GrowCut produces DIFFERENT results from intensity."""
    vol, gt, emb = make_embedding_test()
    seeds = simulate_paint_strokes(gt, num_positive=10, num_negative=10,
                                    rng=np.random.default_rng(42))

    vol_t = torch.tensor(vol)
    emb_t = torch.tensor(emb)
    seeds_t = torch.tensor(seeds, dtype=torch.int32)

    config = GrowCutConfig(max_iterations=100)

    # Run both
    labels_int, strength_int = growcut_intensity(vol_t, seeds_t, config)
    labels_emb, strength_emb = growcut_embedding(emb_t, seeds_t, config)

    pred_int = (labels_int == 1).numpy().astype(np.uint8)
    pred_emb = (labels_emb == 1).numpy().astype(np.uint8)

    from ecoseg.metrics.scoring import dice_score
    dice_int = dice_score(pred_int, gt)
    dice_emb = dice_score(pred_emb, gt)

    print(f"\nIntensity GrowCut Dice: {dice_int:.3f}")
    print(f"Embedding GrowCut Dice: {dice_emb:.3f}")
    print(f"Predictions identical: {np.array_equal(pred_int, pred_emb)}")
    print(f"  Intensity pred sum: {pred_int.sum()}, Embedding pred sum: {pred_emb.sum()}")
    print(f"  Intensity strength range: [{strength_int.min():.4f}, {strength_int.max():.4f}]")
    print(f"  Embedding strength range: [{strength_emb.min():.4f}, {strength_emb.max():.4f}]")

    # Debug: check the prototype computation
    emb_norm = F.normalize(emb_t, dim=0)
    for lbl in [1, 2]:
        mask = seeds_t == lbl
        seed_embs = emb_norm[:, mask]
        proto = F.normalize(seed_embs.mean(dim=1), dim=0)
        print(f"\n  Species {lbl} prototype: {proto.numpy()}")
        print(f"  Species {lbl} num seeds: {mask.sum().item()}")

        # Check affinity at foreground vs background voxels
        affinity = (proto.view(-1, 1, 1, 1) * emb_norm).sum(dim=0)
        fg_affinity = affinity[torch.tensor(gt) == 1].mean()
        bg_affinity = affinity[torch.tensor(gt) == 0].mean()
        print(f"  Affinity at foreground: {fg_affinity:.4f}")
        print(f"  Affinity at background: {bg_affinity:.4f}")

    # Check local similarity
    local_sim_z = (emb_norm[:, :-1] * emb_norm[:, 1:]).sum(dim=0)
    print(f"\n  Local similarity (z-neighbors): mean={local_sim_z.mean():.4f}, "
          f"min={local_sim_z.min():.4f}")

    # Check: at the boundary (z, y, x=15 vs x=16), is local sim lower?
    boundary_sim = (emb_norm[:, :, :, 15] * emb_norm[:, :, :, 16]).sum(dim=0)
    interior_sim = (emb_norm[:, :, :, 7] * emb_norm[:, :, :, 8]).sum(dim=0)
    print(f"  Boundary sim (x=15-16): mean={boundary_sim.mean():.4f}")
    print(f"  Interior sim (x=7-8): mean={interior_sim.mean():.4f}")

    assert not np.array_equal(pred_int, pred_emb), "Results should differ!"


def test_seeds_are_preserved():
    """Verify seed voxels are never overwritten."""
    vol, gt = make_simple_test()
    seeds = simulate_paint_strokes(gt, num_positive=20, num_negative=20,
                                    rng=np.random.default_rng(42))
    seeds_t = torch.tensor(seeds, dtype=torch.int32)
    vol_t = torch.tensor(vol)

    config = GrowCutConfig(max_iterations=50)
    labels, _ = growcut_intensity(vol_t, seeds_t, config)

    # Every seed should retain its original label
    seed_mask = seeds > 0
    assert (labels.numpy()[seed_mask] == seeds[seed_mask]).all(), "Seeds were overwritten!"
    print("Seeds preserved: OK")


if __name__ == "__main__":
    print("=== Test 1: Basic intensity GrowCut ===")
    test_basic_growcut()

    print("\n=== Test 2: Seeds preserved ===")
    test_seeds_are_preserved()

    print("\n=== Test 3: Embedding vs Intensity ===")
    test_embedding_growcut_differs()
