"""Detailed GrowCut debugging — trace what happens iteration by iteration."""

import torch
import numpy as np
from ecoseg.models.growcut_embedding import growcut_intensity, GrowCutConfig


def test_growcut_spread():
    """Test that GrowCut actually spreads from seeds in a uniform volume."""
    # Simple uniform volume — everything should be conquered
    vol = torch.ones(16, 16, 16) * 0.5

    # One seed of each class
    seeds = torch.zeros(16, 16, 16, dtype=torch.int32)
    seeds[8, 8, 4] = 1   # foreground seed on left
    seeds[8, 8, 12] = 2  # background seed on right

    config = GrowCutConfig(max_iterations=50, convergence_threshold=0.0)

    # Run with verbose tracking
    labels = seeds.clone()
    strength = (seeds > 0).float()

    offsets = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

    for iteration in range(20):
        changed = 0
        new_labels = labels.clone()
        new_strength = strength.clone()

        for dz, dy, dx in offsets:
            neighbor_vol = torch.roll(vol, shifts=(-dz,-dy,-dx), dims=(0,1,2))
            neighbor_labels = torch.roll(labels, shifts=(-dz,-dy,-dx), dims=(0,1,2))
            neighbor_strength = torch.roll(strength, shifts=(-dz,-dy,-dx), dims=(0,1,2))

            boundary_mask = torch.ones(16,16,16, dtype=torch.bool)
            if dz == -1: boundary_mask[-1,:,:] = False
            elif dz == 1: boundary_mask[0,:,:] = False
            if dy == -1: boundary_mask[:,-1,:] = False
            elif dy == 1: boundary_mask[:,0,:] = False
            if dx == -1: boundary_mask[:,:,-1] = False
            elif dx == 1: boundary_mask[:,:,0] = False

            fitness = 1.0 - torch.abs(vol - neighbor_vol)
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

        # Preserve seeds
        seed_mask = seeds > 0
        labels[seed_mask] = seeds[seed_mask]
        strength[seed_mask] = 1.0

        n_labeled = (labels > 0).sum().item()
        n_total = 16**3
        n_fg = (labels == 1).sum().item()
        n_bg = (labels == 2).sum().item()
        print(f"Iter {iteration:2d}: changed={changed:5d}, "
              f"labeled={n_labeled}/{n_total} ({n_labeled/n_total*100:.1f}%), "
              f"fg={n_fg}, bg={n_bg}, "
              f"strength range=[{strength[labels>0].min():.4f}, {strength[labels>0].max():.4f}]"
              if n_labeled > 0 else
              f"Iter {iteration:2d}: changed={changed}, labeled=0")

    # Check final state
    n_unlabeled = (labels == 0).sum().item()
    print(f"\nFinal: {n_unlabeled} unlabeled out of {16**3}")
    assert n_unlabeled == 0, f"GrowCut should label everything in a uniform volume! {n_unlabeled} unlabeled"


def test_growcut_two_regions():
    """Test GrowCut on a volume with two distinct intensity regions."""
    vol = torch.zeros(16, 16, 16)
    vol[:, :, :8] = 0.3   # left region
    vol[:, :, 8:] = 0.7   # right region

    seeds = torch.zeros(16, 16, 16, dtype=torch.int32)
    seeds[8, 8, 4] = 1    # seed in left region
    seeds[8, 8, 12] = 2   # seed in right region

    config = GrowCutConfig(max_iterations=50)
    labels, strength = growcut_intensity(vol, seeds, config)

    n_unlabeled = (labels == 0).sum().item()
    n_fg = (labels == 1).sum().item()
    n_bg = (labels == 2).sum().item()

    print(f"\nTwo regions test:")
    print(f"  Unlabeled: {n_unlabeled}")
    print(f"  Foreground (label 1): {n_fg}")
    print(f"  Background (label 2): {n_bg}")
    print(f"  Strength range: [{strength.min():.4f}, {strength.max():.4f}]")

    from ecoseg.metrics.scoring import dice_score
    gt = torch.zeros(16, 16, 16, dtype=torch.uint8)
    gt[:, :, :8] = 1
    d = dice_score((labels == 1).numpy().astype(np.uint8), gt.numpy())
    print(f"  Dice: {d:.3f}")

    assert n_unlabeled == 0, f"Should label everything! {n_unlabeled} unlabeled"


if __name__ == "__main__":
    print("=== Test: Uniform volume spread ===")
    test_growcut_spread()

    print("\n=== Test: Two region separation ===")
    test_growcut_two_regions()
