"""Integration test: full species train -> infer -> score pipeline on synthetic data.

Generates visualization plots in tests/output/ for review.
"""

import numpy as np
from pathlib import Path

from ecoseg.models.species import SpeciesRegistry
from ecoseg.models.trainer import TrainingConfig, train_species, normalize_ct
from ecoseg.models.inference import InferenceConfig, infer_volume
from ecoseg.metrics.scoring import score_segmentation


OUTPUT_DIR = Path(__file__).parent / "output"


def _make_two_class_volume(shape=(96, 96, 96), seed=42):
    """Synthetic volume with a bright sphere (lymph node) in a dark background.

    Uses a larger volume (96^3) so there are enough distinct 32^3 patches
    for the model to learn from, and a clear intensity separation.
    """
    rng = np.random.default_rng(seed)
    # Background: low intensity with some noise
    volume = rng.normal(loc=-50, scale=20, size=shape).astype(np.float32)

    # Sphere: high intensity, clearly separable
    center = np.array(shape) // 2
    radius = min(shape) // 4  # Larger sphere = more positive patches
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    mask = (dist <= radius).astype(np.uint8)
    volume[mask > 0] = rng.normal(loc=200, scale=20, size=mask.sum()).astype(np.float32)

    return volume, mask


def _save_plots(
    vol, gt_mask, prediction, fitness_map, ln_losses, bg_losses, score, species_names
):
    """Generate review plots and save to OUTPUT_DIR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mid_z = vol.shape[0] // 2

    # --- Figure 1: Training loss curves ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(ln_losses, label="Lymph node species", color="red")
    ax.plot(bg_losses, label="Background species", color="blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Species Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Axial slice comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: Volume, Ground truth, Prediction
    axes[0, 0].imshow(vol[mid_z], cmap="gray")
    axes[0, 0].set_title("CT Volume (axial)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(vol[mid_z], cmap="gray")
    axes[0, 1].contour(gt_mask[mid_z], levels=[0.5], colors=["lime"], linewidths=1.5)
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(vol[mid_z], cmap="gray")
    axes[0, 2].contour(prediction[mid_z], levels=[0.5], colors=["red"], linewidths=1.5)
    axes[0, 2].contour(gt_mask[mid_z], levels=[0.5], colors=["lime"], linewidths=1.0, linestyles="dashed")
    axes[0, 2].set_title(f"Prediction (Dice={score.dice:.3f})")
    axes[0, 2].axis("off")

    # Row 2: Fitness maps for each species + difference
    ln_idx = species_names.index("lymph_node")
    bg_idx = species_names.index("background")

    im0 = axes[1, 0].imshow(fitness_map[ln_idx, mid_z], cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Lymph Node Fitness")
    axes[1, 0].axis("off")
    plt.colorbar(im0, ax=axes[1, 0], fraction=0.046)

    im1 = axes[1, 1].imshow(fitness_map[bg_idx, mid_z], cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title("Background Fitness")
    axes[1, 1].axis("off")
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)

    diff = fitness_map[ln_idx, mid_z] - fitness_map[bg_idx, mid_z]
    im2 = axes[1, 2].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 2].set_title("Fitness Difference (LN - BG)")
    axes[1, 2].axis("off")
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    fig.suptitle(
        f"EcoSeg Synthetic Test — Dice: {score.dice:.3f}, ASSD: {score.assd:.2f}mm",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "segmentation_results.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: 3D fitness profile through center ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    center_y, center_x = vol.shape[1] // 2, vol.shape[2] // 2
    ax.plot(
        fitness_map[ln_idx, :, center_y, center_x],
        label="Lymph node", color="red",
    )
    ax.plot(
        fitness_map[bg_idx, :, center_y, center_x],
        label="Background", color="blue",
    )
    ax.axvline(mid_z, color="gray", linestyle="--", alpha=0.5, label="Center slice")
    ax.set_xlabel("Slice (z)")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Profile Through Volume Center")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fitness_profile.png", dpi=150)
    plt.close(fig)


class TestFullPipeline:
    def test_train_infer_score_with_plots(self):
        """Train two species on synthetic data, infer, score, and generate plots."""
        device = "cpu"
        registry = SpeciesRegistry(device=device)
        ln_species = registry.add_species("lymph_node")
        bg_species = registry.add_species("background")

        # Create training data
        vol_raw, gt_mask = _make_two_class_volume()
        vol = normalize_ct(vol_raw, window_center=50, window_width=300)

        config = TrainingConfig(
            patch_size=32, batch_size=32, num_epochs=50,
            learning_rate=1e-3,
            positive_patches_per_scan=300, negative_patches_per_scan=300,
        )

        # Train lymph node species on positive regions
        ln_losses = train_species(
            ln_species, [vol], [gt_mask],
            config=config, device=device,
        )

        # Train background species on negative regions
        bg_mask = (gt_mask == 0).astype(np.uint8)
        bg_losses = train_species(
            bg_species, [vol], [bg_mask],
            config=config, device=device,
        )

        assert len(ln_losses) == 50
        assert len(bg_losses) == 50

        # Infer on a new synthetic volume (same distribution, different seed)
        test_vol_raw, test_gt = _make_two_class_volume(seed=99)
        test_vol = normalize_ct(test_vol_raw, window_center=50, window_width=300)

        inf_config = InferenceConfig(patch_size=32, stride=16, batch_size=32)
        labels, fitness_map = infer_volume(registry, test_vol, inf_config)

        # Species 0 = lymph_node
        prediction = (labels == 0).astype(np.uint8)

        score = score_segmentation(
            "synthetic_test", prediction, test_gt, spacing=(1.0, 1.0, 1.0)
        )

        print(f"\nDice: {score.dice:.3f}, ASSD: {score.assd:.2f}mm")

        # Generate review plots
        _save_plots(
            test_vol, test_gt, prediction, fitness_map,
            ln_losses, bg_losses, score, registry.species_names,
        )

        print(f"Plots saved to {OUTPUT_DIR}/")

        # With clear synthetic signal, model should do reasonably well
        assert score.dice > 0.3, f"Dice too low: {score.dice}"
