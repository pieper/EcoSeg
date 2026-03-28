"""Sweep inference stride to characterize the resolution/compute tradeoff.

Trains species once, then runs inference at strides 16, 8, 4, 2, 1
and plots Dice, ASSD, and visual results at each stride.
"""

import time
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecoseg.models.species import SpeciesRegistry
from ecoseg.models.trainer import TrainingConfig, train_species, normalize_ct
from ecoseg.models.inference import InferenceConfig, infer_volume
from ecoseg.metrics.scoring import score_segmentation


OUTPUT_DIR = Path(__file__).parent / "output"


def _make_two_class_volume(shape=(96, 96, 96), seed=42):
    rng = np.random.default_rng(seed)
    volume = rng.normal(loc=-50, scale=20, size=shape).astype(np.float32)
    center = np.array(shape) // 2
    radius = min(shape) // 4
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    mask = (dist <= radius).astype(np.uint8)
    volume[mask > 0] = rng.normal(loc=200, scale=20, size=mask.sum()).astype(np.float32)
    return volume, mask


def run_stride_sweep():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    registry = SpeciesRegistry(device=device)
    registry.add_species("lymph_node")
    registry.add_species("background")

    # Train once on training volume
    vol_raw, gt_mask = _make_two_class_volume(seed=42)
    vol = normalize_ct(vol_raw, window_center=50, window_width=300)

    config = TrainingConfig(
        patch_size=32, batch_size=32, num_epochs=50,
        learning_rate=1e-3,
        positive_patches_per_scan=300, negative_patches_per_scan=300,
    )

    print("Training lymph node species...")
    train_species(registry.species["lymph_node"], [vol], [gt_mask], config=config, device=device)
    print("Training background species...")
    bg_mask = (gt_mask == 0).astype(np.uint8)
    train_species(registry.species["background"], [vol], [bg_mask], config=config, device=device)

    # Test volume
    test_vol_raw, test_gt = _make_two_class_volume(seed=99)
    test_vol = normalize_ct(test_vol_raw, window_center=50, window_width=300)

    strides = [16, 8, 4, 2, 1]
    results = []

    for stride in strides:
        print(f"\nInference stride={stride}...")
        t0 = time.time()
        inf_config = InferenceConfig(patch_size=32, stride=stride, batch_size=64)
        labels, fitness_map = infer_volume(registry, test_vol, inf_config)
        elapsed = time.time() - t0

        prediction = (labels == 0).astype(np.uint8)
        score = score_segmentation("test", prediction, test_gt, spacing=(1.0, 1.0, 1.0))

        ln_idx = registry.species_names.index("lymph_node")
        bg_idx = registry.species_names.index("background")

        results.append({
            "stride": stride,
            "dice": score.dice,
            "assd": score.assd,
            "time_s": elapsed,
            "prediction": prediction,
            "fitness_ln": fitness_map[ln_idx],
            "fitness_bg": fitness_map[bg_idx],
        })
        print(f"  Dice: {score.dice:.3f}, ASSD: {score.assd:.2f}mm, Time: {elapsed:.1f}s")

    # --- Plot 1: Metrics vs stride ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    strides_arr = [r["stride"] for r in results]
    dices = [r["dice"] for r in results]
    assds = [r["assd"] for r in results]
    times = [r["time_s"] for r in results]

    axes[0].plot(strides_arr, dices, "o-", color="green", linewidth=2, markersize=8)
    axes[0].set_xlabel("Stride")
    axes[0].set_ylabel("Dice")
    axes[0].set_title("Dice vs Stride")
    axes[0].invert_xaxis()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    for s, d in zip(strides_arr, dices):
        axes[0].annotate(f"{d:.3f}", (s, d), textcoords="offset points", xytext=(0, 10), ha="center")

    axes[1].plot(strides_arr, assds, "o-", color="orange", linewidth=2, markersize=8)
    axes[1].set_xlabel("Stride")
    axes[1].set_ylabel("ASSD (mm)")
    axes[1].set_title("ASSD vs Stride")
    axes[1].invert_xaxis()
    axes[1].grid(True, alpha=0.3)
    for s, a in zip(strides_arr, assds):
        axes[1].annotate(f"{a:.1f}", (s, a), textcoords="offset points", xytext=(0, 10), ha="center")

    axes[2].plot(strides_arr, times, "o-", color="red", linewidth=2, markersize=8)
    axes[2].set_xlabel("Stride")
    axes[2].set_ylabel("Time (s)")
    axes[2].set_title("Inference Time vs Stride")
    axes[2].invert_xaxis()
    axes[2].grid(True, alpha=0.3)
    for s, t in zip(strides_arr, times):
        axes[2].annotate(f"{t:.1f}s", (s, t), textcoords="offset points", xytext=(0, 10), ha="center")

    fig.suptitle("Stride Sweep: Resolution vs Compute Tradeoff", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stride_sweep_metrics.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Visual comparison at each stride ---
    mid_z = test_vol.shape[0] // 2
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 11))

    for i, r in enumerate(results):
        # Row 1: Prediction overlay
        axes[0, i].imshow(test_vol[mid_z], cmap="gray")
        axes[0, i].contour(r["prediction"][mid_z], levels=[0.5], colors=["red"], linewidths=1.5)
        axes[0, i].contour(test_gt[mid_z], levels=[0.5], colors=["lime"], linewidths=1.0, linestyles="dashed")
        axes[0, i].set_title(f"Stride {r['stride']}\nDice={r['dice']:.3f}")
        axes[0, i].axis("off")

        # Row 2: LN fitness
        im = axes[1, i].imshow(r["fitness_ln"][mid_z], cmap="hot", vmin=0, vmax=1)
        axes[1, i].set_title(f"LN Fitness")
        axes[1, i].axis("off")

        # Row 3: Fitness difference
        diff = r["fitness_ln"][mid_z] - r["fitness_bg"][mid_z]
        axes[2, i].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[2, i].set_title(f"LN - BG")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Prediction", fontsize=12)
    axes[1, 0].set_ylabel("LN Fitness", fontsize=12)
    axes[2, 0].set_ylabel("Fitness Diff", fontsize=12)

    fig.suptitle("Visual Comparison Across Strides", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stride_sweep_visual.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Fitness profiles through center at each stride ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    center_y, center_x = test_vol.shape[1] // 2, test_vol.shape[2] // 2
    colors = plt.cm.viridis(np.linspace(0, 0.9, n))

    for i, r in enumerate(results):
        diff_profile = r["fitness_ln"][:, center_y, center_x] - r["fitness_bg"][:, center_y, center_x]
        ax.plot(diff_profile, label=f"stride={r['stride']}", color=colors[i], linewidth=1.5)

    # Ground truth boundary
    gt_profile = test_gt[:, center_y, center_x].astype(float)
    gt_diff = gt_profile * 2 - 1  # Map 0->-1, 1->+1
    ax.plot(gt_diff, "--", color="gray", linewidth=2, alpha=0.5, label="Ground truth")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Slice (z)")
    ax.set_ylabel("Fitness Difference (LN - BG)")
    ax.set_title("Fitness Difference Profile: Convergence with Decreasing Stride")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stride_sweep_profiles.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_stride_sweep()
