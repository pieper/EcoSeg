"""GrowCut on embeddings experiment.

Compares three methods on the 120 fully-annotated LNQ2023 scans:
1. Traditional GrowCut (intensity similarity)
2. Embedding GrowCut (cosine similarity of SwinUNETR features)
3. (Future) Learned discriminator GrowCut

Efficiency:
- Only encodes the 120 fully-annotated scans (not all 513)
- Crops to a ROI ~10cm past the lymph node label extents
- Computes embeddings only on the cropped region
- Re-uses zarr-cached CT volumes (no DICOM re-reading)

Usage:
    python -m ecoseg.experiments.growcut_experiment run \
        --data-root /path/to/lnq --cache-dir /media/volume/EcoSegCache --device cuda
"""

import argparse
import json
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from ecoseg.data.dicom_loader import LNQDataset
from ecoseg.models.ecosegnet import EcoSegNet, EncoderConfig
from ecoseg.models.growcut_embedding import (
    growcut_embedding, growcut_intensity,
    simulate_paint_strokes, GrowCutConfig,
)
from ecoseg.models.trainer import normalize_ct
from ecoseg.metrics.scoring import dice_score

logger = logging.getLogger(__name__)


def compute_roi(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    margin_mm: float = 100.0,
    volume_shape: tuple[int, int, int] = None,
) -> tuple[slice, slice, slice]:
    """Compute a bounding box ROI around the mask with a margin in mm.

    Args:
        mask: (D, H, W) binary mask
        spacing: (sz, sy, sx) voxel spacing in mm
        margin_mm: margin to add around the mask extents in mm
        volume_shape: (D, H, W) to clamp the ROI

    Returns:
        Tuple of 3 slices defining the ROI
    """
    if volume_shape is None:
        volume_shape = mask.shape

    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (slice(0, volume_shape[0]), slice(0, volume_shape[1]), slice(0, volume_shape[2]))

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    # Convert margin from mm to voxels per axis
    margin_vox = [int(np.ceil(margin_mm / s)) for s in spacing]

    slices = []
    for dim in range(3):
        lo = max(0, mins[dim] - margin_vox[dim])
        hi = min(volume_shape[dim], maxs[dim] + margin_vox[dim] + 1)
        slices.append(slice(lo, hi))

    return tuple(slices)


def encode_crop(
    model: EcoSegNet,
    volume_crop: np.ndarray,
    device: torch.device,
    target_dim: int = 32,
) -> torch.Tensor:
    """Encode a cropped volume on GPU using multi-scale features, reduced via PCA.

    Computes raw 720-dim features from the encoder, then applies PCA
    to reduce to target_dim while preserving the most discriminative
    variance. This is much better than the random projector because
    PCA is fitted to the actual data distribution.

    Args:
        model: EcoSegNet with frozen encoder
        volume_crop: (D, H, W) normalized CT volume
        device: compute device
        target_dim: output embedding dimension (default 32)

    Returns:
        (target_dim, D, H, W) tensor on GPU — PCA-reduced embeddings
    """
    from monai.inferers import SlidingWindowInferer

    D, H, W = volume_crop.shape

    vol_t = torch.tensor(volume_crop, dtype=torch.float32, device=device)
    vol_t = vol_t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # Pad to multiples of 32
    pad_d = (32 - D % 32) % 32
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        vol_t = F.pad(vol_t, (0, pad_w, 0, pad_h, 0, pad_d))

    inferer = SlidingWindowInferer(
        roi_size=(96, 96, 96),
        sw_batch_size=2,
        overlap=0.25,
        mode='gaussian',
    )

    with torch.no_grad():
        def _encode_patch(x):
            return model._multiscale_features(x)

        emb = inferer(vol_t, _encode_patch)

    # Move to CPU for PCA — 720-dim embeddings are too large for GPU PCA
    emb = emb[:, :, :D, :H, :W].squeeze(0).cpu()  # (720, D, H, W)

    # Free GPU memory from encoder
    del vol_t
    torch.cuda.empty_cache()

    # PCA reduction on CPU: 720 -> target_dim
    C = emb.shape[0]
    flat = emb.reshape(C, -1)  # (720, N)

    mean = flat.mean(dim=1, keepdim=True)
    centered = flat - mean

    # SVD on subsample to find principal components
    N = centered.shape[1]
    n_subsample = min(50000, N)
    idx = torch.randperm(N)[:n_subsample]
    subsample = centered[:, idx]  # (720, n_subsample)

    U, S, _ = torch.linalg.svd(subsample, full_matrices=False)
    components = U[:, :target_dim]  # (720, target_dim)

    # Project all voxels
    projected = components.T @ centered  # (target_dim, N)
    result = projected.reshape(target_dim, D, H, W)

    var_explained = (S[:target_dim]**2).sum() / (S**2).sum()
    logger.info(
        f"PCA: 720 -> {target_dim} dims, "
        f"variance explained: {var_explained:.1%}"
    )

    # Move result to GPU for GrowCut
    return result.to(device)


def run_experiment(args):
    """Run GrowCut comparison experiment.

    Loads and processes each scan one at a time — no bulk pre-loading.
    Results appear as each scan completes.
    """
    device = torch.device(args.device)
    rng = np.random.default_rng(42)

    cache_dir = Path(args.cache_dir)
    dataset = LNQDataset(Path(args.data_root), cache_dir=cache_dir)
    dataset.discover_studies()

    # Get fully-annotated study IDs (don't load them all upfront)
    fully_ids = dataset.get_validation_ids(20) + dataset.get_test_ids(20)
    logger.info(f"Will process {len(fully_ids)} fully-annotated studies one at a time")

    # Build encoder
    enc_config = EncoderConfig(feature_dim=args.feature_dim, pretrained=True)
    model = EcoSegNet(enc_config).to(device)

    stroke_counts = [5, 10, 20, 50, 100, 200]
    methods = ["intensity", "embedding"]
    results = {m: {n: [] for n in stroke_counts} for m in methods}

    gc_config = GrowCutConfig(max_iterations=200, convergence_threshold=0.001)

    viz_cases = []

    for i, sid in enumerate(fully_ids):
        study = dataset.load_study(sid)
        if study.seg_mask is None:
            continue

        gt_full = (study.seg_mask > 0).astype(np.uint8)
        if gt_full.sum() == 0:
            continue

        vol_full = normalize_ct(study.volume)

        # Crop to ROI around lymph nodes (10cm margin)
        roi = compute_roi(gt_full, study.spacing, margin_mm=100.0, volume_shape=vol_full.shape)
        vol_crop = vol_full[roi]
        gt_crop = gt_full[roi]

        crop_shape = vol_crop.shape
        logger.debug(
            f"{sid}: full={vol_full.shape}, crop={crop_shape}, "
            f"reduction={np.prod(crop_shape)/np.prod(vol_full.shape):.1%}"
        )

        # Compute embeddings on cropped region only
        emb_crop = encode_crop(model, vol_crop, device)  # (C, D', H', W') on GPU
        vol_crop_t = torch.tensor(vol_crop, dtype=torch.float32, device=device)

        for n_strokes in stroke_counts:
            # Simulate paint strokes on cropped GT
            seeds_np = simulate_paint_strokes(
                gt_crop,
                num_positive=n_strokes,
                num_negative=n_strokes,
                rng=np.random.default_rng(rng.integers(2**31) + n_strokes),
            )
            seeds_t = torch.tensor(seeds_np, dtype=torch.int32, device=device)

            # Traditional GrowCut (intensity)
            t0 = time.time()
            labels_int, strength_int = growcut_intensity(vol_crop_t, seeds_t, gc_config)
            time_int = time.time() - t0
            pred_int = (labels_int == 1).cpu().numpy().astype(np.uint8)
            dice_int = dice_score(pred_int, gt_crop)

            results["intensity"][n_strokes].append({
                "study_id": sid, "dice": float(dice_int), "time_s": time_int,
            })

            # Embedding GrowCut
            t0 = time.time()
            labels_emb, strength_emb = growcut_embedding(emb_crop, seeds_t, gc_config)
            time_emb = time.time() - t0
            pred_emb = (labels_emb == 1).cpu().numpy().astype(np.uint8)
            dice_emb = dice_score(pred_emb, gt_crop)

            results["embedding"][n_strokes].append({
                "study_id": sid, "dice": float(dice_emb), "time_s": time_emb,
            })

            # Save data for visualization (first 3 cases, 50 strokes)
            if len(viz_cases) < 3 and n_strokes == 50:
                viz_cases.append({
                    "study_id": sid,
                    "volume": vol_crop,
                    "gt": gt_crop,
                    "seeds": seeds_np,
                    "pred_intensity": pred_int,
                    "pred_embedding": pred_emb,
                    "strength_intensity": strength_int.cpu().numpy(),
                    "strength_embedding": strength_emb.cpu().numpy(),
                    "dice_intensity": dice_int,
                    "dice_embedding": dice_emb,
                })

        # Free GPU and CPU memory — don't accumulate all 120 studies
        del emb_crop, vol_crop_t
        if sid in dataset._studies:
            del dataset._studies[sid]
        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0 or i + 1 == len(fully_ids):
            parts = []
            for n in [10, 50, 200]:
                if results["intensity"][n] and results["embedding"][n]:
                    di = np.mean([r["dice"] for r in results["intensity"][n]])
                    de = np.mean([r["dice"] for r in results["embedding"][n]])
                    parts.append(f"{n}pt: int={di:.3f} emb={de:.3f}")
            logger.info(f"  [{i+1}/{len(fully_ids)}] " + ", ".join(parts))

    # Save results
    output_dir = cache_dir / "growcut_experiment"
    output_dir.mkdir(exist_ok=True)

    summary = {}
    for method in methods:
        summary[method] = {}
        for n in stroke_counts:
            if results[method][n]:
                dices = [r["dice"] for r in results[method][n]]
                times = [r["time_s"] for r in results[method][n]]
                summary[method][f"{n}_strokes"] = {
                    "mean_dice": float(np.mean(dices)),
                    "std_dice": float(np.std(dices)),
                    "median_dice": float(np.median(dices)),
                    "mean_time_s": float(np.mean(times)),
                    "num_cases": len(dices),
                }

    with open(output_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "per_case": results}, f, indent=2)

    # Print summary table
    logger.info("\n=== Results Summary ===")
    logger.info(f"{'Strokes':>8} | {'Intensity Dice':>15} | {'Embedding Dice':>15}")
    logger.info("-" * 45)
    for n in stroke_counts:
        di = summary["intensity"].get(f"{n}_strokes", {}).get("mean_dice", 0)
        de = summary["embedding"].get(f"{n}_strokes", {}).get("mean_dice", 0)
        logger.info(f"{n:>8} | {di:>15.3f} | {de:>15.3f}")

    # Generate visualization
    if viz_cases:
        _generate_visualizations(viz_cases, output_dir)

    logger.info(f"Results saved to {output_dir}")


def _generate_visualizations(viz_cases: list, output_dir: Path):
    """Generate comparison images for representative cases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for case_idx, case in enumerate(viz_cases):
        vol = case["volume"]
        gt = case["gt"]
        seeds = case["seeds"]
        pred_int = case["pred_intensity"]
        pred_emb = case["pred_embedding"]
        strength_int = case["strength_intensity"]
        strength_emb = case["strength_embedding"]

        # Find the slice with the most ground truth
        gt_per_slice = gt.sum(axis=(1, 2))
        best_slice = np.argmax(gt_per_slice)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # --- Row 1: Segmentation comparison ---

        # CT with ground truth + seeds
        axes[0, 0].imshow(vol[best_slice], cmap="gray")
        axes[0, 0].contour(gt[best_slice], levels=[0.5], colors="lime", linewidths=2)
        pos_seeds = seeds[best_slice] == 1
        neg_seeds = seeds[best_slice] == 2
        if pos_seeds.any():
            ys, xs = np.where(pos_seeds)
            axes[0, 0].scatter(xs, ys, c="red", s=15, zorder=5,
                             edgecolors="white", linewidths=0.5)
        if neg_seeds.any():
            ys, xs = np.where(neg_seeds)
            axes[0, 0].scatter(xs, ys, c="blue", s=15, zorder=5,
                             edgecolors="white", linewidths=0.5)
        axes[0, 0].set_title("CT + GT (green) + Seeds (red/blue)")
        axes[0, 0].axis("off")

        # Intensity GrowCut
        axes[0, 1].imshow(vol[best_slice], cmap="gray")
        axes[0, 1].contour(gt[best_slice], levels=[0.5], colors="lime",
                          linewidths=1, linestyles="dashed")
        if pred_int[best_slice].any():
            axes[0, 1].contour(pred_int[best_slice], levels=[0.5], colors="red", linewidths=2)
        axes[0, 1].set_title(f"Intensity GrowCut\nDice={case['dice_intensity']:.3f}")
        axes[0, 1].axis("off")

        # Embedding GrowCut
        axes[0, 2].imshow(vol[best_slice], cmap="gray")
        axes[0, 2].contour(gt[best_slice], levels=[0.5], colors="lime",
                          linewidths=1, linestyles="dashed")
        if pred_emb[best_slice].any():
            axes[0, 2].contour(pred_emb[best_slice], levels=[0.5], colors="cyan", linewidths=2)
        axes[0, 2].set_title(f"Embedding GrowCut\nDice={case['dice_embedding']:.3f}")
        axes[0, 2].axis("off")

        # Overlay
        axes[0, 3].imshow(vol[best_slice], cmap="gray")
        axes[0, 3].contour(gt[best_slice], levels=[0.5], colors="lime",
                          linewidths=1.5, linestyles="dashed")
        if pred_int[best_slice].any():
            axes[0, 3].contour(pred_int[best_slice], levels=[0.5], colors="red", linewidths=1.5)
        if pred_emb[best_slice].any():
            axes[0, 3].contour(pred_emb[best_slice], levels=[0.5], colors="cyan", linewidths=1.5)
        axes[0, 3].set_title("Overlay\nGT=green, Int=red, Emb=cyan")
        axes[0, 3].axis("off")

        # --- Row 2: Strength maps ---

        axes[1, 0].imshow(vol[best_slice], cmap="gray")
        axes[1, 0].set_title("CT Volume (cropped)")
        axes[1, 0].axis("off")

        im1 = axes[1, 1].imshow(strength_int[best_slice], cmap="hot", vmin=0, vmax=1)
        axes[1, 1].set_title("Intensity Strength")
        axes[1, 1].axis("off")
        plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)

        im2 = axes[1, 2].imshow(strength_emb[best_slice], cmap="hot", vmin=0, vmax=1)
        axes[1, 2].set_title("Embedding Strength")
        axes[1, 2].axis("off")
        plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

        diff = strength_emb[best_slice] - strength_int[best_slice]
        im3 = axes[1, 3].imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[1, 3].set_title("Emb − Int Strength")
        axes[1, 3].axis("off")
        plt.colorbar(im3, ax=axes[1, 3], fraction=0.046)

        fig.suptitle(
            f"{case['study_id']} — 50 strokes per class (cropped ROI)\n"
            f"Intensity Dice: {case['dice_intensity']:.3f}, "
            f"Embedding Dice: {case['dice_embedding']:.3f}",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"comparison_{case_idx}_{case['study_id']}.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved visualization for {case['study_id']}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="GrowCut on embeddings experiment")
    parser.add_argument("command", choices=["run"],
                       help="Run the GrowCut comparison experiment")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--feature-dim", type=int, default=16)

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    run_experiment(args)


if __name__ == "__main__":
    main()
