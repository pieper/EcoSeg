"""GrowCut on embeddings experiment.

Compares three methods:
1. Traditional GrowCut (intensity similarity)
2. Embedding GrowCut (cosine similarity of SwinUNETR features)
3. (Future) Learned discriminator GrowCut

Usage:
    # Step 1: Precompute embeddings (one-time)
    python -m ecoseg.experiments.growcut_experiment precompute \
        --data-root /path/to/lnq --cache-dir /media/volume/EcoSegCache --device cuda

    # Step 2: Run experiment (fast)
    python -m ecoseg.experiments.growcut_experiment run \
        --data-root /path/to/lnq --cache-dir /media/volume/EcoSegCache --device cuda
"""

import argparse
import json
import logging
import time
import numpy as np
import torch
from pathlib import Path

from ecoseg.data.dicom_loader import LNQDataset
from ecoseg.models.ecosegnet import EcoSegNet, EncoderConfig
from ecoseg.models.embedding_cache import EmbeddingCache
from ecoseg.models.growcut_embedding import (
    growcut_embedding, growcut_intensity,
    simulate_paint_strokes, GrowCutConfig,
)
from ecoseg.models.trainer import normalize_ct
from ecoseg.metrics.scoring import dice_score

logger = logging.getLogger(__name__)


def precompute_embeddings(args):
    """Precompute and cache embeddings for all fully-annotated scans."""
    device = torch.device(args.device)

    dataset = LNQDataset(Path(args.data_root), cache_dir=Path(args.cache_dir))
    dataset.discover_studies()

    logger.info("Pre-loading studies...")
    all_ids = list(dataset._index.keys())
    dataset.preload_studies(all_ids)

    config = EncoderConfig(feature_dim=args.feature_dim, pretrained=True)
    model = EcoSegNet(config).to(device)
    emb_cache = EmbeddingCache(Path(args.cache_dir), feature_dim=args.feature_dim)

    fully_ids = dataset.get_validation_ids(20) + dataset.get_test_ids(20)
    logger.info(f"Encoding {len(fully_ids)} fully-annotated studies...")

    for i, sid in enumerate(fully_ids):
        if emb_cache.has(sid):
            if (i + 1) % 20 == 0:
                logger.info(f"  [{i+1}/{len(fully_ids)}] (cached)")
            continue

        study = dataset.load_study(sid)
        vol = normalize_ct(study.volume)
        emb_cache.encode_and_cache(model, sid, vol, device)
        logger.info(f"  [{i+1}/{len(fully_ids)}] {sid}: encoded")

    logger.info("Precomputation complete.")


def run_experiment(args):
    """Run GrowCut comparison experiment."""
    device = torch.device(args.device)
    rng = np.random.default_rng(42)

    cache_dir = Path(args.cache_dir)
    dataset = LNQDataset(Path(args.data_root), cache_dir=cache_dir)
    dataset.discover_studies()
    emb_cache = EmbeddingCache(cache_dir, feature_dim=args.feature_dim)

    fully_ids = dataset.get_validation_ids(20) + dataset.get_test_ids(20)
    available = [sid for sid in fully_ids if emb_cache.has(sid)]
    logger.info(f"Running on {len(available)} studies with cached embeddings")

    dataset.preload_studies(available)

    stroke_counts = [5, 10, 20, 50, 100, 200]
    methods = ["intensity", "embedding"]
    results = {m: {n: [] for n in stroke_counts} for m in methods}

    gc_config = GrowCutConfig(max_iterations=200, convergence_threshold=0.001)

    # Collect a few cases for visualization
    viz_cases = []

    for i, sid in enumerate(available):
        study = dataset.load_study(sid)
        if study.seg_mask is None:
            continue

        gt = (study.seg_mask > 0).astype(np.uint8)
        if gt.sum() == 0:
            continue

        vol = normalize_ct(study.volume)

        # Load embeddings
        emb_np = emb_cache.load(sid)
        if emb_np is None:
            continue
        emb_t = torch.tensor(emb_np, dtype=torch.float32, device=device)
        vol_t = torch.tensor(vol, dtype=torch.float32, device=device)

        for n_strokes in stroke_counts:
            # Use same seeds for both methods (fair comparison)
            seeds_np = simulate_paint_strokes(
                gt,
                num_positive=n_strokes,
                num_negative=n_strokes,
                rng=np.random.default_rng(rng.integers(2**31) + n_strokes),
            )
            seeds_t = torch.tensor(seeds_np, dtype=torch.int32, device=device)

            # Traditional GrowCut (intensity)
            t0 = time.time()
            labels_int, strength_int = growcut_intensity(vol_t, seeds_t, gc_config)
            time_int = time.time() - t0
            pred_int = (labels_int == 1).cpu().numpy().astype(np.uint8)
            dice_int = dice_score(pred_int, gt)

            results["intensity"][n_strokes].append({
                "study_id": sid, "dice": float(dice_int), "time_s": time_int,
            })

            # Embedding GrowCut
            t0 = time.time()
            labels_emb, strength_emb = growcut_embedding(emb_t, seeds_t, gc_config)
            time_emb = time.time() - t0
            pred_emb = (labels_emb == 1).cpu().numpy().astype(np.uint8)
            dice_emb = dice_score(pred_emb, gt)

            results["embedding"][n_strokes].append({
                "study_id": sid, "dice": float(dice_emb), "time_s": time_emb,
            })

            # Save data for visualization (first 3 cases, 50 strokes)
            if len(viz_cases) < 3 and n_strokes == 50:
                viz_cases.append({
                    "study_id": sid,
                    "volume": vol,
                    "gt": gt,
                    "seeds": seeds_np,
                    "pred_intensity": pred_int,
                    "pred_embedding": pred_emb,
                    "strength_intensity": strength_int.cpu().numpy(),
                    "strength_embedding": strength_emb.cpu().numpy(),
                    "dice_intensity": dice_int,
                    "dice_embedding": dice_emb,
                    "spacing": study.spacing,
                })

        # Free GPU memory
        del emb_t, vol_t
        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0 or i + 1 == len(available):
            parts = []
            for n in stroke_counts:
                if results["intensity"][n] and results["embedding"][n]:
                    di = np.mean([r["dice"] for r in results["intensity"][n]])
                    de = np.mean([r["dice"] for r in results["embedding"][n]])
                    parts.append(f"{n}pt: int={di:.3f} emb={de:.3f}")
            logger.info(f"  [{i+1}/{len(available)}] " + ", ".join(parts))

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

        # Find the slice with the most ground truth lymph node
        gt_per_slice = gt.sum(axis=(1, 2))
        best_slice = np.argmax(gt_per_slice)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # --- Row 1: Segmentation comparison ---

        # CT with ground truth
        axes[0, 0].imshow(vol[best_slice], cmap="gray")
        axes[0, 0].contour(gt[best_slice], levels=[0.5], colors="lime", linewidths=2)
        # Show seed points
        pos_seeds = seeds[best_slice] == 1
        neg_seeds = seeds[best_slice] == 2
        if pos_seeds.any():
            ys, xs = np.where(pos_seeds)
            axes[0, 0].scatter(xs, ys, c="red", s=15, zorder=5, edgecolors="white", linewidths=0.5)
        if neg_seeds.any():
            ys, xs = np.where(neg_seeds)
            axes[0, 0].scatter(xs, ys, c="blue", s=15, zorder=5, edgecolors="white", linewidths=0.5)
        axes[0, 0].set_title("CT + GT (green) + Seeds (red/blue)")
        axes[0, 0].axis("off")

        # Intensity GrowCut result
        axes[0, 1].imshow(vol[best_slice], cmap="gray")
        axes[0, 1].contour(gt[best_slice], levels=[0.5], colors="lime", linewidths=1, linestyles="dashed")
        if pred_int[best_slice].any():
            axes[0, 1].contour(pred_int[best_slice], levels=[0.5], colors="red", linewidths=2)
        axes[0, 1].set_title(f"Intensity GrowCut\nDice={case['dice_intensity']:.3f}")
        axes[0, 1].axis("off")

        # Embedding GrowCut result
        axes[0, 2].imshow(vol[best_slice], cmap="gray")
        axes[0, 2].contour(gt[best_slice], levels=[0.5], colors="lime", linewidths=1, linestyles="dashed")
        if pred_emb[best_slice].any():
            axes[0, 2].contour(pred_emb[best_slice], levels=[0.5], colors="cyan", linewidths=2)
        axes[0, 2].set_title(f"Embedding GrowCut\nDice={case['dice_embedding']:.3f}")
        axes[0, 2].axis("off")

        # Side by side overlay
        axes[0, 3].imshow(vol[best_slice], cmap="gray")
        axes[0, 3].contour(gt[best_slice], levels=[0.5], colors="lime", linewidths=1.5, linestyles="dashed")
        if pred_int[best_slice].any():
            axes[0, 3].contour(pred_int[best_slice], levels=[0.5], colors="red", linewidths=1.5)
        if pred_emb[best_slice].any():
            axes[0, 3].contour(pred_emb[best_slice], levels=[0.5], colors="cyan", linewidths=1.5)
        axes[0, 3].set_title("Overlay\nGT=green, Int=red, Emb=cyan")
        axes[0, 3].axis("off")

        # --- Row 2: Strength/confidence maps ---

        axes[1, 0].imshow(vol[best_slice], cmap="gray")
        axes[1, 0].set_title("CT Volume")
        axes[1, 0].axis("off")

        im1 = axes[1, 1].imshow(strength_int[best_slice], cmap="hot", vmin=0, vmax=1)
        axes[1, 1].set_title("Intensity Strength")
        axes[1, 1].axis("off")
        plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)

        im2 = axes[1, 2].imshow(strength_emb[best_slice], cmap="hot", vmin=0, vmax=1)
        axes[1, 2].set_title("Embedding Strength")
        axes[1, 2].axis("off")
        plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

        # Strength difference
        diff = strength_emb[best_slice] - strength_int[best_slice]
        im3 = axes[1, 3].imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[1, 3].set_title("Emb - Int Strength")
        axes[1, 3].axis("off")
        plt.colorbar(im3, ax=axes[1, 3], fraction=0.046)

        fig.suptitle(
            f"{case['study_id']} — 50 strokes per class\n"
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
    subparsers = parser.add_subparsers(dest="command")

    pre = subparsers.add_parser("precompute", help="Precompute embeddings")
    pre.add_argument("--data-root", type=str, required=True)
    pre.add_argument("--cache-dir", type=str, required=True)
    pre.add_argument("--device", type=str, default="auto")
    pre.add_argument("--feature-dim", type=int, default=16)

    run = subparsers.add_parser("run", help="Run GrowCut experiment")
    run.add_argument("--data-root", type=str, required=True)
    run.add_argument("--cache-dir", type=str, required=True)
    run.add_argument("--device", type=str, default="auto")
    run.add_argument("--feature-dim", type=int, default=16)

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    if args.command == "precompute":
        precompute_embeddings(args)
    elif args.command == "run":
        run_experiment(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
