"""Diagnostic: check what embeddings look like on real LNQ data.

Run on the H100 to diagnose why embedding GrowCut gives identical
results to intensity GrowCut.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from ecoseg.data.dicom_loader import LNQDataset
from ecoseg.models.ecosegnet import EcoSegNet, EncoderConfig
from ecoseg.models.trainer import normalize_ct
from ecoseg.models.growcut_embedding import (
    growcut_intensity, growcut_embedding,
    simulate_paint_strokes, GrowCutConfig,
)
from ecoseg.metrics.scoring import dice_score


def diagnose(data_root: str, cache_dir: str, device_str: str = "cuda"):
    device = torch.device(device_str)

    dataset = LNQDataset(Path(data_root), cache_dir=Path(cache_dir))
    dataset.discover_studies()

    # Load just one study
    val_ids = dataset.get_validation_ids(20)
    sid = val_ids[0]
    print(f"Loading {sid}...")
    dataset.preload_studies([sid])
    study = dataset.load_study(sid)

    gt = (study.seg_mask > 0).astype(np.uint8)
    vol = normalize_ct(study.volume)
    print(f"Volume shape: {vol.shape}, GT sum: {gt.sum()}")

    # Crop to ROI
    from ecoseg.experiments.growcut_experiment import compute_roi, encode_crop
    roi = compute_roi(gt, study.spacing, margin_mm=100.0, volume_shape=vol.shape)
    vol_crop = vol[roi]
    gt_crop = gt[roi]
    print(f"Crop shape: {vol_crop.shape}, GT crop sum: {gt_crop.sum()}")

    # Encode
    enc_config = EncoderConfig(feature_dim=16, pretrained=True)
    model = EcoSegNet(enc_config).to(device)

    print("Encoding (raw 720-dim features, no projection)...")
    emb_crop = encode_crop(model, vol_crop, device)
    print(f"Embedding shape: {emb_crop.shape}")
    print(f"Embedding range: [{emb_crop.min():.4f}, {emb_crop.max():.4f}]")
    print(f"Embedding mean: {emb_crop.mean():.4f}, std: {emb_crop.std():.4f}")

    # Check if embeddings are all the same
    emb_norm = F.normalize(emb_crop, dim=0)
    print(f"Normalized embedding range: [{emb_norm.min():.4f}, {emb_norm.max():.4f}]")

    # Check prototype discrimination
    gt_crop_t = torch.tensor(gt_crop, device=device)
    fg_embs = emb_norm[:, gt_crop_t == 1]  # (C, num_fg)
    bg_embs = emb_norm[:, gt_crop_t == 0]  # (C, num_bg)

    fg_proto = F.normalize(fg_embs.mean(dim=1), dim=0)
    bg_proto = F.normalize(bg_embs.mean(dim=1), dim=0)

    print(f"\nFG prototype: {fg_proto.cpu().numpy()}")
    print(f"BG prototype: {bg_proto.cpu().numpy()}")
    print(f"Proto cosine similarity: {(fg_proto * bg_proto).sum():.4f}")

    # Check affinity maps
    fg_affinity = (fg_proto.view(-1, 1, 1, 1) * emb_norm).sum(dim=0)
    bg_affinity = (bg_proto.view(-1, 1, 1, 1) * emb_norm).sum(dim=0)

    print(f"\nFG affinity at FG voxels: {fg_affinity[gt_crop_t == 1].mean():.4f}")
    print(f"FG affinity at BG voxels: {fg_affinity[gt_crop_t == 0].mean():.4f}")
    print(f"BG affinity at FG voxels: {bg_affinity[gt_crop_t == 1].mean():.4f}")
    print(f"BG affinity at BG voxels: {bg_affinity[gt_crop_t == 0].mean():.4f}")

    # Now run both GrowCut methods with same seeds
    seeds_np = simulate_paint_strokes(gt_crop, num_positive=50, num_negative=50,
                                       rng=np.random.default_rng(42))
    seeds_t = torch.tensor(seeds_np, dtype=torch.int32, device=device)
    vol_crop_t = torch.tensor(vol_crop, dtype=torch.float32, device=device)

    config = GrowCutConfig(max_iterations=200)

    print("\nRunning intensity GrowCut...")
    labels_int, str_int = growcut_intensity(vol_crop_t, seeds_t, config)
    pred_int = (labels_int == 1).cpu().numpy().astype(np.uint8)

    print("Running embedding GrowCut...")
    labels_emb, str_emb = growcut_embedding(emb_crop, seeds_t, config)
    pred_emb = (labels_emb == 1).cpu().numpy().astype(np.uint8)

    d_int = dice_score(pred_int, gt_crop)
    d_emb = dice_score(pred_emb, gt_crop)

    print(f"\nIntensity Dice: {d_int:.4f}, pred sum: {pred_int.sum()}")
    print(f"Embedding Dice: {d_emb:.4f}, pred sum: {pred_emb.sum()}")
    print(f"Identical: {np.array_equal(pred_int, pred_emb)}")

    # Check how many voxels differ
    diff = (pred_int != pred_emb)
    print(f"Voxels that differ: {diff.sum()} / {diff.size} ({diff.mean()*100:.2f}%)")

    # Check label distributions
    labels_int_np = labels_int.cpu().numpy()
    labels_emb_np = labels_emb.cpu().numpy()
    for lbl in [0, 1, 2]:
        print(f"  Label {lbl}: intensity={np.sum(labels_int_np == lbl)}, "
              f"embedding={np.sum(labels_emb_np == lbl)}")


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "/media/share/LNQ-data/mediastinal_lymph_node_seg"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "/media/volume/EcoSegCache"
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    diagnose(data_root, cache_dir, device)
