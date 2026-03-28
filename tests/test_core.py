"""End-to-end test of the core EcoSeg pipeline using synthetic data.

Tests the full loop: create species -> train on patches -> infer on volume -> score.
No DICOM or real data needed.
"""

import numpy as np
import torch
import pytest

from ecoseg.models.species import SpeciesRegistry, ThreeLayerCNN
from ecoseg.models.trainer import (
    TrainingConfig, train_species, extract_patches, normalize_ct
)
from ecoseg.models.inference import InferenceConfig, infer_volume
from ecoseg.metrics.scoring import dice_score, average_symmetric_surface_distance, score_segmentation


def _make_synthetic_volume(shape=(64, 64, 64), seed=42):
    """Create a synthetic CT volume with a bright sphere as 'lymph node'.

    Returns volume (in pseudo-HU) and binary ground truth mask.
    """
    rng = np.random.default_rng(seed)
    volume = rng.normal(loc=0, scale=50, size=shape).astype(np.float32)

    # Place a sphere of higher intensity in the center
    center = np.array(shape) // 2
    radius = min(shape) // 6
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        (zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2
    )
    mask = (dist <= radius).astype(np.uint8)
    volume[mask > 0] += 200  # Brighter region

    return volume, mask


class TestSpeciesModel:
    def test_forward_shape(self):
        model = ThreeLayerCNN()
        x = torch.randn(4, 1, 32, 32, 32)
        out = model(x)
        assert out.shape == (4, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_registry_fitness(self):
        registry = SpeciesRegistry(device="cpu")
        registry.add_species("a")
        registry.add_species("b")

        patches = torch.randn(8, 1, 32, 32, 32)
        results = registry.fitness_all(patches)
        assert "a" in results and "b" in results
        assert results["a"].shape == (8,)

    def test_registry_argmax(self):
        registry = SpeciesRegistry(device="cpu")
        registry.add_species("a")
        registry.add_species("b")

        patches = torch.randn(8, 1, 32, 32, 32)
        labels, fitness = registry.inference_argmax(patches)
        assert labels.shape == (8,)
        assert fitness.shape == (8,)
        assert set(labels.numpy().tolist()).issubset({0, 1})


class TestPatchExtraction:
    def test_extract_patches(self):
        vol, mask = _make_synthetic_volume()
        vol_norm = normalize_ct(vol, window_center=100, window_width=400)
        patches, labels = extract_patches(
            vol_norm, mask, patch_size=32,
            num_positive=50, num_negative=50,
        )
        assert patches.shape[0] == 100
        assert patches.shape[1:] == (1, 32, 32, 32)
        assert labels.shape == (100,)
        assert (labels == 1.0).sum() == 50
        assert (labels == 0.0).sum() == 50

    def test_normalize_ct(self):
        vol = np.array([-160, 40, 240], dtype=np.float32)
        norm = normalize_ct(vol, window_center=40, window_width=400)
        np.testing.assert_allclose(norm, [0.0, 0.5, 1.0])


class TestTraining:
    def test_train_species(self):
        vol, mask = _make_synthetic_volume()
        vol_norm = normalize_ct(vol, window_center=100, window_width=400)

        registry = SpeciesRegistry(device="cpu")
        species = registry.add_species("test")

        config = TrainingConfig(
            patch_size=32, batch_size=32,
            num_epochs=5, learning_rate=1e-3,
            positive_patches_per_scan=50,
            negative_patches_per_scan=50,
        )
        losses = train_species(
            species, [vol_norm], [mask],
            config=config, device="cpu",
        )
        assert len(losses) == 5
        # Loss should generally decrease
        assert losses[-1] < losses[0] + 0.1  # Allow some tolerance


class TestInference:
    def test_infer_small_volume(self):
        registry = SpeciesRegistry(device="cpu")
        registry.add_species("a")
        registry.add_species("b")

        vol = np.random.rand(32, 32, 32).astype(np.float32)
        config = InferenceConfig(patch_size=32, stride=32, batch_size=4)
        labels, fitness_map = infer_volume(registry, vol, config)

        assert labels.shape == (32, 32, 32)
        assert fitness_map.shape == (2, 32, 32, 32)
        assert set(np.unique(labels)).issubset({0, 1})


class TestScoring:
    def test_dice_perfect(self):
        mask = np.ones((10, 10, 10), dtype=np.uint8)
        assert dice_score(mask, mask) == 1.0

    def test_dice_empty(self):
        empty = np.zeros((10, 10, 10), dtype=np.uint8)
        assert dice_score(empty, empty) == 1.0

    def test_dice_no_overlap(self):
        a = np.zeros((10, 10, 10), dtype=np.uint8)
        b = np.zeros((10, 10, 10), dtype=np.uint8)
        a[:5] = 1
        b[5:] = 1
        assert dice_score(a, b) == 0.0

    def test_assd_perfect(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        assd = average_symmetric_surface_distance(mask, mask, spacing=(1.0, 1.0, 1.0))
        assert assd == 0.0

    def test_assd_shifted(self):
        a = np.zeros((20, 20, 20), dtype=np.uint8)
        b = np.zeros((20, 20, 20), dtype=np.uint8)
        a[5:15, 5:15, 5:15] = 1
        b[6:16, 5:15, 5:15] = 1  # Shifted by 1 voxel in z
        assd = average_symmetric_surface_distance(a, b, spacing=(1.0, 1.0, 1.0))
        assert 0.0 < assd < 2.0  # Should be around 1mm

    def test_score_segmentation(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        score = score_segmentation("test_study", mask, mask)
        assert score.dice == 1.0
        assert score.assd == 0.0
