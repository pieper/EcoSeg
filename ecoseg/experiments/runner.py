"""Experiment runner for the LNQ2023 species segmentation experiment.

Orchestrates the train -> infer -> score loop:
1. Train species on initial fully-annotated scans (validation set)
2. Infer on held-out test scans
3. Score with Dice and ASSD
4. Incrementally add partially-annotated scans in batches
5. Retrain and re-evaluate after each batch
"""

import json
import logging
import time
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from ecoseg import available_workers
from ecoseg.models.species import SpeciesRegistry
from ecoseg.models.trainer import TrainingConfig, train_species, normalize_ct, extract_patches
from ecoseg.models.inference import InferenceConfig, infer_volume
from ecoseg.metrics.scoring import score_segmentation, SegmentationScore
from ecoseg.data.dicom_loader import LNQDataset, StudyData

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an LNQ experiment run."""
    name: str = "baseline_20-100_cnn3_batch20"
    data_root: str = ""
    output_dir: str = "output"

    # Dataset splits
    num_validation: int = 20
    batch_size_partial: int = 20

    # Species configuration
    architecture: str = "cnn3"
    species_names: list[str] = field(default_factory=lambda: ["lymph_node", "background"])

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Inference
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Device
    device: str = "auto"

    # Cache
    cache_dir: Optional[str] = None

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        training = TrainingConfig(**data.pop("training", {}))
        inference = InferenceConfig(**data.pop("inference", {}))
        return cls(training=training, inference=inference, **data)

    def to_json(self, path: Path) -> None:
        data = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class GenerationResult:
    """Results from one generation of training + evaluation."""
    generation: int
    num_training_scans: int
    scores: list[SegmentationScore]
    mean_dice: float
    mean_assd: float
    training_time_s: float
    inference_time_s: float

    def summary(self) -> dict:
        return {
            "generation": self.generation,
            "num_training_scans": self.num_training_scans,
            "mean_dice": self.mean_dice,
            "mean_assd": self.mean_assd,
            "training_time_s": self.training_time_s,
            "inference_time_s": self.inference_time_s,
            "num_evaluated": len(self.scores),
        }


class ExperimentRunner:
    """Runs the incremental species training experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.resolve_device()
        self.registry = SpeciesRegistry(device=self.device)
        self.results: list[GenerationResult] = []
        self.dataset: Optional[LNQDataset] = None
        self.rng = np.random.default_rng(42)

        # Output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment '{config.name}' using device: {self.device}")

    def setup(self) -> None:
        """Initialize dataset, pre-load all studies, and create species."""
        # Load dataset
        cache_path = Path(self.config.cache_dir) if self.config.cache_dir else None
        self.dataset = LNQDataset(Path(self.config.data_root), cache_dir=cache_path)
        self.dataset.discover_studies()

        # Pre-load ALL studies in parallel using all available cores
        all_ids = list(self.dataset._index.keys())
        self.dataset.preload_studies(all_ids)

        # Create species
        for name in self.config.species_names:
            self.registry.add_species(name, self.config.architecture)

        logger.info(
            f"Created {len(self.registry.species)} species: "
            f"{self.registry.species_names}"
        )

    def train_on_studies(
        self,
        fully_ids: list[str],
        partial_ids: list[str],
    ) -> float:
        """Train all species on accumulated fully + partially annotated studies.

        For fully-annotated studies:
          - lymph_node species: positive = labeled voxels
          - background species: positive = everything else

        For partially-annotated studies:
          - lymph_node species: positive = labeled voxels
          - background species: positive = 1-voxel dilation ring around labels

        Species models are re-initialized before each training round so they
        learn from the full accumulated dataset, not incrementally from stale weights.

        Returns training time in seconds.
        """
        from scipy import ndimage

        t0 = time.time()

        # Re-initialize species weights to avoid accumulating bias
        for name in self.config.species_names:
            species = self.registry.species[name]
            from ecoseg.models.species import ARCHITECTURES
            species.network = ARCHITECTURES[species.architecture]().to(self.device)

        volumes = []
        ln_masks = []
        bg_masks = []

        def _prepare_study(sid: str, annotation_type: str):
            study = self.dataset.load_study(sid)
            vol = normalize_ct(study.volume)
            volumes.append(vol)

            if study.seg_mask is None:
                return

            ln_mask = (study.seg_mask > 0).astype(np.uint8)
            ln_masks.append(ln_mask)

            if annotation_type == "fully_annotated":
                bg_mask = (study.seg_mask == 0).astype(np.uint8)
            else:
                if study.spacing[0] > 3.0:
                    struct = np.zeros((3, 3, 3), dtype=bool)
                    struct[1, :, :] = True
                else:
                    struct = np.ones((3, 3, 3), dtype=bool)
                dilated = ndimage.binary_dilation(ln_mask.astype(bool), struct)
                bg_mask = (dilated & ~ln_mask.astype(bool)).astype(np.uint8)

            bg_masks.append(bg_mask)

        for sid in fully_ids:
            _prepare_study(sid, "fully_annotated")
        for sid in partial_ids:
            _prepare_study(sid, "partially_annotated")

        logger.info(
            f"Training on {len(fully_ids)} full + {len(partial_ids)} partial scans "
            f"({len(ln_masks)} with LN masks, {len(bg_masks)} with BG masks)"
        )

        # Train lymph node species
        ln_species = self.registry.species["lymph_node"]
        if ln_masks:
            losses = train_species(
                ln_species, volumes, ln_masks,
                config=self.config.training,
                device=self.device,
                rng=self.rng,
            )
            if losses:
                logger.info(
                    f"Lymph node training: final loss={losses[-1]:.4f} "
                    f"({len(losses)} epochs)"
                )

        # Train background species
        bg_species = self.registry.species["background"]
        if bg_masks:
            losses = train_species(
                bg_species, volumes, bg_masks,
                config=self.config.training,
                device=self.device,
                rng=self.rng,
            )
            if losses:
                logger.info(
                    f"Background training: final loss={losses[-1]:.4f} "
                    f"({len(losses)} epochs)"
                )

        return time.time() - t0

    def evaluate(self, study_ids: list[str]) -> tuple[list[SegmentationScore], float]:
        """Run inference on studies and score against ground truth.

        Inference runs on GPU sequentially (one scan at a time), but scoring
        (Dice + ASSD) is submitted to a thread pool so the CPU computes
        metrics while the GPU processes the next scan.

        Returns (scores, inference_time_s).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        t0 = time.time()
        ln_index = self.registry.species_names.index("lymph_node")

        # Use threads for scoring (scipy releases GIL during distance_transform)
        scoring_futures = []

        with ThreadPoolExecutor(max_workers=available_workers()) as scorer:
            for i, sid in enumerate(study_ids):
                study = self.dataset.load_study(sid)
                vol = normalize_ct(study.volume)

                labels, fitness_map = infer_volume(
                    self.registry, vol, self.config.inference
                )

                prediction = (labels == ln_index).astype(np.uint8)

                if study.seg_mask is not None:
                    ground_truth = (study.seg_mask > 0).astype(np.uint8)
                    # Submit scoring to thread pool — runs while GPU does next scan
                    future = scorer.submit(
                        score_segmentation,
                        sid, prediction, ground_truth, study.spacing,
                    )
                    scoring_futures.append(future)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Inferred {i + 1}/{len(study_ids)} scans")

            # Collect all scores
            scores = []
            for future in scoring_futures:
                score = future.result()
                scores.append(score)

        elapsed = time.time() - t0
        return scores, elapsed

    def run_generation(
        self,
        fully_ids: list[str],
        partial_ids: list[str],
        eval_ids: list[str],
    ) -> GenerationResult:
        """Run one generation: train on all accumulated data, evaluate on eval_ids."""
        gen = len(self.results)
        total_train = len(fully_ids) + len(partial_ids)
        logger.info(
            f"=== Generation {gen}: training on {total_train} scans "
            f"({len(fully_ids)} full + {len(partial_ids)} partial), "
            f"evaluating on {len(eval_ids)} ==="
        )

        train_time = self.train_on_studies(fully_ids, partial_ids)
        scores, infer_time = self.evaluate(eval_ids)

        finite_assd = [s.assd for s in scores if np.isfinite(s.assd)]

        result = GenerationResult(
            generation=gen,
            num_training_scans=total_train,
            scores=scores,
            mean_dice=np.mean([s.dice for s in scores]) if scores else 0.0,
            mean_assd=np.mean(finite_assd) if finite_assd else float("inf"),
            training_time_s=train_time,
            inference_time_s=infer_time,
        )
        self.results.append(result)

        logger.info(
            f"  Mean Dice: {result.mean_dice:.4f}, "
            f"Mean ASSD: {result.mean_assd:.2f}mm, "
            f"Train: {train_time:.1f}s, Infer: {infer_time:.1f}s"
        )

        # Save results
        self._save_generation(result)

        return result

    def run_full_experiment(self) -> list[GenerationResult]:
        """Run the complete incremental experiment.

        1. Train on 20 validation scans (fully annotated)
        2. Evaluate on 100 test scans
        3. Incrementally add partial scans in batches of 20
        4. Retrain on ALL accumulated data each generation, re-evaluate

        Training is cumulative: each generation trains on the 20 fully-
        annotated scans PLUS all partial scans added so far.
        """
        self.setup()

        validation_ids = self.dataset.get_validation_ids(self.config.num_validation)
        test_ids = self.dataset.get_test_ids(self.config.num_validation)

        logger.info(
            f"Validation: {len(validation_ids)} studies, "
            f"Test: {len(test_ids)} studies"
        )

        # Accumulate training data across generations
        # fully_annotated IDs are always included
        self._fully_annotated_ids = list(validation_ids)
        self._partial_annotated_ids: list[str] = []

        # Generation 0: train on fully-annotated validation set only
        self.run_generation(
            self._fully_annotated_ids, [], test_ids,
        )

        # Get partially annotated study IDs
        all_ids = set(self.dataset._index.keys())
        fully_ids = set(validation_ids) | set(test_ids)
        partial_ids = sorted(all_ids - fully_ids)

        # Add partial scans in batches — training is cumulative
        batch_size = self.config.batch_size_partial
        for i in range(0, len(partial_ids), batch_size):
            batch = partial_ids[i:i + batch_size]
            self._partial_annotated_ids.extend(batch)

            logger.info(
                f"Added {len(batch)} partial scans "
                f"(total: {len(self._fully_annotated_ids)} full + "
                f"{len(self._partial_annotated_ids)} partial)"
            )

            self.run_generation(
                self._fully_annotated_ids,
                self._partial_annotated_ids,
                test_ids,
            )

        logger.info(
            f"Experiment complete: {len(self.results)} generations"
        )
        self._save_summary()

        return self.results

    def _save_generation(self, result: GenerationResult) -> None:
        """Save generation results to disk."""
        gen_dir = self.output_dir / f"gen_{result.generation:03d}"
        gen_dir.mkdir(exist_ok=True)

        # Per-scan scores
        scores_data = [
            {"study_id": s.study_id, "dice": s.dice, "assd": s.assd}
            for s in result.scores
        ]
        with open(gen_dir / "scores.json", "w") as f:
            json.dump(scores_data, f, indent=2)

        # Summary
        with open(gen_dir / "summary.json", "w") as f:
            json.dump(result.summary(), f, indent=2)

    def _save_summary(self) -> None:
        """Save overall experiment summary."""
        summary = {
            "config": asdict(self.config),
            "generations": [r.summary() for r in self.results],
        }
        with open(self.output_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
