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
    model_type: str = "species"  # "species" (original) or "encoder" (EcoSegNet)
    species_names: list[str] = field(default_factory=lambda: ["lymph_node", "background"])

    # Encoder config (only used when model_type == "encoder")
    feature_dim: int = 16
    embedding_cache_dir: Optional[str] = None

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
        self.results: list[GenerationResult] = []
        self.dataset: Optional[LNQDataset] = None
        self.rng = np.random.default_rng(42)

        # Model — either species registry or EcoSegNet
        self.registry: Optional[SpeciesRegistry] = None
        self.ecosegnet: Optional['EcoSegNet'] = None
        self.emb_cache: Optional['EmbeddingCache'] = None

        # Output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment '{config.name}' using device: {self.device}")

    def setup(self) -> None:
        """Initialize dataset and create model.

        Only loads the studies needed for the first generation (validation +
        test). Remaining studies are pre-loaded in a background thread so
        training can start immediately.
        """
        import threading

        cache_path = Path(self.config.cache_dir) if self.config.cache_dir else None
        self.dataset = LNQDataset(Path(self.config.data_root), cache_dir=cache_path)
        self.dataset.discover_studies()

        if self.config.model_type == "encoder":
            self._setup_encoder()
        else:
            self._setup_species()

        # Load only the studies needed for generation 0
        validation_ids = self.dataset.get_validation_ids(self.config.num_validation)
        test_ids = self.dataset.get_test_ids(self.config.num_validation)
        gen0_ids = validation_ids + test_ids
        logger.info(f"Loading {len(gen0_ids)} studies for generation 0...")
        self.dataset.preload_studies(gen0_ids)

        # Pre-load everything else in a background thread
        all_ids = list(self.dataset._index.keys())
        remaining = [sid for sid in all_ids if sid not in self.dataset._studies]
        if remaining:
            logger.info(f"Background loading {len(remaining)} remaining studies...")
            self._bg_loader = threading.Thread(
                target=self.dataset.preload_studies,
                args=(remaining,),
                daemon=True,
            )
            self._bg_loader.start()
        else:
            self._bg_loader = None

    def _setup_species(self) -> None:
        """Initialize the original species registry."""
        self.registry = SpeciesRegistry(device=self.device)
        for name in self.config.species_names:
            self.registry.add_species(name, self.config.architecture)
        logger.info(
            f"Created {len(self.registry.species)} species: "
            f"{self.registry.species_names}"
        )

    def _setup_encoder(self) -> None:
        """Initialize EcoSegNet with shared encoder + species heads."""
        from ecoseg.models.ecosegnet import EcoSegNet, EncoderConfig
        from ecoseg.models.embedding_cache import EmbeddingCache

        enc_config = EncoderConfig(feature_dim=self.config.feature_dim)
        self.ecosegnet = EcoSegNet(enc_config).to(self.device)

        for name in self.config.species_names:
            self.ecosegnet.add_species(name)

        # Set up embedding cache
        emb_cache_dir = Path(
            self.config.embedding_cache_dir
            or self.config.cache_dir
            or str(Path.home() / ".ecoseg" / "cache")
        )
        self.emb_cache = EmbeddingCache(emb_cache_dir, self.config.feature_dim)

        logger.info(
            f"Created EcoSegNet with {len(self.ecosegnet.species_names)} species heads: "
            f"{self.ecosegnet.species_names}"
        )

    def _wait_for_background_load(self) -> None:
        """Wait for background pre-loading to finish, if still running."""
        if hasattr(self, '_bg_loader') and self._bg_loader is not None:
            if self._bg_loader.is_alive():
                logger.info("Waiting for background data loading to complete...")
                self._bg_loader.join()
                logger.info("Background loading complete.")
            self._bg_loader = None

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

        Pre-normalizes all volumes and pipelines GPU inference with CPU
        scoring. Volumes are pre-transferred to GPU one ahead so the
        GPU never waits for CPU→GPU transfer.

        Returns (scores, inference_time_s).
        """
        from concurrent.futures import ThreadPoolExecutor
        import torch

        t0 = time.time()
        ln_index = self.registry.species_names.index("lymph_node")

        # Pre-normalize all volumes (CPU work, done upfront)
        studies = []
        volumes_normalized = []
        for sid in study_ids:
            study = self.dataset.load_study(sid)
            studies.append(study)
            volumes_normalized.append(normalize_ct(study.volume))

        scoring_futures = []

        with ThreadPoolExecutor(max_workers=available_workers()) as scorer:
            # Pre-transfer first volume to GPU
            device = self.registry.device
            next_vol_gpu = torch.tensor(
                volumes_normalized[0], dtype=torch.float32, device=device
            ) if len(volumes_normalized) > 0 else None

            for i in range(len(study_ids)):
                # Current volume is already on GPU
                vol_gpu = next_vol_gpu

                # Start transferring next volume while GPU runs inference
                if i + 1 < len(study_ids):
                    # Use non-blocking transfer via pinned memory
                    next_np = volumes_normalized[i + 1]
                    next_vol_gpu = torch.tensor(
                        next_np, dtype=torch.float32, device=device
                    )
                else:
                    next_vol_gpu = None

                # Run inference using pre-transferred GPU tensor
                labels, fitness_map = infer_volume(
                    self.registry, volumes_normalized[i], self.config.inference
                )

                prediction = (labels == ln_index).astype(np.uint8)

                study = studies[i]
                if study.seg_mask is not None:
                    ground_truth = (study.seg_mask > 0).astype(np.uint8)
                    future = scorer.submit(
                        score_segmentation,
                        study.study_id, prediction, ground_truth, study.spacing,
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

    # --- Encoder-based methods ---

    def train_on_studies_encoder(
        self,
        fully_ids: list[str],
        partial_ids: list[str],
    ) -> float:
        """Train species heads on pre-computed embeddings."""
        from scipy import ndimage
        from ecoseg.models.head_trainer import HeadTrainingConfig, train_species_head

        t0 = time.time()

        # Re-initialize species heads
        for name in self.config.species_names:
            self.ecosegnet.add_species(name)
        self.ecosegnet.to(self.device)

        volumes = []
        embeddings_list = []
        ln_masks = []
        bg_masks = []

        def _prepare(sid, annotation_type):
            study = self.dataset.load_study(sid)
            vol = normalize_ct(study.volume)
            volumes.append(vol)

            # Get or compute embeddings
            emb = self.emb_cache.encode_and_cache(
                self.ecosegnet, sid, vol, self.device
            )
            embeddings_list.append(emb)

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
            _prepare(sid, "fully_annotated")
        for sid in partial_ids:
            _prepare(sid, "partially_annotated")

        logger.info(
            f"Training heads on {len(fully_ids)} full + {len(partial_ids)} partial scans "
            f"({len(ln_masks)} with LN masks, {len(bg_masks)} with BG masks)"
        )

        head_config = HeadTrainingConfig()

        # Train lymph node head
        ln_head = self.ecosegnet.species_heads["lymph_node"]
        if ln_masks:
            # Use embeddings corresponding to scans that have LN masks
            ln_embs = embeddings_list[:len(ln_masks)]
            train_species_head(
                ln_head, ln_embs, ln_masks,
                config=head_config, device=self.device,
                volumes=volumes[:len(ln_masks)], rng=self.rng,
            )

        # Train background head
        bg_head = self.ecosegnet.species_heads["background"]
        if bg_masks:
            bg_embs = embeddings_list[:len(bg_masks)]
            train_species_head(
                bg_head, bg_embs, bg_masks,
                config=head_config, device=self.device,
                volumes=volumes[:len(bg_masks)], rng=self.rng,
            )

        return time.time() - t0

    def evaluate_encoder(self, study_ids: list[str]) -> tuple[list[SegmentationScore], float]:
        """Run inference using EcoSegNet and score against ground truth."""
        from concurrent.futures import ThreadPoolExecutor

        t0 = time.time()
        ln_index = self.ecosegnet.species_names.index("lymph_node")

        scoring_futures = []

        with ThreadPoolExecutor(max_workers=available_workers()) as scorer:
            for i, sid in enumerate(study_ids):
                study = self.dataset.load_study(sid)
                vol = normalize_ct(study.volume)

                # Get or compute embeddings
                emb = self.emb_cache.encode_and_cache(
                    self.ecosegnet, sid, vol, self.device
                )

                # Run species heads on embeddings
                emb_t = torch.tensor(emb, dtype=torch.float32, device=self.device)
                emb_t = emb_t.unsqueeze(0)  # (1, feature_dim, D, H, W)

                with torch.no_grad():
                    labels, fitness = self.ecosegnet.segment(emb_t)

                prediction = (labels.squeeze(0) == ln_index).cpu().numpy().astype(np.uint8)

                if study.seg_mask is not None:
                    ground_truth = (study.seg_mask > 0).astype(np.uint8)
                    future = scorer.submit(
                        score_segmentation,
                        sid, prediction, ground_truth, study.spacing,
                    )
                    scoring_futures.append(future)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Inferred {i + 1}/{len(study_ids)} scans")

            scores = []
            for future in scoring_futures:
                scores.append(future.result())

        elapsed = time.time() - t0
        return scores, elapsed

    def _collect_pending_evaluation(self) -> None:
        """If there's a pending evaluation from the previous generation,
        collect results and log them."""
        if not hasattr(self, '_pending_eval') or self._pending_eval is None:
            return

        gen, total_train, train_time, future = self._pending_eval
        self._pending_eval = None

        scores, infer_time = future.result()
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
            f"  Gen {gen} results: Mean Dice: {result.mean_dice:.4f}, "
            f"Mean ASSD: {result.mean_assd:.2f}mm, "
            f"Train: {train_time:.1f}s, Infer: {infer_time:.1f}s"
        )
        self._save_generation(result)

    def run_generation(
        self,
        fully_ids: list[str],
        partial_ids: list[str],
        eval_ids: list[str],
    ) -> None:
        """Run one generation: collect previous eval, train, launch eval in background.

        Evaluation (GPU inference + CPU scoring) runs in a background thread
        so that the next generation's CPU-side data preparation can overlap.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        # Collect results from previous generation's evaluation (if any)
        self._collect_pending_evaluation()

        gen = len(self.results)
        total_train = len(fully_ids) + len(partial_ids)
        logger.info(
            f"=== Generation {gen}: training on {total_train} scans "
            f"({len(fully_ids)} full + {len(partial_ids)} partial), "
            f"evaluating on {len(eval_ids)} ==="
        )

        if self.config.model_type == "encoder":
            train_time = self.train_on_studies_encoder(fully_ids, partial_ids)
        else:
            train_time = self.train_on_studies(fully_ids, partial_ids)

        # Launch evaluation in background — GPU does inference while
        # the main thread can prepare the next generation's data
        if not hasattr(self, '_eval_pool'):
            self._eval_pool = ThreadPoolExecutor(max_workers=1)

        eval_fn = self.evaluate_encoder if self.config.model_type == "encoder" else self.evaluate
        future = self._eval_pool.submit(eval_fn, eval_ids)
        self._pending_eval = (gen, total_train, train_time, future)

    def run_full_experiment(self) -> list[GenerationResult]:
        """Run the complete incremental experiment, resuming from checkpoint if available.

        Pipeline: while gen N evaluates on the GPU, gen N+1's data
        preparation (patch extraction, normalization) runs on CPU.

        Training is cumulative: each generation trains on the 20 fully-
        annotated scans PLUS all partial scans added so far.
        """
        self.setup()

        validation_ids = self.dataset.get_validation_ids(self.config.num_validation)
        test_ids = self.dataset.get_test_ids(self.config.num_validation)

        # Try to resume from checkpoint
        resumed = self.load_checkpoint()

        if not resumed:
            self._fully_annotated_ids = list(validation_ids)
            self._partial_annotated_ids = []

        logger.info(
            f"Validation: {len(validation_ids)} studies, "
            f"Test: {len(test_ids)} studies"
            + (f" (resumed from gen {len(self.results)})" if resumed else "")
        )

        # Get partially annotated study IDs (deterministic order)
        all_ids = set(self.dataset._index.keys())
        fully_ids = set(validation_ids) | set(test_ids)
        partial_ids = sorted(all_ids - fully_ids)

        # Build list of all generations we need to run
        batch_size = self.config.batch_size_partial
        all_batches = []
        for i in range(0, len(partial_ids), batch_size):
            all_batches.append(partial_ids[i:i + batch_size])

        # Figure out which generation to start from
        start_gen = len(self.results)

        if start_gen == 0:
            # Generation 0: train on fully-annotated validation set only
            self.run_generation(
                self._fully_annotated_ids, [], test_ids,
            )
            start_gen = 1

        # Ensure background loading is done before we need partial scans
        self._wait_for_background_load()

        # Run remaining generations, skipping already-completed ones
        for batch_idx in range(len(all_batches)):
            gen_num = batch_idx + 1  # gen 0 is the initial fully-annotated run

            if gen_num < start_gen:
                # Already completed — just accumulate the partial IDs
                # (only needed if not restored from checkpoint)
                if not resumed:
                    self._partial_annotated_ids.extend(all_batches[batch_idx])
                continue

            # Add this batch
            batch = all_batches[batch_idx]
            self._partial_annotated_ids.extend(batch)

            logger.info(
                f"Added {len(batch)} partial scans "
                f"(total: {len(self._fully_annotated_ids)} full + "
                f"{len(self._partial_annotated_ids)} partial)"
            )

            self.run_generation(
                self._fully_annotated_ids,
                list(self._partial_annotated_ids),
                test_ids,
            )

        # Collect final generation's evaluation
        self._collect_pending_evaluation()

        logger.info(
            f"Experiment complete: {len(self.results)} generations"
        )
        self._save_summary()

        return self.results

    def save_checkpoint(self) -> None:
        """Save experiment state so it can be resumed later.

        Saves: generation count, accumulated study IDs, species weights,
        and all generation results.
        """
        ckpt_path = self.output_dir / "checkpoint.pt"
        ckpt = {
            "generation": len(self.results),
            "fully_annotated_ids": getattr(self, '_fully_annotated_ids', []),
            "partial_annotated_ids": getattr(self, '_partial_annotated_ids', []),
            "species_states": {
                name: species.state_dict()
                for name, species in self.registry.species.items()
            },
            "results": [
                {
                    "generation": r.generation,
                    "num_training_scans": r.num_training_scans,
                    "mean_dice": r.mean_dice,
                    "mean_assd": r.mean_assd,
                    "training_time_s": r.training_time_s,
                    "inference_time_s": r.inference_time_s,
                    "scores": [
                        {"study_id": s.study_id, "dice": s.dice, "assd": s.assd}
                        for s in r.scores
                    ],
                }
                for r in self.results
            ],
        }
        torch.save(ckpt, ckpt_path)
        logger.info(f"Checkpoint saved: generation {len(self.results)}")

    def load_checkpoint(self) -> bool:
        """Load experiment state from checkpoint. Returns True if loaded."""
        ckpt_path = self.output_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            return False

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Restore species weights
        from ecoseg.models.species import SpeciesModel
        for name, state in ckpt["species_states"].items():
            if name in self.registry.species:
                self.registry.species[name] = SpeciesModel.from_state_dict(
                    state, device=self.device
                )

        # Restore accumulated IDs
        self._fully_annotated_ids = ckpt["fully_annotated_ids"]
        self._partial_annotated_ids = ckpt["partial_annotated_ids"]

        # Restore results
        self.results = []
        for r_data in ckpt["results"]:
            scores = [
                SegmentationScore(
                    study_id=s["study_id"], dice=s["dice"], assd=s["assd"]
                )
                for s in r_data["scores"]
            ]
            self.results.append(GenerationResult(
                generation=r_data["generation"],
                num_training_scans=r_data["num_training_scans"],
                scores=scores,
                mean_dice=r_data["mean_dice"],
                mean_assd=r_data["mean_assd"],
                training_time_s=r_data["training_time_s"],
                inference_time_s=r_data["inference_time_s"],
            ))

        logger.info(
            f"Resumed from checkpoint: generation {len(self.results)}, "
            f"{len(self._partial_annotated_ids)} partial scans accumulated"
        )
        return True

    def _save_generation(self, result: GenerationResult) -> None:
        """Save generation results and checkpoint to disk."""
        gen_dir = self.output_dir / f"gen_{result.generation:03d}"
        gen_dir.mkdir(exist_ok=True)

        # Per-scan scores
        scores_data = [
            {"study_id": s.study_id, "dice": s.dice, "assd": s.assd}
            for s in result.scores
        ]
        with open(gen_dir / "scores.json", "w") as f:
            json.dump(scores_data, f, indent=2)

        # Save checkpoint after every generation
        self.save_checkpoint()

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
