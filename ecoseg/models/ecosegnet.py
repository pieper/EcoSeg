"""EcoSegNet: Shared encoder with per-species decoder heads.

Uses a pre-trained SwinUNETR encoder (frozen) to produce dense per-voxel
feature embeddings. Each species is a lightweight decoder head that maps
embeddings → per-voxel fitness.

Architecture:
    CT Volume → [Frozen Encoder] → per-voxel embeddings (the "terrain")
    Embeddings → [Species Head A] → per-voxel fitness A
    Embeddings → [Species Head B] → per-voxel fitness B
    ...
    Competition: per-voxel argmax across species

The encoder runs once per scan. Species heads train in seconds on
frozen embeddings, enabling interactive feedback.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """Configuration for the shared encoder."""
    feature_dim: int = 16  # Project encoder features to this dim for caching
    patch_size: tuple[int, int, int] = (96, 96, 96)
    encoder_name: str = "swinunetr"
    pretrained: bool = True
    frozen: bool = True


class FeatureProjector(nn.Module):
    """Projects encoder features to a lower-dimensional embedding space.

    Reduces memory for caching while preserving discriminative information.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SpeciesHead(nn.Module):
    """Lightweight per-species decoder head.

    Maps per-voxel embeddings → per-voxel fitness score.
    Small enough to train in seconds on frozen embeddings.

    Architecture: two 1x1x1 conv layers with dropout.
    ~(feature_dim * 32 + 32) parameters = ~550 for feature_dim=16.
    """

    def __init__(self, feature_dim: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv3d(feature_dim, 32, kernel_size=1),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, feature_dim, D, H, W)
        Returns:
            fitness: (batch, 1, D, H, W) in [0, 1]
        """
        return self.head(embeddings)


class EcoSegNet(nn.Module):
    """Shared encoder with dynamically-added species heads.

    The encoder is frozen after initialization — it produces the
    "ecosystem terrain" (feature embeddings) that species adapt to.
    Species heads are lightweight and can be added/removed at any time.
    """

    def __init__(self, config: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.config = config

        # Build encoder
        self.encoder = self._build_encoder()
        self.encoder_out_dim = self._get_encoder_out_dim()

        # Projection to lower-dimensional embedding space
        self.projector = FeatureProjector(self.encoder_out_dim, config.feature_dim)

        # Per-species decoder heads (dynamically added)
        self.species_heads: nn.ModuleDict = nn.ModuleDict()

        # Freeze encoder if configured
        if config.frozen:
            self._freeze_encoder()

    def _build_encoder(self) -> nn.Module:
        """Build the SwinUNETR encoder from MONAI."""
        from monai.networks.nets import SwinUNETR

        model = SwinUNETR(
            in_channels=1,
            out_channels=1,  # We won't use the decoder output
            feature_size=48,
            spatial_dims=3,
            use_v2=True,
        )

        if self.config.pretrained:
            try:
                # Try to download BTCV pre-trained weights
                weight_dir = Path.home() / ".ecoseg" / "models"
                weight_dir.mkdir(parents=True, exist_ok=True)
                weight_file = weight_dir / "swin_unetr_btcv.pt"

                if not weight_file.exists():
                    logger.info("Downloading pre-trained SwinUNETR weights...")
                    url = (
                        "https://github.com/Project-MONAI/MONAI-extra-test-data/"
                        "releases/download/0.8.1/model_swinvit.pt"
                    )
                    torch.hub.download_url_to_file(url, str(weight_file))

                state = torch.load(str(weight_file), map_location="cpu", weights_only=True)
                # The pre-trained file contains swinViT weights
                if "state_dict" in state:
                    state = state["state_dict"]
                model.swinViT.load_state_dict(state, strict=False)
                logger.info("Loaded pre-trained SwinUNETR encoder weights")
            except Exception as e:
                logger.warning(f"Could not load pre-trained weights: {e}")
                logger.info("Using randomly initialized encoder")

        # We only need the encoder part — extract feature maps
        return model.swinViT

    def _get_encoder_out_dim(self) -> int:
        """Determine the encoder output channel count.

        SwinUNETR with feature_size=48 produces features at 5 scales:
          Stage 0: 48 channels at 1/2 resolution  (local texture)
          Stage 1: 96 channels at 1/4 resolution  (local structure)
          Stage 2: 192 channels at 1/8 resolution (regional context)
          Stage 3: 384 channels at 1/16 resolution (body region)
          Stage 4: 768 channels at 1/32 resolution (global context)

        We concatenate stages 0-3 upsampled to full resolution:
          48 + 96 + 192 + 384 = 720 channels
        (Skip stage 4 — too coarse for voxel-level discrimination)
        """
        return 48 + 96 + 192 + 384  # 720

    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen — only species heads will train")

    def add_species(self, name: str) -> SpeciesHead:
        """Add a new species decoder head."""
        head = SpeciesHead(self.config.feature_dim)
        self.species_heads[name] = head
        logger.info(f"Added species head: {name}")
        return head

    def remove_species(self, name: str):
        """Remove a species decoder head."""
        if name in self.species_heads:
            del self.species_heads[name]

    def _multiscale_features(self, volume: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate multi-scale features from the encoder.

        Upsamples features from stages 0-3 to full resolution and
        concatenates them, giving each voxel a rich feature vector
        with both local texture and broad anatomical context.

        Args:
            volume: (batch, 1, D, H, W)

        Returns:
            (batch, 720, D, H, W) concatenated multi-scale features
        """
        target_size = volume.shape[2:]
        features = self.encoder(volume)  # List of 5 feature maps

        # Upsample stages 0-3 to full resolution and concatenate
        upsampled = []
        for stage_idx in range(4):  # Skip stage 4 (too coarse)
            feat = features[stage_idx]
            if feat.shape[2:] != target_size:
                feat = nn.functional.interpolate(
                    feat, size=target_size, mode='trilinear', align_corners=False
                )
            upsampled.append(feat)

        return torch.cat(upsampled, dim=1)  # (batch, 720, D, H, W)

    def encode(self, volume: torch.Tensor) -> torch.Tensor:
        """Compute dense embeddings for a volume.

        Uses multi-scale features (stages 0-3 concatenated) projected
        to a lower-dimensional embedding space.

        Args:
            volume: (batch, 1, D, H, W) normalized CT volume

        Returns:
            embeddings: (batch, feature_dim, D, H, W) projected features
        """
        with torch.no_grad():
            feat = self._multiscale_features(volume)

        embeddings = self.projector(feat)
        return embeddings

    def encode_sliding_window(
        self,
        volume: torch.Tensor,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.25,
    ) -> torch.Tensor:
        """Encode a full-size volume using sliding window inference.

        Uses multi-scale features from the encoder, projected to the
        embedding space. Handles volumes larger than the patch size.

        Args:
            volume: (1, 1, D, H, W) normalized CT volume
            patch_size: encoder patch size
            overlap: fraction of overlap between patches

        Returns:
            embeddings: (1, feature_dim, D, H, W) projected features
        """
        from monai.inferers import SlidingWindowInferer

        inferer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=2,  # Reduced — multi-scale uses more VRAM
            overlap=overlap,
            mode='gaussian',
        )

        with torch.no_grad():
            def _encode_patch(x):
                feat = self._multiscale_features(x)
                return self.projector(feat)

            embeddings = inferer(volume, _encode_patch)

        return embeddings

    def species_fitness(
        self,
        embeddings: torch.Tensor,
        species_name: str,
    ) -> torch.Tensor:
        """Compute per-voxel fitness for one species.

        Args:
            embeddings: (batch, feature_dim, D, H, W)
            species_name: which species head to use

        Returns:
            fitness: (batch, 1, D, H, W) in [0, 1]
        """
        return self.species_heads[species_name](embeddings)

    def all_fitness(
        self,
        embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute per-voxel fitness for all species.

        Returns:
            Dict mapping species name to (batch, 1, D, H, W) fitness maps.
        """
        return {
            name: head(embeddings)
            for name, head in self.species_heads.items()
        }

    def segment(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment by per-voxel argmax across species.

        Returns:
            labels: (batch, D, H, W) int tensor — index of winning species
            fitness: (batch, D, H, W) float tensor — fitness of winner
        """
        all_fit = self.all_fitness(embeddings)
        names = list(all_fit.keys())
        # Stack to (batch, num_species, D, H, W)
        stacked = torch.cat([all_fit[n] for n in names], dim=1)
        fitness, labels = stacked.max(dim=1)
        return labels, fitness

    @property
    def species_names(self) -> list[str]:
        return list(self.species_heads.keys())

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable parameters (heads + projector)."""
        params = list(self.projector.parameters())
        for head in self.species_heads.values():
            params.extend(head.parameters())
        return params
