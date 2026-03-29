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
            img_size=self.config.patch_size,
            in_channels=1,
            out_channels=1,  # We won't use the decoder output
            feature_size=48,
            use_v2=True,
        )

        if self.config.pretrained:
            try:
                from monai.bundle import download
                # Download pre-trained weights
                weight_path = download(
                    name="swin_unetr_btcv_segmentation",
                    bundle_dir=str(Path.home() / ".ecoseg" / "models"),
                )
                # Load only the encoder weights
                state = torch.load(
                    Path(weight_path) / "models" / "model.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                # Filter to only swinViT (encoder) weights
                encoder_state = {
                    k: v for k, v in state.items()
                    if k.startswith("swinViT.") or k.startswith("encoder")
                }
                if encoder_state:
                    model.load_state_dict(encoder_state, strict=False)
                    logger.info("Loaded pre-trained SwinUNETR encoder weights")
                else:
                    model.load_state_dict(state, strict=False)
                    logger.info("Loaded pre-trained SwinUNETR weights (full model)")
            except Exception as e:
                logger.warning(f"Could not load pre-trained weights: {e}")
                logger.info("Using randomly initialized encoder")

        # We only need the encoder part — extract feature maps
        return model.swinViT

    def _get_encoder_out_dim(self) -> int:
        """Determine the encoder output channel count."""
        # SwinUNETR with feature_size=48 produces features at multiple scales.
        # The deepest features have 48 * 8 = 384 channels at 1/32 resolution.
        # For dense per-voxel embeddings, we use the first-scale features (48 channels)
        # which are at 1/2 resolution, then upsample.
        return 48

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

    def encode(self, volume: torch.Tensor) -> torch.Tensor:
        """Compute dense embeddings for a volume.

        Args:
            volume: (batch, 1, D, H, W) normalized CT volume

        Returns:
            embeddings: (batch, feature_dim, D, H, W) projected features
        """
        with torch.no_grad():
            # SwinViT returns features at multiple scales
            # hidden_states_out is a list of feature maps at different resolutions
            features = self.encoder(volume)

            # Use the first scale features (48 channels, 1/2 resolution)
            # and upsample to full resolution
            feat = features[0]  # (batch, 48, D/2, H/2, W/2)

            # Upsample to match input resolution
            feat = nn.functional.interpolate(
                feat, size=volume.shape[2:], mode='trilinear', align_corners=False
            )

        # Project to lower-dimensional embedding space (projector IS trainable)
        embeddings = self.projector(feat)

        return embeddings

    def encode_sliding_window(
        self,
        volume: torch.Tensor,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.25,
    ) -> torch.Tensor:
        """Encode a full-size volume using sliding window inference.

        Handles volumes larger than the encoder's trained patch size by
        processing overlapping patches and averaging embeddings.

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
            sw_batch_size=4,
            overlap=overlap,
            mode='gaussian',
        )

        with torch.no_grad():
            # Use the encoder + projector as the network for sliding window
            def _encode_patch(x):
                features = self.encoder(x)
                feat = features[0]
                feat = nn.functional.interpolate(
                    feat, size=x.shape[2:], mode='trilinear', align_corners=False
                )
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
