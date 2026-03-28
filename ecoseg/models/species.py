"""Species models for ecological segmentation.

Each species is a small neural network that takes a volumetric patch
and returns a fitness score (0-1) indicating how well that species
matches the local image environment.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional


class ThreeLayerCNN(nn.Module):
    """V1 species architecture: 3-layer CNN on 32x32x32 patches.

    Input: (batch, 1, 32, 32, 32) - single-channel CT patch
    Output: (batch, 1) - fitness score in [0, 1]
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> 16x16x16
            nn.Dropout3d(dropout),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> 8x8x8
            nn.Dropout3d(dropout),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # -> 1x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResBlock3d(nn.Module):
    """3D residual block with optional downsampling."""

    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class ResNetSpecies(nn.Module):
    """Residual species architecture with multi-scale pooling.

    Residual connections help learn "what's different about this patch"
    rather than memorizing absolute patterns. Multi-scale average pooling
    captures features at different spatial scales, helping with lymph
    nodes of varying sizes.

    Input: (batch, 1, 32, 32, 32)
    Output: (batch, 1) - fitness score in [0, 1]
    ~300K parameters
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        # Stem: project to feature channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(),
        )

        # Scale 1: 32^3 -> residual -> pool to 16^3
        self.block1 = ResBlock3d(24, dropout)
        self.pool1 = nn.MaxPool3d(2)

        # Scale 2: 16^3 -> residual -> pool to 8^3
        self.up2 = nn.Sequential(
            nn.Conv3d(24, 48, kernel_size=1),
            nn.BatchNorm3d(48),
            nn.ReLU(),
        )
        self.block2 = ResBlock3d(48, dropout)
        self.pool2 = nn.MaxPool3d(2)

        # Scale 3: 8^3 -> residual -> pool to 4^3
        self.up3 = nn.Sequential(
            nn.Conv3d(48, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.block3 = ResBlock3d(64, dropout)

        # Multi-scale pooling: pool each scale to 1x1x1 and concatenate
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Classifier on concatenated multi-scale features (24 + 48 + 64 = 136)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(136, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale 1
        s1 = self.block1(self.stem(x))
        g1 = self.gap(s1)

        # Scale 2
        s2 = self.block2(self.up2(self.pool1(s1)))
        g2 = self.gap(s2)

        # Scale 3
        s3 = self.block3(self.up3(self.pool2(s2)))
        g3 = self.gap(s3)

        # Concatenate multi-scale features
        multi = torch.cat([g1, g2, g3], dim=1)
        return self.classifier(multi)


# Registry of available architectures
ARCHITECTURES = {
    "cnn3": ThreeLayerCNN,
    "resnet": ResNetSpecies,
}


@dataclass
class SpeciesModel:
    """A segment species with its neural network and metadata.

    Each species is an independent binary classifier trained with BCE loss.
    The fitness score output represents how strongly a patch belongs to
    this species.
    """

    name: str
    network: nn.Module
    architecture: str = "cnn3"
    generation: int = 0
    training_scans: list[str] = field(default_factory=list)

    def fitness(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute fitness scores for a batch of patches.

        Args:
            patches: (batch, 1, 32, 32, 32) tensor of image patches

        Returns:
            (batch,) tensor of fitness scores in [0, 1]
        """
        self.network.eval()
        with torch.no_grad():
            return self.network(patches).squeeze(-1)

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "generation": self.generation,
            "training_scans": self.training_scans,
            "weights": self.network.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: torch.device | str = "cpu") -> "SpeciesModel":
        arch_cls = ARCHITECTURES[state["architecture"]]
        network = arch_cls().to(device)
        network.load_state_dict(state["weights"])
        return cls(
            name=state["name"],
            network=network,
            architecture=state["architecture"],
            generation=state["generation"],
            training_scans=state["training_scans"],
        )


class SpeciesRegistry:
    """Manages all species in an ecosystem.

    Handles batched inference across all species for efficient
    GPU utilization.
    """

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.species: dict[str, SpeciesModel] = {}

    def add_species(
        self,
        name: str,
        architecture: str = "cnn3",
    ) -> SpeciesModel:
        arch_cls = ARCHITECTURES[architecture]
        network = arch_cls().to(self.device)
        species = SpeciesModel(
            name=name,
            network=network,
            architecture=architecture,
        )
        self.species[name] = species
        return species

    def fitness_all(self, patches: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute fitness for all species on the same patches.

        Args:
            patches: (batch, 1, 32, 32, 32) tensor

        Returns:
            Dict mapping species name to (batch,) fitness tensors
        """
        results = {}
        for name, species in self.species.items():
            results[name] = species.fitness(patches)
        return results

    def inference_argmax(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference mode: per-patch argmax over all species.

        Returns:
            labels: (batch,) int tensor - index of winning species
            fitness: (batch,) float tensor - fitness of the winner
        """
        all_fitness = self.fitness_all(patches)
        # Stack into (num_species, batch) then find argmax
        names = list(all_fitness.keys())
        stacked = torch.stack([all_fitness[n] for n in names], dim=0)
        fitness, labels = stacked.max(dim=0)
        return labels, fitness

    @property
    def species_names(self) -> list[str]:
        return list(self.species.keys())
