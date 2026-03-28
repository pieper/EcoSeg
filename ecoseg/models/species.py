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

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> 16x16x16
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> 8x8x8
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # -> 1x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# Registry of available architectures
ARCHITECTURES = {
    "cnn3": ThreeLayerCNN,
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
