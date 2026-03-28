"""EcoSeg - Ecological Segmentation Server."""

import os


def available_workers() -> int:
    """Return nproc - 2, minimum 1."""
    return max(1, (os.cpu_count() or 4) - 2)
