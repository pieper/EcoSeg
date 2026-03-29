"""Embedding cache for pre-computed encoder features.

Stores per-voxel embeddings as zarr arrays with blosc/lz4 compression.
Embeddings are computed once per scan by the frozen encoder and reused
across all generations — only the species heads need retraining.

A 512x512x300 volume at 16-dim float16 = ~2.5GB per scan.
120 eval scans = ~300GB, which fits in the cache volume.
"""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """On-disk cache for dense per-voxel embeddings.

    Uses zarr v3 with blosc/lz4 compression. Each scan's embeddings
    are stored as a separate zarr array.
    """

    def __init__(self, cache_dir: Path, feature_dim: int = 16):
        self.cache_dir = Path(cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dim = feature_dim
        logger.info(f"Embedding cache: {self.cache_dir} (dim={feature_dim})")

    def _path(self, study_id: str) -> Path:
        return self.cache_dir / f"{study_id}.zarr"

    def has(self, study_id: str) -> bool:
        return self._path(study_id).exists()

    def save(self, study_id: str, embeddings: np.ndarray) -> None:
        """Save embeddings to cache.

        Args:
            study_id: study identifier
            embeddings: (feature_dim, D, H, W) float16 array
        """
        import zarr
        from zarr.codecs import BloscCodec

        path = self._path(study_id)
        comp = BloscCodec(cname='lz4', clevel=3)

        root = zarr.open_group(str(path), mode='w')
        root.create_array(
            'embeddings',
            data=embeddings,
            compressors=[comp],
            chunks=(
                embeddings.shape[0],  # All features together
                min(64, embeddings.shape[1]),
                min(128, embeddings.shape[2]),
                min(128, embeddings.shape[3]),
            ),
        )
        root.attrs['feature_dim'] = int(embeddings.shape[0])
        root.attrs['shape'] = list(embeddings.shape)

    def load(self, study_id: str) -> Optional[np.ndarray]:
        """Load embeddings from cache.

        Returns:
            (feature_dim, D, H, W) float16 array, or None on miss.
        """
        path = self._path(study_id)
        if not path.exists():
            return None
        try:
            import zarr
            root = zarr.open_group(str(path), mode='r')
            return root['embeddings'][:]
        except Exception as e:
            logger.warning(f"Embedding cache read failed for {study_id}: {e}")
            return None

    def extract_at_coords(
        self,
        study_id: str,
        coords: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extract feature vectors at specific voxel coordinates from cache.

        Reads only the needed chunks from zarr, avoiding loading the full
        embedding volume into memory. For a scan with 1000 labeled voxels,
        this reads ~64KB instead of ~1GB.

        Args:
            study_id: study identifier
            coords: (N, 3) int array of (z, y, x) voxel coordinates

        Returns:
            (N, feature_dim) float32 array, or None on cache miss
        """
        path = self._path(study_id)
        if not path.exists():
            return None
        try:
            import zarr
            root = zarr.open_group(str(path), mode='r')
            emb = root['embeddings']  # Don't load full array — lazy access

            # Extract feature vectors at each coordinate
            features = np.zeros((len(coords), self.feature_dim), dtype=np.float32)
            for i, (z, y, x) in enumerate(coords):
                features[i] = emb[:, z, y, x]

            return features
        except Exception as e:
            logger.warning(f"Feature extraction failed for {study_id}: {e}")
            return None

    def ensure_cached(
        self,
        model: 'EcoSegNet',
        study_id: str,
        volume: np.ndarray,
        device: torch.device,
    ) -> bool:
        """Ensure embeddings are cached for a study. Returns True if cached."""
        if self.has(study_id):
            return True
        self.encode_and_cache(model, study_id, volume, device)
        return self.has(study_id)

    def encode_and_cache(
        self,
        model: 'EcoSegNet',
        study_id: str,
        volume: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        """Encode a volume and cache the result. Returns embeddings as numpy.

        If already cached, loads from disk. Otherwise computes using the
        encoder and saves to disk.

        Args:
            model: the EcoSegNet with frozen encoder
            study_id: study identifier
            volume: (D, H, W) normalized CT volume in [0, 1]
            device: compute device

        Returns:
            (feature_dim, D, H, W) float16 numpy array
        """
        # Check cache first
        cached = self.load(study_id)
        if cached is not None:
            return cached

        # Compute embeddings
        vol_t = torch.tensor(volume, dtype=torch.float32, device=device)
        vol_t = vol_t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        with torch.no_grad():
            embeddings = model.encode_sliding_window(vol_t)

        # Convert to float16 numpy for caching
        emb_np = embeddings.squeeze(0).cpu().half().numpy()  # (feature_dim, D, H, W)

        # Save to cache
        self.save(study_id, emb_np)
        logger.info(
            f"Cached embeddings for {study_id}: "
            f"shape={emb_np.shape}, "
            f"size={emb_np.nbytes / 1e9:.2f}GB"
        )

        return emb_np

    def encode_batch(
        self,
        model: 'EcoSegNet',
        study_ids: list[str],
        volumes: dict[str, np.ndarray],
        device: torch.device,
    ) -> dict[str, np.ndarray]:
        """Encode multiple volumes, using cache where available.

        Args:
            model: the EcoSegNet with frozen encoder
            study_ids: list of study IDs to encode
            volumes: dict mapping study_id to (D, H, W) normalized volume
            device: compute device

        Returns:
            Dict mapping study_id to (feature_dim, D, H, W) float16 arrays
        """
        results = {}
        to_compute = []

        for sid in study_ids:
            cached = self.load(sid)
            if cached is not None:
                results[sid] = cached
            else:
                to_compute.append(sid)

        if to_compute:
            logger.info(
                f"Encoding {len(to_compute)} volumes "
                f"({len(results)} cached)..."
            )

        for i, sid in enumerate(to_compute):
            if sid not in volumes:
                logger.warning(f"No volume for {sid}, skipping")
                continue
            results[sid] = self.encode_and_cache(model, sid, volumes[sid], device)
            if (i + 1) % 10 == 0 or i + 1 == len(to_compute):
                logger.info(f"  Encoded {i + 1}/{len(to_compute)} volumes")

        return results
