"""Trajectory database: load pre-collected episodes and nearest-neighbor lookup."""
import os
import glob
import numpy as np


class TrajectoryDB:
    """Stores trajectory keys (6-dim object positions) and action sequences.

    Each .npz file in the directory must contain:
        - 'key': np.ndarray shape (6,) — [obj1_x, obj1_y, obj2_x, obj2_y, obj3_x, obj3_y]
        - 'actions': np.ndarray shape (T, 8) — full action sequence for the episode
    """

    def __init__(self, data_dir: str):
        self.keys: np.ndarray = np.empty((0, 6), dtype=np.float32)
        self.actions: list[np.ndarray] = []
        self._load(data_dir)

    @property
    def size(self) -> int:
        return len(self.actions)

    def _load(self, data_dir: str) -> None:
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not files:
            return
        keys_list = []
        for f in files:
            data = np.load(f)
            keys_list.append(data["key"].astype(np.float32))
            self.actions.append(data["actions"].astype(np.float32))
        self.keys = np.stack(keys_list, axis=0)

    def query(self, key: np.ndarray) -> tuple[np.ndarray, float]:
        """Find nearest trajectory by Euclidean distance.

        Args:
            key: shape (6,) query vector

        Returns:
            (actions, distance) — actions shape (T, 8), distance is L2 norm
        """
        diffs = self.keys - key.astype(np.float32).reshape(1, 6)
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        return self.actions[idx], float(dists[idx])
