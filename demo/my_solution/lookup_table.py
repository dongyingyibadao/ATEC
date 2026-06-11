"""IK 查表模块: 加载离线预计算表, 最近邻查询.

每个物体一个表, 预计算时使用该物体专用的抓取朝向。
运行时用最近邻 (不插值) 避免中间状态碰撞/越限。
"""
import os
import numpy as np
from .config import DATA_DIR, OBJECT_ORDER


class IKLookupTable:
    """加载离线预计算的 per-object IK 查表, 提供最近邻查询.

    每个表 (.npy) 的格式:
        {
            "x_grid": np.ndarray (Nx,),
            "y_grid": np.ndarray (Ny,),
            "waypoints": np.ndarray (Nx, Ny, 3, 8),  # [pregrasp, grasp, lift] × 8 joints
            "valid": np.ndarray (Nx, Ny), bool
        }
    """

    def __init__(self):
        self.tables: dict[str, dict | None] = {}
        for obj_name in OBJECT_ORDER:
            path = os.path.join(DATA_DIR, f"ik_{obj_name}.npy")
            if os.path.isfile(path):
                data = np.load(path, allow_pickle=True).item()
                self.tables[obj_name] = data
            else:
                self.tables[obj_name] = None

    def query(self, obj_name: str, x: float, y: float) -> dict[str, np.ndarray] | None:
        """最近邻查询: 给定物体世界坐标 (x,y), 返回最近有效格点的 waypoints.

        Args:
            obj_name: "sugar", "mustard", or "banana"
            x: world x coordinate
            y: world y coordinate

        Returns:
            dict with keys: "pregrasp", "grasp", "lift"
            每个值是 np.ndarray shape (8,) — 绝对关节角 (含夹爪位)
            Returns None if table not loaded or no valid neighbor found.
        """
        table = self.tables.get(obj_name)
        if table is None:
            return None

        x_grid = table["x_grid"]
        y_grid = table["y_grid"]
        waypoints = table["waypoints"]
        valid = table.get("valid", np.ones(waypoints.shape[:2], dtype=bool))

        xi = int(np.argmin(np.abs(x_grid - x)))
        yi = int(np.argmin(np.abs(y_grid - y)))

        if not valid[xi, yi]:
            found = False
            for radius in range(1, 5):
                best_dist = float("inf")
                best_xi, best_yi = xi, yi
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nxi, nyi = xi + dx, yi + dy
                        if 0 <= nxi < len(x_grid) and 0 <= nyi < len(y_grid):
                            if valid[nxi, nyi]:
                                dist = (x_grid[nxi] - x) ** 2 + (y_grid[nyi] - y) ** 2
                                if dist < best_dist:
                                    best_dist = dist
                                    best_xi, best_yi = nxi, nyi
                                    found = True
                if found:
                    xi, yi = best_xi, best_yi
                    break
            if not found:
                return None

        return {
            "pregrasp": waypoints[xi, yi, 0].astype(np.float32),
            "grasp": waypoints[xi, yi, 1].astype(np.float32),
            "lift": waypoints[xi, yi, 2].astype(np.float32),
        }
