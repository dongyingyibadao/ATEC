"""感知模块: 深度反投影 + Y条带 + 连通域定位.

坐标变换完全复用学长 grasp_drake.py 的 _depth_to_base() 方式:
    base_points = cam_points_cv @ camera_pose["rotation_matrix_ros"].T + camera_pose["position_base"]
    world_points = base_points + ROBOT_BASE_W

注: cam_points_cv 是 pinhole 反投影的 OpenCV 坐标系点 (z-forward, x-right, y-down),
后续乘 R_ros.T 时一步完成了 OpenCV→ROS→base 的变换。

学长的 camera_pose_base() FK 链已经 bake 了 BASE_FRAME_CORRECTION (180° Z),
所以 base→world 直接加 ROBOT_BASE_W 即可, 无需额外旋转。
"""
import numpy as np
from scipy import ndimage

from .config import (
    FX, FY, CX, CY,
    ROBOT_BASE_W,
    DEPTH_MIN_M, DEPTH_MAX_M,
    OBJECT_WORLD_X_MIN, OBJECT_WORLD_X_MAX,
    OBJECT_WORLD_Z_MIN, OBJECT_WORLD_Z_MAX,
    OBJECT_STRIPS, STRIP_Y_PAD,
    MIN_COMPONENT_PIXELS,
)


class Perception:
    """从腕部深度图中定位三个物体的世界坐标."""

    def __init__(self, camera_pose: dict):
        """
        Args:
            camera_pose: 离线预计算的相机位姿, 包含:
                - "rotation_matrix_ros": np.ndarray (3,3)
                - "position_base": np.ndarray (3,)
        """
        self.cam_rot_ros = camera_pose["rotation_matrix_ros"]
        self.cam_pos_base = camera_pose["position_base"]

    def depth_to_base_and_world(self, depth: np.ndarray) -> np.ndarray:
        """将深度图反投影为世界坐标点云.

        Args:
            depth: shape (H, W), float32, 单位米

        Returns:
            world_points: shape (H, W, 3), 世界坐标
        """
        H, W = depth.shape
        v_coords, u_coords = np.indices(depth.shape, dtype=np.float32)

        z_cam = depth
        x_cam = (u_coords - CX) * depth / FX
        y_cam = (v_coords - CY) * depth / FY
        cam_points_cv = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float64)

        cam_flat = cam_points_cv.reshape(-1, 3)
        base_flat = cam_flat @ self.cam_rot_ros.T + self.cam_pos_base
        base_points = base_flat.reshape(H, W, 3)

        world_points = base_points + ROBOT_BASE_W
        return world_points

    def locate_all_objects(self, depth: np.ndarray) -> dict[str, np.ndarray | None]:
        """一次性定位所有三个物体 (batch 模式用).

        Args:
            depth: shape (H, W), float32, 单位米

        Returns:
            dict: obj_name → world_center (np.array shape (3,)) 或 None
        """
        world_points = self.depth_to_base_and_world(depth)

        valid = np.isfinite(depth) & (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M)
        finite = np.all(np.isfinite(world_points), axis=-1)
        x_ok = (
            (world_points[..., 0] >= OBJECT_WORLD_X_MIN)
            & (world_points[..., 0] <= OBJECT_WORLD_X_MAX)
        )
        z_ok = (
            (world_points[..., 2] >= OBJECT_WORLD_Z_MIN)
            & (world_points[..., 2] <= OBJECT_WORLD_Z_MAX)
        )
        common_mask = valid & finite & x_ok & z_ok

        results = {}
        for obj_name, strip_info in OBJECT_STRIPS.items():
            y_min, y_max = strip_info["y_range"]
            y_ok = (
                (world_points[..., 1] >= (y_min - STRIP_Y_PAD))
                & (world_points[..., 1] <= (y_max + STRIP_Y_PAD))
            )
            strip_mask = common_mask & y_ok

            labeled, num_labels = ndimage.label(
                strip_mask.astype(np.uint8),
                structure=np.ones((3, 3), dtype=np.uint8),
            )

            best_center = None
            best_size = 0
            for label_idx in range(1, num_labels + 1):
                component_mask = labeled == label_idx
                size = int(np.count_nonzero(component_mask))
                if size < MIN_COMPONENT_PIXELS:
                    continue
                if size > best_size:
                    best_size = size
                    pts = world_points[component_mask]
                    best_center = np.median(pts, axis=0).astype(np.float64)

            results[obj_name] = best_center

        return results
