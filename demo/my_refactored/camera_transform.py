"""相机内参计算、深度图反投影到base/world坐标系。"""
import numpy as np

from .config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CAMERA_FOCAL_MM,
    CAMERA_SENSOR_WIDTH_MM,
    ROBOT_BASE_W,
    SystemParams,
)


# 焦距像素值
FX = IMAGE_WIDTH * CAMERA_FOCAL_MM / CAMERA_SENSOR_WIDTH_MM
FY = FX
CX = (IMAGE_WIDTH - 1.0) / 2.0
CY = (IMAGE_HEIGHT - 1.0) / 2.0


def project_depth_to_base(depth: np.ndarray, camera_pose: dict, params: SystemParams) -> dict:
    """将深度图反投影到 base 和 world 坐标系。

    Args:
        depth: (H, W) 深度图，单位米
        camera_pose: arm_solver.compute_camera_pose() 的返回值
        params: 系统参数

    Returns:
        包含 points_base, points_world, points_cam_ros, points_cam_frame 的字典
    """
    v_coords, u_coords = np.indices(depth.shape, dtype=np.float32)
    z_cam = depth
    x_cam = params.camera_u_sign * (u_coords - CX) * depth / FX
    y_cam = params.camera_v_sign * (v_coords - CY) * depth / FY
    cam_points_ros = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float64)
    base_points = cam_points_ros @ camera_pose["rotation_matrix_ros"].T + camera_pose["position_base"]
    return {
        "points_base": base_points,
        "points_world": base_points + ROBOT_BASE_W,
        "points_cam_ros": cam_points_ros,
        "points_cam_frame": cam_points_ros,
    }


def build_workspace_mask(depth: np.ndarray, params: SystemParams) -> np.ndarray:
    """构建有效深度的工作空间掩码。"""
    valid = np.isfinite(depth) & (depth > params.depth_min_m) & (depth < params.depth_max_m)
    bottom_mask = _build_bottom_exclusion_mask(depth.shape, params)
    return valid & (~bottom_mask)


def _build_bottom_exclusion_mask(shape: tuple[int, int], params: SystemParams) -> np.ndarray:
    """生成图像底部排除区域的掩码。"""
    h, w = shape
    y_coords = np.indices((h, w))[0]
    return y_coords >= int((1.0 - params.bottom_image_exclusion_ratio) * h)


def project_base_to_pixel(point_base: np.ndarray | None, camera_pose: dict) -> list[int] | None:
    """将base坐标系中的3D点投影回像素坐标。"""
    if point_base is None:
        return None
    point = np.asarray(point_base, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    cam_pos = np.asarray(camera_pose["position_base"], dtype=np.float64)
    rot_ros = np.asarray(camera_pose["rotation_matrix_ros"], dtype=np.float64)
    point_cam_ros = rot_ros.T @ (point - cam_pos)
    z = float(point_cam_ros[2])
    if z <= 1e-6 or not np.isfinite(z):
        return None
    u = FX * float(point_cam_ros[0]) / z + CX
    v = FY * float(point_cam_ros[1]) / z + CY
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return [int(np.clip(round(u), 0, IMAGE_WIDTH - 1)), int(np.clip(round(v), 0, IMAGE_HEIGHT - 1))]
