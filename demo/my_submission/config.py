"""场景参数、机器人预设、物体抓取模板配置。"""
import math
import os
from dataclasses import dataclass

import numpy as np


# === 路径定位 ===
_THIS_DIR = os.path.dirname(__file__)
_DEMO_DIR = os.path.dirname(_THIS_DIR)
_ASSETS_DIR = os.path.join(_DEMO_DIR, "assets")

# === 场景几何（Isaac 世界坐标系）===
TABLE_CENTER_X = 1.00
TABLE_CENTER_Y = 0.00
TABLE_CENTER_Z = 0.00
TABLE_SCALE = 0.01
_TABLE_DIMS_AT_0P008 = (0.6468062441005529, 0.9084968693231588, 0.6613141183247961)
TABLE_DIMS = tuple(dim * (TABLE_SCALE / 0.008) for dim in _TABLE_DIMS_AT_0P008)
TABLE_HALF_X = TABLE_DIMS[0] * 0.5
TABLE_TOP_Z = TABLE_CENTER_Z + TABLE_DIMS[2]
ROBOT_BASE_W = np.array((TABLE_CENTER_X + TABLE_HALF_X, TABLE_CENTER_Y, TABLE_TOP_Z), dtype=np.float64)

# === 腕部相机参数 ===
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
CAMERA_FOCAL_MM = 15.0
CAMERA_SENSOR_WIDTH_MM = 20.955

# === 相机安装偏移（相对于仿真夹爪 prim/body）===
CAMERA_OFFSET_POSITION = np.array((-0.05, 0.0, 0.06), dtype=np.float64)
GRIPPER_BODY_TO_PRIM_POS = np.array((0.0962770383, 0.0, -0.0066803808), dtype=np.float64)
GRIPPER_BODY_TO_PRIM_QUAT_WXYZ = np.array((0.9693394981, 0.0, -0.2457253289, 0.0), dtype=np.float64)
CAMERA_LOCAL_QUAT_WXYZ = np.array((0.70710689, 0.0, -0.70710667, 0.0), dtype=np.float64)
WORLD_TO_ROS_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

# === 机器人关节预设 ===
DEFAULT_JOINT_POS = np.array([0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035], dtype=np.float32)
HOME_JOINT_POS = np.array(
    [-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033, 0.035000, -0.035000],
    dtype=np.float32,
)
OBSERVE_JOINT_POS = np.array(
    [-0.3491, 49.3 / 180 * np.pi, -60 / 180 * np.pi, 0.0, 70 / 180 * np.pi, 0.0, 0.035, -0.035],
    dtype=np.float32,
)
PLACE_JOINT_POS = np.array(
    [0.6981, 1.38, -1.3, 0.0, 0.0, 0.0, -0.015, 0.015],
    dtype=np.float32,
)
GRIPPER_OPEN = np.array([0.035, -0.035], dtype=np.float32)
GRIPPER_CLOSE = np.array([-0.015, 0.015], dtype=np.float32)
ACTION_SCALE = 0.5

# === 坐标系修正 ===
FIXED_GRASP_EULER_DEG = (-177.876926, 20.256916, 167.942924)
BASE_CORRECTION_EULER_DEG = (0.0, 0.0, 180.0)

# === 物体检测参数（base坐标系）===
DETECTION_Z_MIN_BASE = -0.049048
DETECTION_Z_MAX_BASE = 0.302389


def _quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """四元数(wxyz顺序)转旋转矩阵。"""
    w, x, y, z = quat_wxyz
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """旋转矩阵转四元数(wxyz顺序)。"""
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


@dataclass
class SystemParams:
    """系统运行参数集合。"""
    # 控制器时序
    startup_zero_steps: int = 0
    observation_hold_steps: int = 0
    post_lift_settle_steps: int = 0
    failed_plan_cooldown_steps: int = 0
    probe_move_steps: int = 10
    probe_hold_steps: int = 0

    # 调试开关
    observation_only: bool = False
    debug_enabled: bool = False
    debug_dirname: str = "debug_mesh_pose"
    aggressive_cycle_mode: bool = True

    # 奖励阈值
    task_e_object_order: tuple[str, ...] = ("mustard", "sugar", "banana")
    reward_pick_success_delta: float = 2.5
    reward_place_success_delta: float = 2.5

    # 深度过滤
    depth_min_m: float = 0.097613
    depth_max_m: float = 2.456496
    candidate_keep_best: int = 6

    # 点云匹配参数
    mesh_model_max_points: int = 1948
    mesh_scene_max_points: int = 4027
    mesh_icp_threshold_m: float = 0.044577
    mesh_mask_distance_m: float = 0.020884

    # 世界坐标物体条带过滤
    object_world_x_min_m: float = TABLE_CENTER_X - 0.25
    object_world_x_max_m: float = TABLE_CENTER_X + 0.25
    object_world_strip_pad_m: float = 0.015
    object_world_z_min_m: float = TABLE_TOP_Z + 0.005
    object_world_z_max_m: float = TABLE_TOP_Z + 0.35

    # 夹持检测
    grasp_hold_gap_threshold: float = 0.001517

    # IK 与运动插值
    ee_grasp_point_in_frame: tuple[float, float, float] = (0.0, 0.0, 0.061317)
    ik_position_tolerance_m: float = 0.004801
    ik_orientation_tolerance_rad: float = 0.047535
    lift_num_waypoints: int = 1
    grasp_angle_scale: float = 0.972373
    approach_steps: int = 8
    descend_steps: int = 26
    close_steps: int = 6
    lift_steps: int = 6
    place_steps: int = 11
    release_steps: int = 11
    return_steps: int = 18
    max_joint_delta_per_step: float = 0.049129
    observation_ready_joint_tol: float = 0.041513
    sugar_pregrasp_fallback_step_m: float = 0.005077
    sugar_pregrasp_fallback_max_steps: int = 18

    # 探测行为
    probe_offset_m: float = 0.051288

    # 相机反投影
    camera_u_sign: float = 1.0
    camera_v_sign: float = 1.0
    bottom_image_exclusion_ratio: float = 0.0


@dataclass(frozen=True)
class ApproachSpec:
    """单物体的预抓取/抬升运动参数。"""
    pregrasp_clearance_m: float
    retreat_toward_base_m: float
    lift_clearance_m: float


@dataclass(frozen=True)
class AnchorSpec:
    """单物体的抓取锚点偏移参数。"""
    offset_base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_y_base_m: float | None = None
    dynamic_offset_x_near_far_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class GraspTemplate:
    """物体抓取模板——将位姿估计结果转换为抓取姿态所需的全部参数。"""
    name: str
    y_min_base: float
    grasp_angle_rad: float
    base_rotation_xyz_deg: tuple[float, float, float] | None
    approach: ApproachSpec
    anchor: AnchorSpec
    dynamic_x_base_near_far_m: tuple[float, float] | None = None
    dynamic_rotation_y_deg_near_far: tuple[float, float] | None = None


@dataclass(frozen=True)
class DetectionSpec:
    """物体检测规格——ICP定位所需的参数。"""
    name: str
    world_strip_y_range: tuple[float, float]
    model_points_path: str
    seed_rotation_wxyz: tuple[float, float, float, float]
    fixed_center_xy_offset_base_m: tuple[float, float]
    fixed_center_z_base_m: float


# === 各物体的抓取模板 ===
GRASP_TEMPLATES = {
    "sugar": GraspTemplate(
        name="sugar",
        y_min_base=0.230995,
        grasp_angle_rad=math.pi / 2.016383,
        base_rotation_xyz_deg=(-181.262701, 9.710919, 0.0),
        approach=ApproachSpec(
            pregrasp_clearance_m=0.100407,
            retreat_toward_base_m=0.0,
            lift_clearance_m=0.098388,
        ),
        anchor=AnchorSpec(
            offset_base_xyz=(0.039915, 0.0, -0.104163),
            fixed_y_base_m=0.215706,
            dynamic_offset_x_near_far_m=(0.035213, 0.045235),
        ),
        dynamic_x_base_near_far_m=(
            TABLE_CENTER_X + 0.260765 - ROBOT_BASE_W[0],
            TABLE_CENTER_X - 0.244884 - ROBOT_BASE_W[0],
        ),
        dynamic_rotation_y_deg_near_far=(10.426876, 28.676171),
    ),
    "mustard": GraspTemplate(
        name="mustard",
        y_min_base=0.106508,
        grasp_angle_rad=0.0,
        base_rotation_xyz_deg=(-185.019993, 39.386986, 0.0),
        approach=ApproachSpec(
            pregrasp_clearance_m=0.146105,
            retreat_toward_base_m=0.048556,
            lift_clearance_m=0.095986,
        ),
        anchor=AnchorSpec(
            offset_base_xyz=(0.030304, 0.0, -0.048708),
        ),
    ),
    "banana": GraspTemplate(
        name="banana",
        y_min_base=-983848173.026452,
        grasp_angle_rad=0.0,
        base_rotation_xyz_deg=(-174.024878, 20.786707, 0.0),
        approach=ApproachSpec(
            pregrasp_clearance_m=0.117577,
            retreat_toward_base_m=0.076133,
            lift_clearance_m=0.098805,
        ),
        anchor=AnchorSpec(
            offset_base_xyz=(-0.028667, 0.01036, -0.079917),
        ),
    ),
}

TEMPLATE_FALLBACK_ORDER = ("sugar", "mustard", "banana")

# === 各物体的检测规格 ===
DETECTION_SPECS = {
    "sugar": DetectionSpec(
        name="sugar",
        world_strip_y_range=(TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),
        model_points_path=os.path.join(_ASSETS_DIR, "template_pcl", "sg.npy"),
        seed_rotation_wxyz=(0.0, 0.707, 0.0, 0.707),
        fixed_center_xy_offset_base_m=(0.0, 0.0),
        fixed_center_z_base_m=0.219194,
    ),
    "mustard": DetectionSpec(
        name="mustard",
        world_strip_y_range=(TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),
        model_points_path=os.path.join(_ASSETS_DIR, "template_pcl", "mt.npy"),
        seed_rotation_wxyz=(0.0, 0.0, -0.707, 0.707),
        fixed_center_xy_offset_base_m=(0.0, 0.0),
        fixed_center_z_base_m=0.190433,
    ),
    "banana": DetectionSpec(
        name="banana",
        world_strip_y_range=(TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09),
        model_points_path=os.path.join(_ASSETS_DIR, "template_pcl", "bn.npy"),
        seed_rotation_wxyz=(0.0, 0.0, -0.707, 0.707),
        fixed_center_xy_offset_base_m=(0.049421, -0.024268),
        fixed_center_z_base_m=0.108416,
    ),
}

DETECTION_EXTRACTION_ORDER = ("banana", "mustard", "sugar")
MODEL_SEED_ROTATIONS = {
    name: _quat_wxyz_to_rotation_matrix(np.asarray(spec.seed_rotation_wxyz, dtype=np.float64))
    for name, spec in DETECTION_SPECS.items()
}
