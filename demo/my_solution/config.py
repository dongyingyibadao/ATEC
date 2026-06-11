"""混合方案配置: 场景几何、相机内参、关节预设、per-object 抓取参数."""
import math
import os
import numpy as np

# --- 路径 ---
MY_SOLUTION_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MY_SOLUTION_DIR, "data")

# --- 场景几何 (世界坐标系) ---
TABLE_CENTER_X = 1.00
TABLE_CENTER_Y = 0.00
TABLE_CENTER_Z = 0.00
TABLE_SCALE = 0.01
TABLE_DIMS_AT_0P008 = (0.6468062441005529, 0.9084968693231588, 0.6613141183247961)
TABLE_DIMS = tuple(dim * (TABLE_SCALE / 0.008) for dim in TABLE_DIMS_AT_0P008)
TABLE_HALF_X = TABLE_DIMS[0] * 0.5
TABLE_TOP_Z = TABLE_CENTER_Z + TABLE_DIMS[2]
ROBOT_BASE_W = np.array(
    (TABLE_CENTER_X + TABLE_HALF_X, TABLE_CENTER_Y, TABLE_TOP_Z), dtype=np.float64
)

# --- 相机内参 ---
CAM_H = 480
CAM_W = 640
EE_CAMERA_FOCAL_MM = 15.0
EE_CAMERA_SENSOR_WIDTH_MM = 20.955
FX = CAM_W * EE_CAMERA_FOCAL_MM / EE_CAMERA_SENSOR_WIDTH_MM  # ~458.1
FY = FX
CX = (CAM_W - 1.0) / 2.0
CY = (CAM_H - 1.0) / 2.0

# --- 相机安装 (用于离线 FK 链计算) ---
EE_CAMERA_OFFSET_POS = np.array((-0.05, 0.0, 0.06), dtype=np.float64)
GRIPPER_BODY_TO_PRIM_POS = np.array((0.0962770383, 0.0, -0.0066803808), dtype=np.float64)
GRIPPER_BODY_TO_PRIM_ROT_WXYZ = np.array(
    (0.9693394981, 0.0, -0.2457253289, 0.0), dtype=np.float64
)
EE_CAMERA_LOCAL_ROT_WXYZ = np.array(
    (0.70710689, 0.0, -0.70710667, 0.0), dtype=np.float64
)
WORLD_CAMERA_TO_ROS_ROT = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# --- 关节预设 ---
DEFAULT_JOINT_POS = np.array(
    [0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035], dtype=np.float32
)
OBSERVATION_JOINT_POS = np.array(
    [-0.3491, 49.3 / 180 * np.pi, -60 / 180 * np.pi, 0.0, 70 / 180 * np.pi, 0.0, 0.035, -0.035],
    dtype=np.float32,
)
PLACE_JOINT_POS = np.array(
    [0.6981, 1.38, -1.3, 0.0, 0.0, 0.0, -0.015, 0.015], dtype=np.float32
)
GRIPPER_OPEN = np.array([0.035, -0.035], dtype=np.float32)
GRIPPER_CLOSE = np.array([-0.015, 0.015], dtype=np.float32)
ACTION_SCALE = 0.5

# --- 物体 Y 条带 (世界坐标) ---
OBJECT_STRIPS = {
    "sugar": {"y_range": (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29)},
    "mustard": {"y_range": (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20)},
    "banana": {"y_range": (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09)},
}

# --- 物体处理顺序 (同学长: mustard → sugar → banana) ---
OBJECT_ORDER = ("mustard", "sugar", "banana")

# --- 深度过滤范围 ---
DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 2.50
OBJECT_WORLD_X_MIN = TABLE_CENTER_X - 0.25
OBJECT_WORLD_X_MAX = TABLE_CENTER_X + 0.25
OBJECT_WORLD_Z_MIN = TABLE_TOP_Z + 0.005
OBJECT_WORLD_Z_MAX = TABLE_TOP_Z + 0.35
STRIP_Y_PAD = 0.015

# --- 连通域阈值 ---
MIN_COMPONENT_PIXELS = 120

# --- 运动参数 (与学长一致) ---
MAX_JOINT_DELTA_PER_STEP = 0.05
APPROACH_STEPS = 7
DESCEND_STEPS = 25
CLOSE_STEPS = 5
LIFT_STEPS = 5
PLACE_STEPS = 10
RELEASE_STEPS = 10
RETURN_STEPS = 18

# ============================================================
# Per-Object 抓取配置 (从学长 grasp_config.py 提取核心参数)
# ============================================================

OBJECT_GRASP_CONFIGS = {
    "sugar": {
        "grasp_angle_rad": math.pi / 2.0,
        "base_rotation_xyz_deg": (-180.0, 10.0, 0.0),
        "dynamic_rotation_y_deg_near_far": (10.0, 30.0),
        "dynamic_x_base_near_far_m": (
            TABLE_CENTER_X + 0.25 - ROBOT_BASE_W[0],
            TABLE_CENTER_X - 0.25 - ROBOT_BASE_W[0],
        ),
        "anchor_offset_base_xyz": (0.04, 0.0, -0.1),
        "anchor_fixed_y_base_m": 0.212,
        "anchor_dynamic_offset_x_near_far_m": (0.035, 0.045),
        "pregrasp_clearance_m": 0.1,
        "retreat_toward_base_m": 0.0,
        "lift_clearance_m": 0.10,
    },
    "mustard": {
        "grasp_angle_rad": 0.0,
        "base_rotation_xyz_deg": (-180.0, 40.0, 0.0),
        "dynamic_rotation_y_deg_near_far": None,
        "dynamic_x_base_near_far_m": None,
        "anchor_offset_base_xyz": (0.03, 0.0, -0.05),
        "anchor_fixed_y_base_m": None,
        "anchor_dynamic_offset_x_near_far_m": None,
        "pregrasp_clearance_m": 0.15,
        "retreat_toward_base_m": 0.05,
        "lift_clearance_m": 0.10,
    },
    "banana": {
        "grasp_angle_rad": 0.0,
        "base_rotation_xyz_deg": (-180.0, 20.0, 0.0),
        "dynamic_rotation_y_deg_near_far": None,
        "dynamic_x_base_near_far_m": None,
        "anchor_offset_base_xyz": (-0.03, 0.01, -0.08),
        "anchor_fixed_y_base_m": None,
        "anchor_dynamic_offset_x_near_far_m": None,
        "pregrasp_clearance_m": 0.12,
        "retreat_toward_base_m": 0.08,
        "lift_clearance_m": 0.10,
    },
}

# --- IK 查表网格配置 (1cm 密度, 最近邻) ---
IK_GRID_RESOLUTION_M = 0.01
IK_TABLE_CONFIGS = {
    "sugar": {
        "x_range": (TABLE_CENTER_X - 0.20, TABLE_CENTER_X + 0.20),
        "y_range": (TABLE_CENTER_Y + 0.24, TABLE_CENTER_Y + 0.30),
    },
    "mustard": {
        "x_range": (TABLE_CENTER_X - 0.20, TABLE_CENTER_X + 0.20),
        "y_range": (TABLE_CENTER_Y + 0.13, TABLE_CENTER_Y + 0.21),
    },
    "banana": {
        "x_range": (TABLE_CENTER_X - 0.20, TABLE_CENTER_X + 0.20),
        "y_range": (TABLE_CENTER_Y + 0.02, TABLE_CENTER_Y + 0.10),
    },
}
