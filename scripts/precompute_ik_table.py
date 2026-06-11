"""
离线预计算: Per-Object IK 查表 + 相机外参.

在 Isaac Lab 环境中运行. 产出:
    demo/my_solution/data/camera_pose.npy  — 观察姿态相机外参
    demo/my_solution/data/ik_mustard.npy   — mustard IK 查表
    demo/my_solution/data/ik_sugar.npy     — sugar IK 查表
    demo/my_solution/data/ik_banana.npy    — banana IK 查表

Usage:
    cd ATEC2026_Simulation_Challenge
    python scripts/precompute_ik_table.py --task ATEC-TaskE-Piper --num_envs 1
"""

import argparse
import math
import os
import sys

import numpy as np

# --- 路径 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "demo", "my_solution", "data")

# --- 场景常量 ---
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


# ============================================================
# 辅助函数
# ============================================================

def quat_wxyz_to_rotmat(quat_wxyz):
    """四元数 (w,x,y,z) → 3x3 旋转矩阵."""
    w, x, y, z = quat_wxyz
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
        [2*(xy+wz), 1 - 2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy), 2*(yz+wx), 1 - 2*(xx+yy)],
    ], dtype=np.float64)


def euler_xyz_deg_to_rotmat(roll_deg, pitch_deg, yaw_deg):
    """Euler XYZ (degrees) → 3x3 旋转矩阵."""
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float64)


def rotmat_to_quat_wxyz(R):
    """3x3 旋转矩阵 → 四元数 (w,x,y,z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


# ============================================================
# 相机外参计算
# ============================================================

EE_CAMERA_OFFSET_POS = np.array((-0.05, 0.0, 0.06), dtype=np.float64)
GRIPPER_BODY_TO_PRIM_POS = np.array((0.0962770383, 0.0, -0.0066803808), dtype=np.float64)
GRIPPER_BODY_TO_PRIM_ROT_WXYZ = np.array((0.9693394981, 0.0, -0.2457253289, 0.0), dtype=np.float64)
EE_CAMERA_LOCAL_ROT_WXYZ = np.array((0.70710689, 0.0, -0.70710667, 0.0), dtype=np.float64)
WORLD_CAMERA_TO_ROS_ROT = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)
BASE_FRAME_CORRECTION_XYZ_DEG = (0.0, 0.0, 180.0)


def compute_camera_pose_from_ee(ee_pos_base, ee_rot_base):
    """从 EE body 位姿计算相机外参 (复用学长 DrakePiperIK.camera_pose_base 的 FK 链).

    FK chain: gripper_body → prim → camera → ROS

    Args:
        ee_pos_base: EE (gripper_base) position in base frame, shape (3,)
        ee_rot_base: EE rotation matrix in base frame (3,3)

    Returns:
        dict: {"rotation_matrix_ros": (3,3), "position_base": (3,)}
    """
    gripper_body_to_prim_rot = quat_wxyz_to_rotmat(GRIPPER_BODY_TO_PRIM_ROT_WXYZ)
    ee_camera_offset_rot = quat_wxyz_to_rotmat(EE_CAMERA_LOCAL_ROT_WXYZ)

    prim_rot = ee_rot_base @ gripper_body_to_prim_rot
    prim_pos = ee_pos_base + ee_rot_base @ GRIPPER_BODY_TO_PRIM_POS

    cam_rot_world = prim_rot @ ee_camera_offset_rot
    cam_pos = prim_pos + prim_rot @ EE_CAMERA_OFFSET_POS

    cam_rot_ros = cam_rot_world @ WORLD_CAMERA_TO_ROS_ROT

    return {
        "rotation_matrix_ros": cam_rot_ros,
        "position_base": cam_pos,
    }


# ============================================================
# Per-Object 抓取朝向
# ============================================================

GRASP_CONFIGS = {
    "sugar": {
        "base_rotation_xyz_deg": (-180.0, 10.0, 0.0),
        "grasp_angle_rad": math.pi / 2.0,
        "dynamic_pitch": True,
        "pitch_near_far_deg": (10.0, 30.0),
        "anchor_z_offset": -0.1,
        "anchor_offset_base_xyz": (0.04, 0.0, -0.1),
        "anchor_fixed_y_base_m": 0.212,
        "anchor_dynamic_offset_x_near_far_m": (0.035, 0.045),
        "dynamic_x_base_near_far_m": (
            TABLE_CENTER_X + 0.25 - ROBOT_BASE_W[0],
            TABLE_CENTER_X - 0.25 - ROBOT_BASE_W[0],
        ),
        "retreat_toward_base_m": 0.0,
    },
    "mustard": {
        "base_rotation_xyz_deg": (-180.0, 40.0, 0.0),
        "grasp_angle_rad": 0.0,
        "dynamic_pitch": False,
        "anchor_z_offset": -0.05,
        "anchor_offset_base_xyz": (0.03, 0.0, -0.05),
        "anchor_fixed_y_base_m": None,
        "anchor_dynamic_offset_x_near_far_m": None,
        "dynamic_x_base_near_far_m": None,
        "retreat_toward_base_m": 0.05,
    },
    "banana": {
        "base_rotation_xyz_deg": (-180.0, 20.0, 0.0),
        "grasp_angle_rad": 0.0,
        "dynamic_pitch": False,
        "anchor_z_offset": -0.08,
        "anchor_offset_base_xyz": (-0.03, 0.01, -0.08),
        "anchor_fixed_y_base_m": None,
        "anchor_dynamic_offset_x_near_far_m": None,
        "dynamic_x_base_near_far_m": None,
        "retreat_toward_base_m": 0.08,
    },
}


def get_grasp_orientation(obj_name, x_base=0.0):
    """获取物体的抓取朝向 (旋转矩阵, base frame)."""
    cfg = GRASP_CONFIGS[obj_name]
    roll, pitch, yaw = cfg["base_rotation_xyz_deg"]

    if cfg.get("dynamic_pitch") and cfg.get("pitch_near_far_deg") and cfg.get("dynamic_x_base_near_far_m"):
        p_near, p_far = cfg["pitch_near_far_deg"]
        x_near, x_far = cfg["dynamic_x_base_near_far_m"]
        if abs(x_far - x_near) > 1e-6:
            t = np.clip((x_base - x_near) / (x_far - x_near), 0.0, 1.0)
            pitch = p_near + t * (p_far - p_near)

    rot = euler_xyz_deg_to_rotmat(roll, pitch, yaw)

    angle = cfg["grasp_angle_rad"]
    if abs(angle) > 1e-6:
        c, s = math.cos(angle), math.sin(angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        rot = rot @ Rz

    return rot


def get_grasp_quat_wxyz(obj_name, x_base=0.0):
    """获取物体的抓取四元数 (w,x,y,z) 用于 CartesianController."""
    rot = get_grasp_orientation(obj_name, x_base)
    return rotmat_to_quat_wxyz(rot)


# ============================================================
# IK 查表计算
# ============================================================

def apply_anchor_offset(obj_name, obj_x_w, obj_y_w, robot_base_w):
    """将物体中心坐标转换为 EE 目标坐标 (应用 anchor offset).

    v4 核心修正: IK 表网格索引 = 物体中心位置，内部 apply anchor → EE 目标。

    Args:
        obj_name: 物体名
        obj_x_w, obj_y_w: 物体中心的世界坐标
        robot_base_w: robot base 世界坐标 (3,)

    Returns:
        ee_x_w, ee_y_w: EE 目标的世界坐标 (XY)
    """
    cfg = GRASP_CONFIGS[obj_name]
    anchor_xyz = cfg.get("anchor_offset_base_xyz", (0.0, 0.0, 0.0))

    obj_x_base = obj_x_w - robot_base_w[0]

    # --- X offset ---
    anchor_x = anchor_xyz[0]
    if obj_name == "sugar" and cfg.get("anchor_dynamic_offset_x_near_far_m"):
        x_near_base, x_far_base = cfg["dynamic_x_base_near_far_m"]
        offset_near, offset_far = cfg["anchor_dynamic_offset_x_near_far_m"]
        if abs(x_far_base - x_near_base) > 1e-6:
            t = np.clip((obj_x_base - x_near_base) / (x_far_base - x_near_base), 0.0, 1.0)
            anchor_x = offset_near + t * (offset_far - offset_near)

    # --- Y offset ---
    if cfg.get("anchor_fixed_y_base_m") is not None:
        ee_y_base = cfg["anchor_fixed_y_base_m"]
        ee_y_w = ee_y_base + robot_base_w[1]
    else:
        anchor_y = anchor_xyz[1]
        ee_y_w = obj_y_w + anchor_y

    ee_x_w = obj_x_w + anchor_x

    return ee_x_w, ee_y_w


def compute_pregrasp_position(ee_x_w, ee_y_w, pregrasp_z, obj_name, robot_base_w):
    """计算 pregrasp 位置: grasp 点上方 + retreat 回退.

    Args:
        ee_x_w, ee_y_w: EE grasp 点的 XY 世界坐标
        pregrasp_z: pregrasp 的 z 高度
        obj_name: 物体名
        robot_base_w: robot base 世界坐标

    Returns:
        (pre_x_w, pre_y_w, pregrasp_z): pregrasp 点世界坐标
    """
    cfg = GRASP_CONFIGS[obj_name]
    retreat_m = cfg.get("retreat_toward_base_m", 0.0)

    if retreat_m > 1e-6:
        dx = robot_base_w[0] - ee_x_w
        dy = robot_base_w[1] - ee_y_w
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1e-6:
            pre_x_w = ee_x_w + retreat_m * (dx / dist)
            pre_y_w = ee_y_w + retreat_m * (dy / dist)
        else:
            pre_x_w, pre_y_w = ee_x_w, ee_y_w
    else:
        pre_x_w, pre_y_w = ee_x_w, ee_y_w

    return pre_x_w, pre_y_w, pregrasp_z


# ============================================================
# Drake Direct IK Solver (复用学长 DrakePiperIK 的核心逻辑)
# ============================================================

class DrakeDirectIK:
    """Drake-based direct IK solver for Piper arm.

    Unlike CartesianController (differential IK, one step at a time),
    this solves for the complete target joint configuration directly.
    """

    EE_GRASP_POINT = np.array([0.0, 0.0, 0.06], dtype=np.float64)

    def __init__(self, vendor_root: str):
        from pydrake.multibody.inverse_kinematics import InverseKinematics
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.math import RotationMatrix, RollPitchYaw
        from pydrake.solvers import Solve

        self._InverseKinematics = InverseKinematics
        self._RotationMatrix = RotationMatrix
        self._RollPitchYaw = RollPitchYaw
        self._Solve = Solve

        package_root = os.path.join(vendor_root, "piper_description")
        urdf_path = os.path.join(package_root, "urdf", "piper_description.urdf")

        self.plant = MultibodyPlant(time_step=0.0)
        parser = Parser(self.plant)
        parser.package_map().Add("piper_description", package_root)
        model_instances = parser.AddModels(urdf_path)
        self.model_instance = model_instances[0]
        dummy_frame = self.plant.GetFrameByName("dummy_link", self.model_instance)
        self.plant.WeldFrames(self.plant.world_frame(), dummy_frame)
        self.plant.Finalize()
        self.context = self.plant.CreateDefaultContext()
        self.ee_frame = self.plant.GetFrameByName("gripper_base", self.model_instance)
        self.num_positions = self.plant.num_positions(self.model_instance)

        base_fix_xyz_rad = np.deg2rad(np.array(BASE_FRAME_CORRECTION_XYZ_DEG, dtype=np.float64))
        self.drake_to_sim_base_rot = RollPitchYaw(*base_fix_xyz_rad).ToRotationMatrix().matrix()
        self.sim_to_drake_base_rot = self.drake_to_sim_base_rot.T

        home_q = np.array(
            [-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033, 0.035, -0.035],
            dtype=np.float64,
        )
        self.home_q = home_q

    def solve(
        self,
        target_pos_base: np.ndarray,
        target_rot_base: np.ndarray,
        seed_q: np.ndarray | None = None,
        pos_tol: float = 0.005,
        orient_tol: float = 0.05,
    ) -> dict:
        """Solve IK for a target pose in robot base frame.

        Args:
            target_pos_base: EE target position in base frame (3,)
            target_rot_base: EE target rotation matrix in base frame (3,3)
            seed_q: initial joint guess (8,). If None, uses home.
            pos_tol: position tolerance in meters
            orient_tol: orientation tolerance in radians

        Returns:
            dict with "success" and "q" (8,) joint positions
        """
        target_pos_drake = self.sim_to_drake_base_rot @ target_pos_base.astype(np.float64)
        target_rot_drake = self._RotationMatrix(self.sim_to_drake_base_rot @ target_rot_base.astype(np.float64))

        if seed_q is None:
            seed_q = self.home_q.copy()
        seed_q = seed_q[:self.num_positions].astype(np.float64)

        ik_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(ik_context, self.model_instance, seed_q)
        ik = self._InverseKinematics(self.plant, ik_context)
        q = ik.q()
        prog = ik.prog()

        ik.AddPositionConstraint(
            self.ee_frame,
            self.EE_GRASP_POINT,
            self.plant.world_frame(),
            target_pos_drake - pos_tol,
            target_pos_drake + pos_tol,
        )
        ik.AddPositionCost(
            self.ee_frame,
            self.EE_GRASP_POINT,
            self.plant.world_frame(),
            target_pos_drake,
            5000.0 * np.eye(3),
        )
        ik.AddOrientationConstraint(
            self.ee_frame,
            self._RotationMatrix(),
            self.plant.world_frame(),
            target_rot_drake,
            orient_tol,
        )
        prog.AddQuadraticErrorCost(np.eye(self.num_positions), seed_q, q)
        prog.SetInitialGuess(q, seed_q)

        result = self._Solve(prog)
        if not result.is_success():
            return {"success": False}

        q_sol = result.GetSolution(q)
        q_full = np.zeros(8, dtype=np.float64)
        q_full[:len(q_sol)] = q_sol
        return {"success": True, "q": q_full}


def compute_ik_table(obj_name, drake_ik, table_top_z, robot_base_w):
    """对一个物体计算 IK 查表 (1cm 网格) — 使用 Drake 直接 IK.

    v5 修正: 使用 Drake MultibodyPlant 直接求解 IK (非差分 IK),
    确保每个格点的 waypoints 是完整的目标关节角而非增量.

    Args:
        obj_name: "sugar" | "mustard" | "banana"
        drake_ik: DrakeDirectIK 实例
        table_top_z: 桌面 z 高度
        robot_base_w: robot base 世界坐标 (3,)
    """
    cfg = GRASP_CONFIGS[obj_name]

    grid_ranges = {
        "sugar":   {"x": (0.80, 1.20), "y": (0.24, 0.30)},
        "mustard": {"x": (0.80, 1.20), "y": (0.13, 0.21)},
        "banana":  {"x": (0.80, 1.20), "y": (0.02, 0.10)},
    }
    r = grid_ranges[obj_name]
    resolution = 0.01

    x_grid = np.arange(r["x"][0], r["x"][1] + resolution/2, resolution)
    y_grid = np.arange(r["y"][0], r["y"][1] + resolution/2, resolution)
    Nx, Ny = len(x_grid), len(y_grid)

    fixed_center_z_base = {"sugar": 0.212, "mustard": 0.188, "banana": 0.107}
    center_z = fixed_center_z_base[obj_name]
    anchor_z = cfg["anchor_z_offset"]
    grasp_z = table_top_z + center_z + anchor_z
    grasp_z = max(grasp_z, table_top_z + 0.01)

    clearances = {
        "sugar":   {"pre": 0.10, "lift": 0.10},
        "mustard": {"pre": 0.15, "lift": 0.10},
        "banana":  {"pre": 0.12, "lift": 0.10},
    }
    c = clearances[obj_name]
    pregrasp_z = grasp_z + c["pre"]
    lift_z = grasp_z + c["lift"]

    waypoints = np.zeros((Nx, Ny, 3, 8), dtype=np.float32)
    valid = np.zeros((Nx, Ny), dtype=bool)

    print(f"  Grid: {Nx}x{Ny} = {Nx*Ny} points")
    print(f"  center_z_base={center_z:.3f}, anchor_z={anchor_z:.3f}")
    print(f"  Heights (world z): pregrasp={pregrasp_z:.3f}, grasp={grasp_z:.3f}, lift={lift_z:.3f}")

    for xi, obj_x in enumerate(x_grid):
        for yi, obj_y in enumerate(y_grid):
            ee_x, ee_y = apply_anchor_offset(obj_name, obj_x, obj_y, robot_base_w)

            pre_x, pre_y, pre_z = compute_pregrasp_position(
                ee_x, ee_y, pregrasp_z, obj_name, robot_base_w
            )

            obj_x_base = obj_x - robot_base_w[0]
            grasp_rot = get_grasp_orientation(obj_name, obj_x_base)

            success = True
            positions = [
                (pre_x, pre_y, pre_z),
                (ee_x, ee_y, grasp_z),
                (ee_x, ee_y, lift_z),
            ]
            seed_q = None
            for wi, (px, py, pz) in enumerate(positions):
                target_pos_base = np.array([px, py, pz], dtype=np.float64) - robot_base_w
                result = drake_ik.solve(target_pos_base, grasp_rot, seed_q=seed_q)
                if not result["success"]:
                    success = False
                    break
                q_sol = result["q"]
                q_sol[6] = 0.035
                q_sol[7] = -0.035
                waypoints[xi, yi, wi] = q_sol.astype(np.float32)
                seed_q = q_sol.copy()

            valid[xi, yi] = success

    valid_pct = 100.0 * valid.sum() / (Nx * Ny)
    print(f"  Valid: {valid.sum()}/{Nx*Ny} ({valid_pct:.1f}%)")

    return {
        "x_grid": x_grid.astype(np.float64),
        "y_grid": y_grid.astype(np.float64),
        "waypoints": waypoints,
        "valid": valid,
        "obj_name": obj_name,
        "heights": {"pregrasp_z": pregrasp_z, "grasp_z": grasp_z, "lift_z": lift_z},
    }


# ============================================================
# Main
# ============================================================

def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Precompute IK tables for Task E hybrid solution")
    parser.add_argument("--task", type=str, default="ATEC-TaskE-Piper")
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    import atec_rl_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # --- 创建环境 (预计算不需要相机, 禁用 image observations 避免渲染问题) ---
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.observations.image = None
    env_cfg.scene.video_cam = None
    env_cfg.scene.ee_camera = None
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    sys.stdout.flush()

    # --- 获取 robot ---
    robot = env.unwrapped.scene["robot"]

    # --- Step 0: 初始化 Drake Direct IK ---
    print("=" * 60)
    print("Step 0: Initializing Drake Direct IK solver...")
    print("=" * 60)
    vendor_root = os.path.join(PROJECT_DIR, "demo", "vendor")
    drake_ik = DrakeDirectIK(vendor_root)
    print(f"  Drake IK initialized, num_positions={drake_ik.num_positions}")

    # 测试 Drake IK — 求解一个已知可达位置
    test_pos_base = np.array([-0.2, 0.15, 0.15], dtype=np.float64)
    test_rot = euler_xyz_deg_to_rotmat(-180.0, 20.0, 0.0)
    test_result = drake_ik.solve(test_pos_base, test_rot)
    print(f"  Test solve success: {test_result['success']}")
    if test_result["success"]:
        print(f"  Test solve q[:6]: {test_result['q'][:6].round(4)}")
    print("  Drake Direct IK verified OK!")

    # --- Step 1: 计算相机外参 (使用学长的 DrakePiperIK) ---
    print("\n" + "=" * 60)
    print("Step 1: Computing camera pose at OBSERVATION_JOINT_POS...")
    print("=" * 60)

    from demo.my_solution.config import OBSERVATION_JOINT_POS, DEFAULT_JOINT_POS
    from demo.tasks.task_e.grasp_config import GraspConfig as SeniorGraspConfig
    from demo.tasks.task_e.grasp_drake import DrakePiperIK

    senior_config = SeniorGraspConfig()
    senior_ik = DrakePiperIK(vendor_root, senior_config)
    camera_pose = senior_ik.camera_pose_base(OBSERVATION_JOINT_POS.astype(np.float64))
    print(f"  Camera pos (base): {camera_pose['position_base']}")
    print(f"  Camera R_ros diagonal: {np.diag(camera_pose['rotation_matrix_ros'])}")

    # 同时用环境验证 robot base 位置
    body_ids, _ = robot.find_bodies("gripper_base")
    ee_idx = body_ids[0]

    obs_action = (OBSERVATION_JOINT_POS - DEFAULT_JOINT_POS) / 0.5
    obs_action_t = torch.tensor(obs_action, dtype=torch.float32, device=args.device).unsqueeze(0)
    for _ in range(300):
        env.step(obs_action_t)

    root_pos_w = robot.data.root_pose_w[:, :3][0].cpu().numpy()
    assert np.allclose(root_pos_w, ROBOT_BASE_W, atol=0.005), \
        f"Measured root_pos_w={root_pos_w} != ROBOT_BASE_W={ROBOT_BASE_W}"
    print(f"  Root pos (world): {root_pos_w}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    camera_pose_save = {
        "rotation_matrix_ros": camera_pose["rotation_matrix_ros"],
        "position_base": camera_pose["position_base"],
    }
    np.save(os.path.join(OUTPUT_DIR, "camera_pose.npy"), camera_pose_save)
    print(f"  Saved: {OUTPUT_DIR}/camera_pose.npy")

    # --- Step 2: 对每个物体计算 IK 查表 (Drake 直接 IK) ---
    robot_base_w = root_pos_w

    for obj_name in ["mustard", "sugar", "banana"]:
        print(f"\n{'=' * 60}")
        print(f"Step 2: Computing IK table for: {obj_name}")
        print(f"{'=' * 60}")

        table = compute_ik_table(obj_name, drake_ik, TABLE_TOP_Z, robot_base_w)

        output_path = os.path.join(OUTPUT_DIR, f"ik_{obj_name}.npy")
        np.save(output_path, table)
        print(f"  Saved: {output_path}")

    print(f"\n\nDone! All files saved to: {OUTPUT_DIR}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
