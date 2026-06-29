"""Piper 机械臂纯 numpy FK/IK 求解器。

替代 pydrake MultibodyPlant，实现相同接口。
使用 URDF 中提取的关节参数进行正运动学，
使用 scipy L-BFGS-B 进行逆运动学。
"""
import numpy as np
from scipy.optimize import minimize as sp_minimize

from .config import (
    BASE_CORRECTION_EULER_DEG,
    CAMERA_LOCAL_QUAT_WXYZ,
    CAMERA_OFFSET_POSITION,
    FIXED_GRASP_EULER_DEG,
    GRIPPER_BODY_TO_PRIM_POS,
    GRIPPER_BODY_TO_PRIM_QUAT_WXYZ,
    GRIPPER_OPEN,
    HOME_JOINT_POS,
    OBSERVE_JOINT_POS,
    WORLD_TO_ROS_ROTATION,
    SystemParams,
    _quat_wxyz_to_rotation_matrix,
    _rotation_matrix_to_quat_wxyz,
)
from .rotation_utils import rpy_to_matrix


# URDF joint parameters (joint1..joint6)
# Each entry: (origin_xyz, origin_rpy, axis_direction)
_JOINT_PARAMS = [
    # joint1: base_link -> link1
    (np.array([0.0, 0.0, 0.123]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    # joint2: link1 -> link2
    (np.array([0.0, 0.0, 0.0]), np.array([1.5708, -0.1359, -3.1416]), np.array([0.0, 0.0, 1.0])),
    # joint3: link2 -> link3
    (np.array([0.28503, 0.0, 0.0]), np.array([0.0, 0.0, -1.7939]), np.array([0.0, 0.0, 1.0])),
    # joint4: link3 -> link4
    (np.array([-0.021984, -0.25075, 0.0]), np.array([1.5708, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    # joint5: link4 -> link5
    (np.array([0.0, 0.0, 0.0]), np.array([-1.5708, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    # joint6: link5 -> link6
    (np.array([8.8259e-05, -0.091, 0.0]), np.array([1.5708, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
]
# Fixed joint: link6 -> gripper_base (identity transform)
_EE_FIXED_TRANSFORM = np.eye(4, dtype=np.float64)

# Joint limits for joints 1-6
_JOINT_LOWER = np.array([-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944], dtype=np.float64)
_JOINT_UPPER = np.array([2.618, 3.14, 0.0, 1.745, 1.22, 2.0944], dtype=np.float64)


def _homogeneous(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _forward_kinematics_full(q: np.ndarray) -> np.ndarray:
    """Compute FK: 4x4 homogeneous transform from dummy_link to gripper_base."""
    T = np.eye(4, dtype=np.float64)
    for i, (origin_xyz, origin_rpy, axis) in enumerate(_JOINT_PARAMS):
        R_fixed = rpy_to_matrix(origin_rpy[0], origin_rpy[1], origin_rpy[2])
        T_fixed = _homogeneous(R_fixed, origin_xyz)
        T = T @ T_fixed
        R_joint = _rot_axis(axis, float(q[i]))
        T_joint = _homogeneous(R_joint, np.zeros(3))
        T = T @ T_joint
    T = T @ _EE_FIXED_TRANSFORM
    return T


class PiperArmIK:
    """Piper 机械臂逆运动学求解器 (纯 numpy 实现)。"""

    def __init__(self, params: SystemParams):
        self.params = params
        self.num_positions = 8

        base_euler_rad = np.deg2rad(np.array(BASE_CORRECTION_EULER_DEG, dtype=np.float64))
        self._drake_to_sim_rot = rpy_to_matrix(*base_euler_rad)
        self._sim_to_drake_rot = self._drake_to_sim_rot.T

        self.home_q = HOME_JOINT_POS.astype(np.float64)
        home_pose = self._forward_kinematics(self.home_q)
        self.home_rotation = home_pose["rotation_matrix"]
        observation_pose = self._forward_kinematics(OBSERVE_JOINT_POS.astype(np.float64))
        self.observation_rotation = observation_pose["rotation_matrix"]

        grasp_euler_rad = np.deg2rad(np.array(FIXED_GRASP_EULER_DEG, dtype=np.float64))
        self.fixed_grasp_rotation = rpy_to_matrix(*grasp_euler_rad)

        self._gripper_body_to_prim_rot = _quat_wxyz_to_rotation_matrix(GRIPPER_BODY_TO_PRIM_QUAT_WXYZ)
        self._camera_offset_rot = _quat_wxyz_to_rotation_matrix(CAMERA_LOCAL_QUAT_WXYZ)

    def solve_waypoint(
        self,
        target_pos_base: np.ndarray,
        seed_q: np.ndarray | None = None,
        target_rotation=None,
        orientation_tolerance_rad: float | None = None,
    ) -> dict:
        """求解指定base坐标系下目标位置的关节角。"""
        target_pos_base = target_pos_base.astype(np.float64)
        target_pos_drake = self._sim_to_drake_pos(target_pos_base)

        if seed_q is None:
            seed_q = self.home_q.copy()
        if target_rotation is None:
            target_rotation = self.fixed_grasp_rotation
        if hasattr(target_rotation, 'matrix'):
            target_rotation = target_rotation.matrix()
        target_rotation = np.asarray(target_rotation, dtype=np.float64)
        target_rotation_drake = self._sim_to_drake_rot @ target_rotation

        if orientation_tolerance_rad is None:
            orientation_tolerance_rad = float(self.params.ik_orientation_tolerance_rad)

        ee_grasp_point = np.array(self.params.ee_grasp_point_in_frame, dtype=np.float64)
        q0 = seed_q[:6].astype(np.float64).copy()

        def cost_fn(q6):
            q_full = np.zeros(8, dtype=np.float64)
            q_full[:6] = q6
            T = _forward_kinematics_full(q_full)
            ee_pos = T[:3, 3]
            ee_rot = T[:3, :3]
            grasp_point = ee_pos + ee_rot @ ee_grasp_point
            pos_err = grasp_point - target_pos_drake
            pos_cost = 5000.0 * np.sum(pos_err**2)

            R_err = target_rotation_drake.T @ ee_rot
            trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            angle_err = np.arccos(trace_val)
            ori_cost = 100.0 * angle_err**2

            reg_cost = np.sum((q6 - q0)**2)

            return pos_cost + ori_cost + reg_cost

        result = sp_minimize(
            cost_fn, q0,
            method='L-BFGS-B',
            bounds=list(zip(_JOINT_LOWER, _JOINT_UPPER)),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
        )

        q_sol = np.zeros(8, dtype=np.float64)
        q_sol[:6] = result.x
        q_sol[6] = float(GRIPPER_OPEN[0])
        q_sol[7] = float(GRIPPER_OPEN[1])

        T_sol = _forward_kinematics_full(q_sol)
        ee_pos = T_sol[:3, 3]
        ee_rot = T_sol[:3, :3]
        grasp_point_drake = ee_pos + ee_rot @ ee_grasp_point
        grasp_point_base = self._drake_to_sim_pos(grasp_point_drake)
        pos_error = float(np.linalg.norm(grasp_point_base - target_pos_base))

        if pos_error > max(0.02, self.params.ik_position_tolerance_m * 2.0):
            return {"success": False, "reason": f"ik_pos_error:{pos_error:.4f}"}

        pose = self._forward_kinematics(q_sol)
        cost = float(np.linalg.norm(q_sol[:6] - seed_q[:6]))
        return {
            "success": True,
            "q": q_sol.astype(np.float32),
            "ee_pos_base": pose["position_base"].astype(np.float32),
            "grasp_point_base": grasp_point_base.astype(np.float32),
            "cost": cost,
            "pos_error": pos_error,
        }

    def compute_camera_pose(self, q: np.ndarray) -> dict:
        """计算给定关节角下的相机在base坐标系中的位姿。"""
        ee_pose = self._forward_kinematics(q)
        ee_rot = ee_pose["rotation_matrix"]
        prim_rot = ee_rot @ self._gripper_body_to_prim_rot
        prim_pos = ee_pose["position_base"] + ee_rot @ GRIPPER_BODY_TO_PRIM_POS
        cam_rot_world = prim_rot @ self._camera_offset_rot
        cam_pos = prim_pos + prim_rot @ CAMERA_OFFSET_POSITION
        cam_rot_ros = cam_rot_world @ WORLD_TO_ROS_ROTATION
        return {
            "position_base": cam_pos,
            "rotation_matrix_world": cam_rot_world,
            "rotation_matrix_ros": cam_rot_ros,
            "rotation_matrix": cam_rot_ros,
            "quat_wxyz": _rotation_matrix_to_quat_wxyz(cam_rot_world),
            "quat_wxyz_ros": _rotation_matrix_to_quat_wxyz(cam_rot_ros),
        }

    def _forward_kinematics(self, q: np.ndarray) -> dict:
        """正运动学——计算末端在base坐标系中的位姿。"""
        q = q.astype(np.float64)
        T = _forward_kinematics_full(q)
        pos_drake = T[:3, 3]
        rot_drake = T[:3, :3]
        pos_base = self._drake_to_sim_pos(pos_drake)
        rot = self._drake_to_sim_rot @ rot_drake
        return {"position_base": pos_base, "rotation_matrix": rot, "rotation": rot}

    def _drake_to_sim_pos(self, pos_drake: np.ndarray) -> np.ndarray:
        return self._drake_to_sim_rot @ pos_drake

    def _sim_to_drake_pos(self, pos_sim: np.ndarray) -> np.ndarray:
        return self._sim_to_drake_rot @ pos_sim
