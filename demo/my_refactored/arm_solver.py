"""Piper 机械臂 Drake IK 求解器。

使用 pydrake MultibodyPlant 进行正/逆运动学计算，
处理仿真坐标系(sim)与Drake坐标系之间的180°Z旋转变换。
"""
import os

import numpy as np
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import Solve

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
    _VENDOR_DIR,
)


class PiperArmIK:
    """Piper 机械臂逆运动学求解器。"""

    def __init__(self, params: SystemParams):
        self.params = params
        package_root = os.path.join(_VENDOR_DIR, "piper_description")
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
        self.joint_names = [f"joint{i}" for i in range(1, 9)]
        self.joints = [self.plant.GetJointByName(name, self.model_instance) for name in self.joint_names]
        self.num_positions = self.plant.num_positions(self.model_instance)

        # sim坐标系 与 Drake坐标系 之间的旋转 (绕Z轴180°)
        base_euler_rad = np.deg2rad(np.array(BASE_CORRECTION_EULER_DEG, dtype=np.float64))
        self._drake_to_sim_rot = RollPitchYaw(*base_euler_rad).ToRotationMatrix().matrix()
        self._sim_to_drake_rot = self._drake_to_sim_rot.T

        # 初始化home位姿
        self.home_q = HOME_JOINT_POS.astype(np.float64)
        self._apply_joint_positions(self.home_q)
        home_pose = self._forward_kinematics(self.home_q)
        self.home_rotation = home_pose["rotation"]
        observation_pose = self._forward_kinematics(OBSERVE_JOINT_POS.astype(np.float64))
        self.observation_rotation = observation_pose["rotation"]

        # 固定抓取朝向
        grasp_euler_rad = np.deg2rad(np.array(FIXED_GRASP_EULER_DEG, dtype=np.float64))
        self.fixed_grasp_rotation = RollPitchYaw(*grasp_euler_rad).ToRotationMatrix()

        # 相机挂载变换
        self._gripper_body_to_prim_rot = _quat_wxyz_to_rotation_matrix(GRIPPER_BODY_TO_PRIM_QUAT_WXYZ)
        self._camera_offset_rot = _quat_wxyz_to_rotation_matrix(CAMERA_LOCAL_QUAT_WXYZ)

    def solve_waypoint(
        self,
        target_pos_base: np.ndarray,
        seed_q: np.ndarray | None = None,
        target_rotation: RotationMatrix | None = None,
        orientation_tolerance_rad: float | None = None,
    ) -> dict:
        """求解指定base坐标系下目标位置的关节角。"""
        target_pos_base = target_pos_base.astype(np.float64)
        target_pos_drake = self._sim_to_drake_pos(target_pos_base)

        if seed_q is None:
            seed_q = self.home_q.copy()
        if target_rotation is None:
            target_rotation = self.fixed_grasp_rotation
        target_rotation_drake = self._sim_to_drake_rotation(target_rotation)
        if orientation_tolerance_rad is None:
            orientation_tolerance_rad = float(self.params.ik_orientation_tolerance_rad)

        ik_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(ik_context, self.model_instance, seed_q.astype(np.float64))
        ik = InverseKinematics(self.plant, ik_context)
        q = ik.q()
        prog = ik.prog()

        ee_grasp_point = np.array(self.params.ee_grasp_point_in_frame, dtype=np.float64)
        tol = float(self.params.ik_position_tolerance_m)
        ik.AddPositionConstraint(
            self.ee_frame,
            ee_grasp_point,
            self.plant.world_frame(),
            target_pos_drake - tol,
            target_pos_drake + tol,
        )
        ik.AddPositionCost(
            self.ee_frame,
            ee_grasp_point,
            self.plant.world_frame(),
            target_pos_drake,
            5000.0 * np.eye(3),
        )
        ik.AddOrientationConstraint(
            self.ee_frame,
            RotationMatrix(),
            self.plant.world_frame(),
            target_rotation_drake,
            float(orientation_tolerance_rad),
        )
        prog.AddQuadraticErrorCost(np.eye(self.num_positions), seed_q, q)
        prog.SetInitialGuess(q, seed_q)
        result = Solve(prog)

        if not result.is_success():
            return {"success": False, "reason": "ik_failed"}

        q_sol = result.GetSolution(q)
        q_sol[6] = float(GRIPPER_OPEN[0])
        q_sol[7] = float(GRIPPER_OPEN[1])
        pose = self._forward_kinematics(q_sol)
        grasp_point_base = pose["position_base"] + pose["rotation_matrix"] @ ee_grasp_point
        pos_error = float(np.linalg.norm(grasp_point_base - target_pos_base))

        if pos_error > max(0.02, self.params.ik_position_tolerance_m * 2.0):
            return {"success": False, "reason": f"ik_pos_error:{pos_error:.4f}"}

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
        self._apply_joint_positions(q)
        X_WE = self.plant.CalcRelativeTransform(self.context, self.plant.world_frame(), self.ee_frame)
        pos_drake = np.array(X_WE.translation(), dtype=np.float64)
        rot_drake = X_WE.rotation().matrix()
        pos_base = self._drake_to_sim_pos(pos_drake)
        rot = self._drake_to_sim_rot @ rot_drake
        return {"position_base": pos_base, "rotation": RotationMatrix(rot), "rotation_matrix": rot}

    def _apply_joint_positions(self, q: np.ndarray) -> None:
        self.plant.SetPositions(self.context, self.model_instance, q.astype(np.float64))

    def _drake_to_sim_pos(self, pos_drake: np.ndarray) -> np.ndarray:
        return self._drake_to_sim_rot @ pos_drake

    def _sim_to_drake_pos(self, pos_sim: np.ndarray) -> np.ndarray:
        return self._sim_to_drake_rot @ pos_sim

    def _sim_to_drake_rotation(self, rot_sim: RotationMatrix | np.ndarray) -> RotationMatrix:
        rot_matrix = rot_sim.matrix() if isinstance(rot_sim, RotationMatrix) else np.asarray(rot_sim, dtype=np.float64)
        return RotationMatrix(self._sim_to_drake_rot @ rot_matrix)
