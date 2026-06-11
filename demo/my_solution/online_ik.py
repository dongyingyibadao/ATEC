"""Online IK solver: Drake-based runtime inverse kinematics.

Replaces the lookup table with exact IK solutions at runtime.
Uses the same Drake MultibodyPlant approach as the precompute script,
but solves for the actual detected object position instead of a grid point.
"""
import math
import os
import numpy as np

from .config import (
    ROBOT_BASE_W, TABLE_TOP_Z,
    OBJECT_GRASP_CONFIGS,
)

# --- Drake IK solver constants (same as precompute_ik_table.py) ---
BASE_FRAME_CORRECTION_XYZ_DEG = (0.0, 0.0, 180.0)
EE_GRASP_POINT = np.array([0.0, 0.0, 0.06], dtype=np.float64)

FIXED_CENTER_Z_BASE = {"sugar": 0.212, "mustard": 0.188, "banana": 0.107}
CLEARANCES = {
    "sugar":   {"pre": 0.10, "lift": 0.10},
    "mustard": {"pre": 0.15, "lift": 0.10},
    "banana":  {"pre": 0.12, "lift": 0.10},
}


def _euler_xyz_deg_to_rotmat(roll_deg, pitch_deg, yaw_deg):
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float64)


def _get_grasp_orientation(obj_name, x_base=0.0):
    cfg = OBJECT_GRASP_CONFIGS[obj_name]
    roll, pitch, yaw = cfg["base_rotation_xyz_deg"]

    if cfg.get("dynamic_rotation_y_deg_near_far") and cfg.get("dynamic_x_base_near_far_m"):
        p_near, p_far = cfg["dynamic_rotation_y_deg_near_far"]
        x_near, x_far = cfg["dynamic_x_base_near_far_m"]
        if abs(x_far - x_near) > 1e-6:
            t = np.clip((x_base - x_near) / (x_far - x_near), 0.0, 1.0)
            pitch = p_near + t * (p_far - p_near)

    rot = _euler_xyz_deg_to_rotmat(roll, pitch, yaw)

    angle = cfg["grasp_angle_rad"]
    if abs(angle) > 1e-6:
        c, s = math.cos(angle), math.sin(angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        rot = rot @ Rz

    return rot


def _apply_anchor_offset(obj_name, obj_x_w, obj_y_w, robot_base_w):
    cfg = OBJECT_GRASP_CONFIGS[obj_name]
    anchor_xyz = cfg.get("anchor_offset_base_xyz", (0.0, 0.0, 0.0))
    obj_x_base = obj_x_w - robot_base_w[0]

    anchor_x = anchor_xyz[0]
    if obj_name == "sugar" and cfg.get("anchor_dynamic_offset_x_near_far_m"):
        x_near_base, x_far_base = cfg["dynamic_x_base_near_far_m"]
        offset_near, offset_far = cfg["anchor_dynamic_offset_x_near_far_m"]
        if abs(x_far_base - x_near_base) > 1e-6:
            t = np.clip((obj_x_base - x_near_base) / (x_far_base - x_near_base), 0.0, 1.0)
            anchor_x = offset_near + t * (offset_far - offset_near)

    if cfg.get("anchor_fixed_y_base_m") is not None:
        ee_y_base = cfg["anchor_fixed_y_base_m"]
        ee_y_w = ee_y_base + robot_base_w[1]
    else:
        anchor_y = anchor_xyz[1]
        ee_y_w = obj_y_w + anchor_y

    ee_x_w = obj_x_w + anchor_x
    return ee_x_w, ee_y_w


def _compute_pregrasp_position(ee_x_w, ee_y_w, pregrasp_z, obj_name, robot_base_w):
    cfg = OBJECT_GRASP_CONFIGS[obj_name]
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


class OnlineIKSolver:
    """Runtime Drake IK solver — exact solutions for any position."""

    def __init__(self):
        from pydrake.multibody.inverse_kinematics import InverseKinematics
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.math import RotationMatrix, RollPitchYaw
        from pydrake.solvers import Solve

        self._InverseKinematics = InverseKinematics
        self._RotationMatrix = RotationMatrix
        self._RollPitchYaw = RollPitchYaw
        self._Solve = Solve

        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        vendor_root = os.path.join(project_dir, "demo", "vendor")
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

        self.home_q = np.array(
            [-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033, 0.035, -0.035],
            dtype=np.float64,
        )

    def _solve_single(self, target_pos_base, target_rot_base, seed_q=None,
                      pos_tol=0.005, orient_tol=0.05):
        target_pos_drake = self.sim_to_drake_base_rot @ target_pos_base.astype(np.float64)
        target_rot_drake = self._RotationMatrix(
            self.sim_to_drake_base_rot @ target_rot_base.astype(np.float64)
        )

        if seed_q is None:
            seed_q = self.home_q.copy()
        seed_q = seed_q[:self.num_positions].astype(np.float64)

        ik_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(ik_context, self.model_instance, seed_q)
        ik = self._InverseKinematics(self.plant, ik_context)
        q = ik.q()
        prog = ik.prog()

        ik.AddPositionConstraint(
            self.ee_frame, EE_GRASP_POINT, self.plant.world_frame(),
            target_pos_drake - pos_tol, target_pos_drake + pos_tol,
        )
        ik.AddPositionCost(
            self.ee_frame, EE_GRASP_POINT, self.plant.world_frame(),
            target_pos_drake, 5000.0 * np.eye(3),
        )
        ik.AddOrientationConstraint(
            self.ee_frame, self._RotationMatrix(), self.plant.world_frame(),
            target_rot_drake, orient_tol,
        )
        prog.AddQuadraticErrorCost(np.eye(self.num_positions), seed_q, q)
        prog.SetInitialGuess(q, seed_q)

        result = self._Solve(prog)
        if not result.is_success():
            return None
        q_sol = result.GetSolution(q)
        q_full = np.zeros(8, dtype=np.float64)
        q_full[:len(q_sol)] = q_sol
        q_full[6] = 0.035
        q_full[7] = -0.035
        return q_full

    def solve_waypoints(self, obj_name: str, obj_x_w: float, obj_y_w: float
                        ) -> dict[str, np.ndarray] | None:
        """Solve IK for pregrasp/grasp/lift waypoints given object world position.

        Args:
            obj_name: "sugar", "mustard", or "banana"
            obj_x_w: object world X coordinate
            obj_y_w: object world Y coordinate

        Returns:
            dict with "pregrasp", "grasp", "lift" — each (8,) joint positions,
            or None if IK fails for any waypoint.
        """
        robot_base_w = ROBOT_BASE_W

        cfg = OBJECT_GRASP_CONFIGS[obj_name]
        center_z = FIXED_CENTER_Z_BASE[obj_name]
        anchor_z = cfg["anchor_offset_base_xyz"][2]
        grasp_z = TABLE_TOP_Z + center_z + anchor_z
        grasp_z = max(grasp_z, TABLE_TOP_Z + 0.01)

        c = CLEARANCES[obj_name]
        pregrasp_z = grasp_z + c["pre"]
        lift_z = grasp_z + c["lift"]

        ee_x, ee_y = _apply_anchor_offset(obj_name, obj_x_w, obj_y_w, robot_base_w)
        pre_x, pre_y, pre_z = _compute_pregrasp_position(
            ee_x, ee_y, pregrasp_z, obj_name, robot_base_w
        )

        obj_x_base = obj_x_w - robot_base_w[0]
        grasp_rot = _get_grasp_orientation(obj_name, obj_x_base)

        positions = [
            (pre_x, pre_y, pre_z),
            (ee_x, ee_y, grasp_z),
            (ee_x, ee_y, lift_z),
        ]
        labels = ["pregrasp", "grasp", "lift"]
        results = {}
        seed_q = None

        for label, (px, py, pz) in zip(labels, positions):
            target_pos_base = np.array([px, py, pz], dtype=np.float64) - robot_base_w
            q_sol = self._solve_single(target_pos_base, grasp_rot, seed_q=seed_q)
            if q_sol is None:
                return None
            results[label] = q_sol.astype(np.float32)
            seed_q = q_sol.copy()

        return results
