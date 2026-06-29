"""抓取候选点评估与IK排序。

将位姿估计产出的候选点转化为完整的抓取方案:
锚点计算 → 预抓取位置 → IK求解 → 抬升验证 → 综合评分排序。
"""
import math

import numpy as np
from .rotation_utils import rpy_to_matrix, compose_rotation

from .config import GRASP_TEMPLATES, TEMPLATE_FALLBACK_ORDER, SystemParams, GraspTemplate


def _compose_grasp_orientation(base_rotation: np.ndarray, angle_rad: float) -> np.ndarray:
    """将基础朝向与绕Z轴旋转角组合为最终抓取朝向。"""
    return compose_rotation(base_rotation, angle_rad)


def _discretize_angle(angle_rad: float) -> float:
    """将连续角度离散化为0或pi/3。"""
    angle_deg = math.degrees(angle_rad)
    angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
    if 45.0 <= abs(angle_deg) <= 135.0:
        return math.pi / 3.0
    return 0.0


class CandidateEvaluator:
    """抓取候选评估器——对候选点进行IK求解并排序。"""

    def __init__(self, params: SystemParams, ik):
        self.params = params
        self.ik = ik

    def get_template(self, name: str) -> GraspTemplate | None:
        """根据名称获取抓取模板。"""
        return GRASP_TEMPLATES.get(name)

    def classify_by_position(self, surface_base: np.ndarray) -> GraspTemplate:
        """根据Y坐标分类物体模板。"""
        y_base = float(surface_base[1])
        for name in TEMPLATE_FALLBACK_ORDER:
            template = GRASP_TEMPLATES[name]
            if y_base >= template.y_min_base:
                return template
        return GRASP_TEMPLATES[TEMPLATE_FALLBACK_ORDER[-1]]

    def compute_grasp_anchor(self, object_type: str, object_center_base: np.ndarray) -> np.ndarray:
        """计算物体的抓取锚点（考虑模板偏移）。"""
        template = self.get_template(object_type)
        center = np.asarray(object_center_base, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            return np.zeros(3, dtype=np.float32)

        anchor = center.copy()
        if template is not None:
            offset_xyz = np.asarray(template.anchor.offset_base_xyz, dtype=np.float64).copy()
            dynamic_offset_x = self._compute_dynamic_offset_x(template, center[0])
            if dynamic_offset_x is not None:
                offset_xyz[0] = dynamic_offset_x
            anchor += offset_xyz
            if template.anchor.fixed_y_base_m is not None:
                anchor[1] = float(template.anchor.fixed_y_base_m)
        return anchor.astype(np.float32)

    def rank_candidates(self, candidates: list[dict], current_q: np.ndarray) -> list[dict]:
        """对候选列表执行IK求解并按分数排序。"""
        solved = []
        seed_q = current_q.astype(np.float64)
        for candidate in candidates:
            rank_debug: dict[str, object] = {}
            template = self.get_template(candidate.get("object_type", "")) or self.classify_by_position(candidate["surface_base"])
            rank_debug["template"] = template.name

            raw_angle = float(candidate["angle"]) * float(self.params.grasp_angle_scale)
            angle = _discretize_angle(raw_angle)
            if candidate.get("candidate_source") == "mesh_pose" and template.base_rotation_xyz_deg is not None:
                angle = template.grasp_angle_rad
            if abs(template.grasp_angle_rad) > 1e-6:
                angle = template.grasp_angle_rad

            rank_debug["raw_angle_rad"] = float(raw_angle)
            rank_debug["template_angle_rad"] = float(angle)

            if candidate.get("candidate_source") == "mesh_pose" and template.base_rotation_xyz_deg is None and template.name != "sugar":
                base_rotation = self.ik.home_rotation
            else:
                object_center_base = candidate.get("object_center_base")
                object_center_x = None
                if object_center_base is not None:
                    object_center_arr = np.asarray(object_center_base, dtype=np.float64)
                    if object_center_arr.shape == (3,) and np.all(np.isfinite(object_center_arr)):
                        object_center_x = float(object_center_arr[0])
                base_rotation = self._compute_base_rotation(template, object_center_x)
            target_rotation = _compose_grasp_orientation(base_rotation, angle)
            orientation_tolerance_rad = float(self.params.ik_orientation_tolerance_rad)

            surf = candidate["surface_base"].astype(np.float64)
            if candidate.get("candidate_source") == "mesh_pose" and candidate.get("grasp_anchor_base") is not None:
                grasp = np.asarray(candidate["grasp_anchor_base"], dtype=np.float64).copy()
            else:
                grasp = surf.copy()

            pregrasp = grasp.copy()
            pregrasp[2] += template.approach.pregrasp_clearance_m
            xy_norm = float(np.linalg.norm(grasp[:2]))
            if xy_norm > 1e-6:
                pregrasp[:2] -= (grasp[:2] / xy_norm) * template.approach.retreat_toward_base_m

            lift_clearance_m = float(template.approach.lift_clearance_m)
            lift = grasp.copy()
            lift[2] += lift_clearance_m

            rank_debug["surface_base"] = surf.astype(np.float32)
            rank_debug["lift_base"] = lift.astype(np.float32)
            pre_attempts: list[dict[str, object]] = []
            pre_res, grasp, pregrasp = self._solve_pregrasp_with_fallback(
                template=template,
                grasp=grasp,
                pregrasp=pregrasp,
                seed_q=seed_q,
                target_rotation=target_rotation,
                orientation_tolerance_rad=orientation_tolerance_rad,
                pre_attempts=pre_attempts,
            )
            rank_debug["pregrasp_attempts"] = pre_attempts
            rank_debug["grasp_base"] = grasp.astype(np.float32)
            rank_debug["pregrasp_base"] = pregrasp.astype(np.float32)
            if not pre_res["success"]:
                rank_debug["status"] = "pregrasp_failed"
                rank_debug["pregrasp_result"] = pre_res
                candidate["rank_debug"] = rank_debug
                continue

            grasp_res = self.ik.solve_waypoint(
                grasp,
                seed_q=pre_res["q"].astype(np.float64),
                target_rotation=target_rotation,
                orientation_tolerance_rad=orientation_tolerance_rad,
            )
            if not grasp_res["success"]:
                rank_debug["status"] = "grasp_failed"
                rank_debug["pregrasp_result"] = pre_res
                rank_debug["grasp_result"] = grasp_res
                candidate["rank_debug"] = rank_debug
                continue

            lift_qs: list[np.ndarray] = []
            lift_bases: list[np.ndarray] = []
            lift_cost = 0.0
            if lift_clearance_m <= 1e-6:
                lift_qs = [grasp_res["q"].astype(np.float32)]
                lift_bases = [grasp.astype(np.float32)]
            else:
                grasp_pose = self.ik._forward_kinematics(grasp_res["q"].astype(np.float64))
                lift_rotation = grasp_pose["rotation"]
                lift_seed_q = grasp_res["q"].astype(np.float64)
                num_lift_waypoints = max(1, int(self.params.lift_num_waypoints))
                for lift_idx in range(1, num_lift_waypoints + 1):
                    alpha = float(lift_idx) / float(num_lift_waypoints)
                    lift_wp = grasp.copy()
                    lift_wp[2] += lift_clearance_m * alpha
                    lift_res = self.ik.solve_waypoint(
                        lift_wp,
                        seed_q=lift_seed_q,
                        target_rotation=lift_rotation,
                        orientation_tolerance_rad=orientation_tolerance_rad,
                    )
                    if not lift_res["success"]:
                        rank_debug["lift_failure_index"] = lift_idx
                        rank_debug["lift_failure_result"] = lift_res
                        lift_qs = []
                        break
                    lift_qs.append(lift_res["q"])
                    lift_bases.append(lift_wp.astype(np.float32))
                    lift_cost += float(lift_res["cost"])
                    lift_seed_q = lift_res["q"].astype(np.float64)
                if not lift_qs:
                    rank_debug["status"] = "lift_failed"
                    rank_debug["pregrasp_result"] = pre_res
                    rank_debug["grasp_result"] = grasp_res
                    candidate["rank_debug"] = rank_debug
                    continue

            total_cost = pre_res["cost"] + grasp_res["cost"] + lift_cost - candidate["quality"] * 0.5
            rank_debug["status"] = "solved"
            rank_debug["pregrasp_result"] = pre_res
            rank_debug["grasp_result"] = grasp_res
            candidate["rank_debug"] = rank_debug
            solved.append(
                {
                    **candidate,
                    "object_type": template.name,
                    "template_angle_rad": float(angle),
                    "pregrasp_base": pregrasp.astype(np.float32),
                    "pre_q": pre_res["q"],
                    "grasp_base": grasp.astype(np.float32),
                    "grasp_q": grasp_res["q"],
                    "lift_base": lift.astype(np.float32),
                    "lift_q": lift_qs[-1],
                    "lift_qs": [q.astype(np.float32) for q in lift_qs],
                    "lift_bases": lift_bases,
                    "grasp_height": float(grasp[2]),
                    "score": float(total_cost),
                }
            )

        solved.sort(key=lambda c: (-c["grasp_height"], c["score"]))
        return solved[: self.params.candidate_keep_best]

    def _solve_pregrasp_with_fallback(
        self,
        template: GraspTemplate,
        grasp: np.ndarray,
        pregrasp: np.ndarray,
        seed_q: np.ndarray,
        target_rotation: np.ndarray,
        orientation_tolerance_rad: float,
        pre_attempts: list[dict[str, object]],
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """预抓取IK求解（sugar失败时沿Y轴回退重试）。"""
        def _record(step_idx, y_shift_m, g, p, res):
            pre_attempts.append({
                "attempt_index": int(step_idx),
                "left_shift_y_m": float(y_shift_m),
                "grasp_base": g.astype(np.float32),
                "pregrasp_base": p.astype(np.float32),
                "result": res,
            })

        pre_res = self.ik.solve_waypoint(
            pregrasp,
            seed_q=seed_q,
            target_rotation=target_rotation,
            orientation_tolerance_rad=orientation_tolerance_rad,
        )
        _record(0, 0.0, grasp, pregrasp, pre_res)
        if pre_res["success"] or template.name != "sugar":
            return pre_res, grasp, pregrasp

        step_m = float(self.params.sugar_pregrasp_fallback_step_m)
        max_steps = max(0, int(self.params.sugar_pregrasp_fallback_max_steps))
        for step_idx in range(1, max_steps + 1):
            y_shift_m = -step_m * float(step_idx)
            grasp_try = grasp.copy()
            pregrasp_try = pregrasp.copy()
            grasp_try[1] += y_shift_m
            pregrasp_try[1] += y_shift_m
            pre_res_try = self.ik.solve_waypoint(
                pregrasp_try,
                seed_q=seed_q,
                target_rotation=target_rotation,
                orientation_tolerance_rad=orientation_tolerance_rad,
            )
            _record(step_idx, y_shift_m, grasp_try, pregrasp_try, pre_res_try)
            if pre_res_try["success"]:
                return pre_res_try, grasp_try, pregrasp_try

        return pre_res, grasp, pregrasp

    def _compute_base_rotation(self, template: GraspTemplate, object_center_x: float | None = None) -> np.ndarray:
        """计算模板指定的基础朝向（考虑动态俯仰角）。"""
        if template.base_rotation_xyz_deg is None:
            return self.ik.fixed_grasp_rotation
        xyz_deg = np.array(template.base_rotation_xyz_deg, dtype=np.float64)
        dynamic_rot_y = self._compute_dynamic_pitch(template, object_center_x)
        if dynamic_rot_y is not None:
            xyz_deg[1] = dynamic_rot_y
        xyz_rad = np.deg2rad(xyz_deg)
        return rpy_to_matrix(*xyz_rad)

    def _interpolate_alpha(self, template: GraspTemplate, object_center_x: float | None) -> float | None:
        """计算近远端插值系数alpha。"""
        if (
            template.dynamic_x_base_near_far_m is None
            or object_center_x is None
            or not np.isfinite(object_center_x)
        ):
            return None
        near_x, far_x = template.dynamic_x_base_near_far_m
        denom = float(near_x - far_x)
        if abs(denom) < 1e-6:
            return 0.0
        alpha = float((near_x - object_center_x) / denom)
        return float(np.clip(alpha, 0.0, 1.0))

    def _compute_dynamic_pitch(self, template: GraspTemplate, object_center_x: float | None) -> float | None:
        """根据物体X坐标动态计算俯仰角。"""
        alpha = self._interpolate_alpha(template, object_center_x)
        if alpha is None or template.dynamic_rotation_y_deg_near_far is None:
            return None
        near_y, far_y = template.dynamic_rotation_y_deg_near_far
        return float((1.0 - alpha) * near_y + alpha * far_y)

    def _compute_dynamic_offset_x(self, template: GraspTemplate, object_center_x: float | None) -> float | None:
        """根据物体X坐标动态计算X偏移。"""
        alpha = self._interpolate_alpha(template, object_center_x)
        if alpha is None or template.anchor.dynamic_offset_x_near_far_m is None:
            return None
        near_x, far_x = template.anchor.dynamic_offset_x_near_far_m
        return float((1.0 - alpha) * near_x + alpha * far_x)
