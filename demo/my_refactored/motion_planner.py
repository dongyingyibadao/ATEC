"""抓取规划编排 + 关节动作序列构建。

负责:
1. 协调感知(pose_estimator)与评估(grasp_selector)生成抓取方案
2. 将IK解转换为可执行的关节目标序列(带max_delta限制的插值)
"""
import os
from datetime import datetime

import numpy as np
import torch

from .config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CAMERA_FOCAL_MM,
    CAMERA_SENSOR_WIDTH_MM,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    OBSERVE_JOINT_POS,
    PLACE_JOINT_POS,
    SystemParams,
    _VENDOR_DIR,
)
from .arm_solver import PiperArmIK
from .camera_transform import FX, FY, CX, CY, project_depth_to_base, build_workspace_mask
from .pose_estimator import MeshPoseEstimator
from .grasp_selector import CandidateEvaluator


class MotionSequenceBuilder:
    """抓取运动序列构建器——从观测到动作队列的完整管线。"""

    def __init__(self, base_dir: str, params: SystemParams | None = None):
        self.base_dir = base_dir
        self.params = params or SystemParams()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_dir = os.path.join(base_dir, self.params.debug_dirname) if self.params.debug_enabled else None
        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.ik = PiperArmIK(self.params)
        self.pose_estimator = MeshPoseEstimator(self.params, FX)
        self.grasp_selector = CandidateEvaluator(self.params, self.ik)
        self.plan_counter = 0
        self.last_object_pose_debug = None

    def generate_grasp_plan(
        self,
        obs: dict,
        current_q: np.ndarray,
        target_object: str | None = None,
        ignored_objects: set[str] | None = None,
    ) -> dict:
        """从观测数据中生成一个完整的抓取方案。"""
        rgb = self._extract_rgb(obs["image"]["ee_rgb"])
        depth = self._extract_depth(obs["image"]["ee_depth"])
        camera_pose = self.ik.compute_camera_pose(current_q.astype(np.float64))
        points = project_depth_to_base(depth, camera_pose, self.params)
        workspace_mask = build_workspace_mask(depth, self.params)
        candidates, self.last_object_pose_debug = self.pose_estimator.extract_candidates(
            rgb,
            points,
            workspace_mask,
            self.grasp_selector.compute_grasp_anchor,
        )
        ignored = set() if ignored_objects is None else {str(name) for name in ignored_objects}
        if ignored:
            candidates = [c for c in candidates if str(c.get("object_type", "")) not in ignored]
        if target_object is not None:
            candidates = [c for c in candidates if str(c.get("object_type", "")) == str(target_object)]
        solved = self.grasp_selector.rank_candidates(candidates, current_q)

        self.plan_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_name = f"plan_{self.plan_counter:04d}_{timestamp}"
        if self.params.debug_enabled:
            from .visualizer import export_debug_frame
            export_debug_frame(self.debug_dir, plan_name, rgb, depth, workspace_mask, candidates, solved, camera_pose, points, self.last_object_pose_debug)

        if not solved:
            return {
                "success": False,
                "reason": "no_ik_candidate",
                "plan_name": plan_name,
                "target_object": target_object,
                "camera_pose": camera_pose,
            }

        best = solved[0]
        queues = self.build_action_sequence(current_q, best)
        return {
            "success": True,
            "best": best,
            "queues": queues,
            "plan_name": plan_name,
            "target_object": target_object,
            "camera_pose": camera_pose,
        }

    def build_action_sequence(
        self,
        current_q: np.ndarray,
        best: dict,
        *,
        via_observation: bool = True,
        return_to_observation: bool = True,
    ) -> dict[str, list[np.ndarray]]:
        """将IK解转换为pick/place两段关节目标序列。"""
        current = current_q.astype(np.float32)
        observation = OBSERVE_JOINT_POS.astype(np.float32).copy()
        place = PLACE_JOINT_POS.astype(np.float32).copy()
        pre_q = best["pre_q"].astype(np.float32)
        grasp_q = best["grasp_q"].astype(np.float32)
        lift_qs = [q.astype(np.float32) for q in best.get("lift_qs", [best["lift_q"]])]

        pre_q[6:] = GRIPPER_OPEN
        grasp_open = grasp_q.copy()
        grasp_open[6:] = GRIPPER_OPEN
        grasp_closed = grasp_q.copy()
        grasp_closed[6:] = GRIPPER_CLOSE
        lift_closed_targets: list[np.ndarray] = []
        for lift_q in lift_qs:
            lift_closed = lift_q.copy()
            lift_closed[6:] = GRIPPER_CLOSE
            lift_closed_targets.append(lift_closed)
        place_closed = place.copy()
        place_closed[6:] = GRIPPER_CLOSE
        place_open = place.copy()
        place_open[6:] = GRIPPER_OPEN
        observation_closed = observation.copy()
        observation_closed[6:] = GRIPPER_CLOSE
        observation_open = observation.copy()
        observation_open[6:] = GRIPPER_OPEN

        pick_queue: list[np.ndarray] = []
        if via_observation:
            pick_queue.extend(self._interpolate(current, observation_open, self.params.return_steps // 2))
            pick_queue.extend(self._interpolate(observation_open, pre_q, self.params.approach_steps))
        else:
            pick_queue.extend(self._interpolate(current, pre_q, self.params.approach_steps))
        pick_queue.extend(self._interpolate(pre_q, grasp_open, self.params.descend_steps))
        pick_queue.extend(self._interpolate(grasp_open, grasp_closed, self.params.close_steps))
        lift_start = grasp_closed
        lift_step_count = max(1, int(self.params.lift_steps / max(1, len(lift_closed_targets))))
        for lift_closed in lift_closed_targets:
            pick_queue.extend(self._interpolate(lift_start, lift_closed, lift_step_count))
            lift_start = lift_closed

        place_queue: list[np.ndarray] = []
        place_queue.extend(self._interpolate(lift_start, place_closed, self.params.place_steps))
        if self.params.release_steps > 0:
            place_queue.extend([place_closed.astype(np.float32).copy() for _ in range(self.params.release_steps)])
        place_queue.append(place_open.astype(np.float32))
        if self.params.release_steps > 0:
            place_queue.extend([place_open.astype(np.float32).copy() for _ in range(self.params.release_steps)])
        if return_to_observation:
            place_queue.extend(self._interpolate(place_open, observation_open, self.params.return_steps))
        return {"pick_queue": pick_queue, "place_queue": place_queue}

    def _interpolate(self, start: np.ndarray, end: np.ndarray, steps: int) -> list[np.ndarray]:
        """线性插值（保证每步关节变化不超过max_joint_delta_per_step）。"""
        delta = np.abs(end - start)
        min_steps_from_delta = int(np.ceil(float(np.max(delta)) / max(1e-6, self.params.max_joint_delta_per_step)))
        steps = max(steps, min_steps_from_delta)
        if steps <= 0:
            return []
        out = []
        for i in range(1, steps + 1):
            alpha = i / float(steps)
            out.append(((1.0 - alpha) * start + alpha * end).astype(np.float32))
        return out

    def _extract_rgb(self, image_tensor: torch.Tensor) -> np.ndarray:
        """将torch图像tensor转为(H,W,3) uint8 numpy数组。"""
        arr = image_tensor.detach().cpu().numpy()
        arr = arr[0]
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        return arr

    def _extract_depth(self, depth_tensor: torch.Tensor) -> np.ndarray:
        """将torch深度tensor转为(H,W) float32 numpy数组。"""
        arr = depth_tensor.detach().cpu().numpy()
        arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.astype(np.float32)
