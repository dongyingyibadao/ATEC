"""顶层控制器: batch-cycle 模式.

对标学长 DrakeBatchCycleController 的执行模式:
    1. IDLE → 移动到观察位 (等到位后感知)
    2. OBSERVE → 一次性定位所有物体, 在线 Drake IK 求解 waypoints
    3. EXECUTE → 连续执行 pick→place × 3, 物体间不回观察位 (via_observation=False)
"""
import numpy as np
import torch

from .config import (
    DEFAULT_JOINT_POS,
    OBSERVATION_JOINT_POS, PLACE_JOINT_POS,
    GRIPPER_OPEN, GRIPPER_CLOSE,
    OBJECT_ORDER,
    APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS,
    LIFT_STEPS, PLACE_STEPS, RELEASE_STEPS, RETURN_STEPS,
)
from .perception import Perception
from .online_ik import OnlineIKSolver
from .motion import joints_to_action, interpolate, set_gripper


class BatchCycleController:
    """混合方案顶层控制器.

    实现 act(obs, current_score) → {"action": [[...]], "giveup": bool}
    """

    def __init__(self, camera_pose: dict):
        self.perception = Perception(camera_pose)
        self.ik_solver = OnlineIKSolver()
        self._reset_state()

    def reset(self, **kwargs):
        self._reset_state()

    SETTLE_STEPS_AFTER_CONVERGE = 15

    def _reset_state(self):
        self.pending_targets: list[np.ndarray] = []
        self.stage = "idle"
        self.startup_step = 0
        self.needs_initial_observation = True
        self.cached_cycle_plans: list[dict] = []
        self.cached_cycle_index = 0
        self.current_place_queue: list[np.ndarray] = []
        self.observation_ready_tol = 0.04
        self._settle_counter = 0

    def act(self, obs: dict, current_score: float) -> dict:
        """每步调用一次. 对标学长 DrakeBatchCycleController.act()."""
        proprio = obs["proprio"]
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.detach().cpu().numpy()
        qpos = proprio[:, :8] + DEFAULT_JOINT_POS[None, :]
        q_batch = qpos.shape[0]

        self._step_count = getattr(self, "_step_count", 0) + 1
        if self._step_count <= 10 or self._step_count % 500 == 0:
            print(f"[HybridCtrl] step={self._step_count} stage={self.stage} "
                  f"pending={len(self.pending_targets)} needs_obs={self.needs_initial_observation} "
                  f"qpos[:6]={qpos[0,:6].round(3)}")

        if self.startup_step < 5:
            self.startup_step += 1
            action = np.zeros((q_batch, 8), dtype=np.float32)
            return {"action": action.tolist(), "giveup": False}

        if self.pending_targets:
            target = self.pending_targets.pop(0)
            if not self.pending_targets:
                if self.stage == "pick":
                    self.pending_targets = self.current_place_queue.copy()
                    self.current_place_queue = []
                    self.stage = "place"
                    if not self.pending_targets:
                        self.stage = "idle"
                elif self.stage == "place":
                    self.stage = "idle"
            return {"action": joints_to_action(target, q_batch), "giveup": False}

        if self.needs_initial_observation:
            obs_target = set_gripper(OBSERVATION_JOINT_POS, gripper_open=True)
            err = float(np.max(np.abs(qpos[0].astype(np.float32) - obs_target)))
            if err > self.observation_ready_tol:
                self._settle_counter = 0
                return {"action": joints_to_action(obs_target, q_batch), "giveup": False}
            self._settle_counter += 1
            if self._settle_counter < self.SETTLE_STEPS_AFTER_CONVERGE:
                if self._settle_counter == 1:
                    print(f"[HybridCtrl] joints converged (err={err:.4f}), waiting {self.SETTLE_STEPS_AFTER_CONVERGE} settle steps...")
                return {"action": joints_to_action(obs_target, q_batch), "giveup": False}
            print(f"[HybridCtrl] settle complete after {self._settle_counter} steps, triggering perception")
            self._do_perception_and_plan(obs)
            self.needs_initial_observation = False

        target = self._start_next_cached_plan(qpos[0].astype(np.float32))
        if target is None:
            action = np.zeros((q_batch, 8), dtype=np.float32)
            return {"action": action.tolist(), "giveup": False}
        return {"action": joints_to_action(target, q_batch), "giveup": False}

    def _do_perception_and_plan(self, obs: dict):
        """一次性感知所有物体并在线求解 IK."""
        depth = self._extract_depth(obs)
        print(f"[HybridCtrl] perception: depth shape={depth.shape}, "
              f"min={depth.min():.3f}, max={depth.max():.3f}, "
              f"nonzero={np.count_nonzero(depth > 0.01)}")
        positions = self.perception.locate_all_objects(depth)
        print(f"[HybridCtrl] perception results: {{{', '.join(f'{k}: {v}' for k, v in positions.items())}}}")

        self.cached_cycle_plans = []
        for obj_name in OBJECT_ORDER:
            obj_pos = positions.get(obj_name)
            if obj_pos is None:
                continue
            waypoints = self.ik_solver.solve_waypoints(obj_name, float(obj_pos[0]), float(obj_pos[1]))
            if waypoints is None:
                print(f"[HybridCtrl] IK solve FAILED for {obj_name} at ({obj_pos[0]:.3f}, {obj_pos[1]:.3f})")
                continue
            self.cached_cycle_plans.append({
                "target_object": obj_name,
                "waypoints": waypoints,
            })
        self.cached_cycle_index = 0
        print(f"[HybridCtrl] planned {len(self.cached_cycle_plans)} objects")

    def _start_next_cached_plan(self, current_q: np.ndarray) -> np.ndarray | None:
        """开始执行下一个 cached plan. 返回第一个 target 或 None."""
        if self.cached_cycle_index >= len(self.cached_cycle_plans):
            return None

        plan = self.cached_cycle_plans[self.cached_cycle_index]
        self.cached_cycle_index += 1
        waypoints = plan["waypoints"]

        print(f"[HybridCtrl] executing plan {self.cached_cycle_index}/{len(self.cached_cycle_plans)} "
              f"obj={plan['target_object']} "
              f"pregrasp[:4]={waypoints['pregrasp'][:4].round(3)} "
              f"grasp[:4]={waypoints['grasp'][:4].round(3)}")

        pregrasp = set_gripper(waypoints["pregrasp"], gripper_open=True)
        grasp_open = set_gripper(waypoints["grasp"], gripper_open=True)
        grasp_closed = set_gripper(waypoints["grasp"], gripper_open=False)
        lift_closed = set_gripper(waypoints["lift"], gripper_open=False)
        place_closed = set_gripper(PLACE_JOINT_POS, gripper_open=False)
        place_open = set_gripper(PLACE_JOINT_POS, gripper_open=True)
        observation_open = set_gripper(OBSERVATION_JOINT_POS, gripper_open=True)

        pick_queue: list[np.ndarray] = []
        pick_queue.extend(interpolate(current_q, pregrasp, APPROACH_STEPS))
        pick_queue.extend(interpolate(pregrasp, grasp_open, DESCEND_STEPS))
        pick_queue.extend(interpolate(grasp_open, grasp_closed, CLOSE_STEPS))
        pick_queue.extend(interpolate(grasp_closed, lift_closed, LIFT_STEPS))

        place_queue: list[np.ndarray] = []
        place_queue.extend(interpolate(lift_closed, place_closed, PLACE_STEPS))
        place_queue.extend([place_closed.copy() for _ in range(RELEASE_STEPS)])
        place_queue.append(place_open.copy())
        place_queue.extend([place_open.copy() for _ in range(RELEASE_STEPS)])
        place_queue.extend(interpolate(place_open, observation_open, RETURN_STEPS))

        self.pending_targets = pick_queue
        self.current_place_queue = place_queue
        self.stage = "pick"

        print(f"[HybridCtrl] queued: pick={len(pick_queue)} place={len(place_queue)}")

        if not self.pending_targets:
            return None
        return self.pending_targets.pop(0)

    def _extract_depth(self, obs: dict) -> np.ndarray:
        """从 obs 中提取 depth 图, 处理 tensor/batch/channel 维度."""
        depth_tensor = obs["image"]["ee_depth"]
        if isinstance(depth_tensor, torch.Tensor):
            depth = depth_tensor.detach().cpu().numpy()
        else:
            depth = np.asarray(depth_tensor)
        if depth.ndim == 4:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        return depth.astype(np.float32)
