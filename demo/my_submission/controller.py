"""批量循环抓取控制器。

工作流程:
1. 启动时移动到观测位
2. 观测一次，为所有目标物体规划抓取方案
3. 按顺序循环执行 pick→place 直到方案耗尽
"""
import numpy as np

from .config import ACTION_SCALE, DEFAULT_JOINT_POS, GRIPPER_OPEN, OBSERVE_JOINT_POS, SystemParams
from .motion_planner import MotionSequenceBuilder


class BatchGraspController:
    """批量抓取循环控制器——一次观测规划全部物体。"""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.params = SystemParams()
        self.planner = MotionSequenceBuilder(base_dir, self.params)
        self._init_state()

    def _init_state(self) -> None:
        """初始化/重置运行时状态。"""
        self.pending_targets: list[np.ndarray] = []
        self.stage = "idle"
        self.startup_step = 0
        self.cached_plans: list[dict] = []
        self.cached_plan_index = 0
        self.current_exec_plan: dict | None = None
        self.current_place_queue: list[np.ndarray] = []
        self.needs_initial_observation = True

    def reset(self, **_: dict) -> None:
        """重置控制器状态（每个episode开始时调用）。"""
        self._init_state()

    def act(self, obs: dict, current_score: float) -> dict:
        """根据当前观测生成动作。"""
        del current_score
        proprio = obs["proprio"].detach().cpu().numpy()
        qpos = proprio[:, :8] + DEFAULT_JOINT_POS[None, :]

        if self.startup_step < self.params.startup_zero_steps:
            self.startup_step += 1
            action = np.zeros((qpos.shape[0], 8), dtype=np.float32)
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
            return self._encode_action(target, qpos.shape[0])

        if self.needs_initial_observation:
            if not self._at_observation_position(qpos):
                return self._encode_action(self._observation_target(), qpos.shape[0])
            if not self.planner or not self.planner.ik:
                action = np.zeros((qpos.shape[0], 8), dtype=np.float32)
                return {"action": action.tolist(), "giveup": False}
            if not self.cached_plans:
                self.cached_plans = self._plan_all_objects(obs, qpos[0])
            self.needs_initial_observation = False

        target = self._start_next_plan(qpos[0].astype(np.float32))
        if target is None:
            action = np.zeros((qpos.shape[0], 8), dtype=np.float32)
            return {"action": action.tolist(), "giveup": False}
        return self._encode_action(target, qpos.shape[0])

    def _plan_all_objects(self, obs: dict, current_q: np.ndarray) -> list[dict]:
        """为所有目标物体批量规划抓取方案。"""
        plans: list[dict] = []
        for object_name in self.params.task_e_object_order:
            plan = self.planner.generate_grasp_plan(
                obs,
                current_q,
                target_object=object_name,
                ignored_objects=None,
            )
            if not plan["success"]:
                continue
            plans.append(
                {
                    "target_object": object_name,
                    "best": plan["best"],
                }
            )
        return plans

    def _start_next_plan(self, current_q: np.ndarray) -> np.ndarray | None:
        """启动下一个缓存的抓取方案。"""
        if not self.cached_plans:
            return None
        plan = self.cached_plans[self.cached_plan_index % len(self.cached_plans)]
        self.cached_plan_index = (self.cached_plan_index + 1) % len(self.cached_plans)
        self.current_exec_plan = plan
        queues = self.planner.build_action_sequence(
            current_q,
            plan["best"],
            via_observation=False,
            return_to_observation=False,
        )
        self.pending_targets = queues["pick_queue"].copy()
        self.current_place_queue = queues["place_queue"].copy()
        self.stage = "pick"
        if not self.pending_targets:
            return None
        return self.pending_targets.pop(0)

    def _encode_action(self, target: np.ndarray, q_batch: int) -> dict:
        """将目标关节角编码为归一化动作。"""
        action = (target - DEFAULT_JOINT_POS) / ACTION_SCALE
        return {"action": action.reshape(q_batch, -1).astype(np.float32).tolist(), "giveup": False}

    def _observation_target(self) -> np.ndarray:
        """观测位的关节目标。"""
        target = OBSERVE_JOINT_POS.copy()
        target[6:] = GRIPPER_OPEN
        return target.astype(np.float32)

    def _at_observation_position(self, qpos: np.ndarray) -> bool:
        """判断当前关节是否已到达观测位。"""
        observation_target = self._observation_target()
        err = float(np.max(np.abs(qpos[0].astype(np.float32) - observation_target)))
        return err <= float(self.params.observation_ready_joint_tol)
