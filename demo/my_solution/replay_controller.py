"""Replay controller: perceive objects → query trajectory DB → replay actions."""
import os
import numpy as np
import torch

from .config import (
    DEFAULT_JOINT_POS,
    OBSERVATION_JOINT_POS,
    ACTION_SCALE,
    GRIPPER_OPEN,
)
from .trajectory_db import TrajectoryDB
from .perception import Perception


class ReplayController:
    """Trajectory-replay controller for Task E.

    Flow:
        1. Move to observation position
        2. Wait for joints to settle
        3. Run perception to get 6-dim object key
        4. Query trajectory database for nearest neighbor
        5. Replay matched action sequence step-by-step
    """

    SETTLE_STEPS = 15
    OBSERVATION_TOL = 0.04

    def __init__(self, camera_pose: dict, trajectory_dir: str):
        self.perception = Perception(camera_pose)
        self.db = TrajectoryDB(trajectory_dir)
        self._reset_state()
        print(f"[ReplayCtrl] loaded {self.db.size} trajectories from {trajectory_dir}")

    def reset(self, **kwargs):
        self._reset_state()

    def _reset_state(self):
        self.stage = "move_to_obs"
        self._settle_counter = 0
        self._startup_step = 0
        self._replay_actions: np.ndarray | None = None
        self._replay_index = 0
        self._match_distance = -1.0

    def act(self, obs: dict, current_score: float) -> dict:
        proprio = obs["proprio"]
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.detach().cpu().numpy()
        qpos = proprio[:, :8] + DEFAULT_JOINT_POS[None, :]
        q_batch = qpos.shape[0]

        if self._startup_step < 5:
            self._startup_step += 1
            return self._zero_action(q_batch)

        if self.stage == "replay":
            return self._step_replay(q_batch)

        if self.stage == "move_to_obs":
            obs_target = OBSERVATION_JOINT_POS.copy()
            obs_target[6:] = GRIPPER_OPEN
            err = float(np.max(np.abs(qpos[0].astype(np.float32) - obs_target)))
            if err > self.OBSERVATION_TOL:
                self._settle_counter = 0
                return self._target_action(obs_target, q_batch)
            self._settle_counter += 1
            if self._settle_counter < self.SETTLE_STEPS:
                return self._target_action(obs_target, q_batch)
            self.stage = "perceive"

        if self.stage == "perceive":
            key = self._extract_key(obs)
            if key is None:
                print("[ReplayCtrl] perception failed, sending zero action")
                return self._zero_action(q_batch)

            actions, dist = self.db.query(key)
            self._replay_actions = actions
            self._replay_index = 0
            self._match_distance = dist
            self.stage = "replay"
            print(f"[ReplayCtrl] matched trajectory: dist={dist:.4f}, "
                  f"steps={len(actions)}, key={key.round(3)}")
            return self._step_replay(q_batch)

        return self._zero_action(q_batch)

    def _step_replay(self, q_batch: int) -> dict:
        if self._replay_actions is None or self._replay_index >= len(self._replay_actions):
            return self._zero_action(q_batch)
        action = self._replay_actions[self._replay_index]
        self._replay_index += 1
        action_2d = action.reshape(1, -1).astype(np.float32)
        return {"action": np.broadcast_to(action_2d, (q_batch, 8)).tolist(), "giveup": False}

    def _extract_key(self, obs: dict) -> np.ndarray | None:
        """Extract 6-dim key: [mustard_x, mustard_y, sugar_x, sugar_y, banana_x, banana_y]."""
        depth = self._get_depth(obs)
        positions = self.perception.locate_all_objects(depth)

        key_order = ("mustard", "sugar", "banana")
        key_parts = []
        for obj_name in key_order:
            pos = positions.get(obj_name)
            if pos is None:
                print(f"[ReplayCtrl] object '{obj_name}' not detected")
                return None
            key_parts.extend([float(pos[0]), float(pos[1])])

        return np.array(key_parts, dtype=np.float32)

    def _get_depth(self, obs: dict) -> np.ndarray:
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

    def _target_action(self, target: np.ndarray, q_batch: int) -> dict:
        action = (target.astype(np.float32) - DEFAULT_JOINT_POS) / ACTION_SCALE
        action_2d = action.reshape(1, -1)
        return {"action": np.broadcast_to(action_2d, (q_batch, 8)).tolist(), "giveup": False}

    def _zero_action(self, q_batch: int) -> dict:
        action = np.zeros((q_batch, 8), dtype=np.float32)
        return {"action": action.tolist(), "giveup": False}
