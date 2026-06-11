"""Trajectory-replay solution entry point for Task E.

Replaces online-IK planning with trajectory retrieval:
    1. Perceive object positions (6-dim key)
    2. Query pre-collected trajectory database (nearest neighbor)
    3. Replay matched action sequence
"""
import os
import numpy as np

from .replay_controller import ReplayController


class AlgSolution:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        camera_pose_path = os.path.join(data_dir, "camera_pose.npy")
        trajectory_dir = os.path.join(data_dir, "trajectories")

        if not os.path.isfile(camera_pose_path):
            raise FileNotFoundError(
                f"camera_pose.npy not found at {camera_pose_path}. "
                "Run scripts/precompute_camera_pose.py in Isaac Lab first."
            )

        if not os.path.isdir(trajectory_dir):
            raise FileNotFoundError(
                f"Trajectory directory not found at {trajectory_dir}. "
                "Run scripts/collect_trajectories.py first."
            )

        camera_pose = np.load(camera_pose_path, allow_pickle=True).item()
        self.controller = ReplayController(camera_pose, trajectory_dir)

    def predicts(self, obs, current_score):
        return self.controller.act(obs, current_score)

    def get_action_spec(self):
        return None

    def reset(self, **kwargs):
        self.controller.reset(**kwargs)
