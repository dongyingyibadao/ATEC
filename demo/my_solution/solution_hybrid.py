"""
混合方案 AlgSolution 入口.

使用方法: 修改 demo/solution.py 指向此模块:
    from .my_solution.solution_hybrid import AlgSolution
"""
import os
import numpy as np

from .controller import BatchCycleController


class AlgSolution:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        camera_pose_path = os.path.join(data_dir, "camera_pose.npy")

        if not os.path.isfile(camera_pose_path):
            raise FileNotFoundError(
                f"camera_pose.npy not found at {camera_pose_path}. "
                "Run scripts/precompute_ik_table.py in Isaac Lab first."
            )

        camera_pose = np.load(camera_pose_path, allow_pickle=True).item()
        self.controller = BatchCycleController(camera_pose)

    def predicts(self, obs, current_score):
        return self.controller.act(obs, current_score)

    def get_action_spec(self):
        return None

    def reset(self, **kwargs):
        self.controller.reset(**kwargs)
