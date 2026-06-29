"""AlgSolution 入口——适配评估框架的统一接口。"""
import os

from .controller import BatchGraspController


class AlgSolution:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.controller = BatchGraspController(base_dir)

    def predicts(self, obs, current_score):
        return self.controller.act(obs, current_score)

    def get_action_spec(self):
        return None

    def reset(self, **kwargs):
        self.controller.reset(**kwargs)
