"""运动模块: 关节空间线性插值 + action 编码.

Action 格式与学长一致: [[j1, j2, ..., j8]] (带 batch 维度)
"""
import numpy as np
from .config import (
    DEFAULT_JOINT_POS, ACTION_SCALE,
    MAX_JOINT_DELTA_PER_STEP,
    GRIPPER_OPEN, GRIPPER_CLOSE,
)


def joints_to_action(target_joints: np.ndarray, q_batch: int = 1) -> list:
    """将绝对关节角转换为 action (符合 predicts() 返回格式).

    action = (target - DEFAULT_JOINT_POS) / ACTION_SCALE
    返回带 batch 维度: [[j1, j2, ..., j8]]
    """
    action = (target_joints.astype(np.float32) - DEFAULT_JOINT_POS) / ACTION_SCALE
    return action.reshape(q_batch, -1).astype(np.float32).tolist()


def interpolate(start: np.ndarray, end: np.ndarray, min_steps: int) -> list[np.ndarray]:
    """在两个关节角之间做线性插值, 保证每步不超过 MAX_JOINT_DELTA_PER_STEP.

    Args:
        start: shape (8,) 起始关节角
        end: shape (8,) 目标关节角
        min_steps: 最少步数

    Returns:
        list of np.ndarray shape (8,), 不含 start, 包含 end
    """
    delta = np.abs(end - start)
    max_delta = float(np.max(delta))
    steps_from_delta = int(np.ceil(max_delta / MAX_JOINT_DELTA_PER_STEP)) if max_delta > 1e-6 else 0
    steps = max(min_steps, steps_from_delta)
    if steps <= 0:
        return []
    out = []
    for i in range(1, steps + 1):
        alpha = i / float(steps)
        out.append(((1.0 - alpha) * start + alpha * end).astype(np.float32))
    return out


def set_gripper(joints: np.ndarray, gripper_open: bool) -> np.ndarray:
    """返回关节角的 copy, 夹爪部分设为 open 或 close."""
    result = joints.copy().astype(np.float32)
    result[6:] = GRIPPER_OPEN if gripper_open else GRIPPER_CLOSE
    return result
