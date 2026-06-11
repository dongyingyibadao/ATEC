"""Record Task E with a close-up camera view of the table.

Usage:
    PYTHONPATH="$(pwd):$(pwd)/scripts:$PYTHONPATH" python scripts/record_task_e.py [--solution senior|mine] [--video_length 2000]

This script wraps play_atec_task.py logic but overrides the viewer camera
to provide a close-up view of the table and robot arm.
"""
import argparse
import os
import time
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record Task E with close-up camera.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of video in steps.")
parser.add_argument("--solution", type=str, default="mine", choices=["senior", "mine", "refactored"],
                    help="Which solution to use: 'senior', 'mine', or 'refactored'.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--debug", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.task = "ATEC-TaskE-Piper"
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
import atec_rl_lab.tasks
from isaaclab_tasks.utils import parse_env_cfg
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec

if args_cli.solution == "senior":
    from demo.tasks.task_e.grasp_drake_batch_cycle import DrakeBatchCycleController
    print("[record] Using SENIOR solution (DrakeBatchCycleController)", flush=True)
elif args_cli.solution == "refactored":
    from demo.my_refactored.entry import AlgSolution
    print("[record] Using REFACTORED solution (my_refactored.AlgSolution)", flush=True)
else:
    from demo.my_solution.solution_hybrid import AlgSolution
    print("[record] Using MY solution (AlgSolution)", flush=True)

# Camera eye/lookat for a close-up table view
# Table center is at (1.0, 0.0, 0.826), robot base at (1.404, 0.0, 0.826)
# We want to look from the front-right, slightly above
CAMERA_EYE = (1.0, 1.2, 1.6)
CAMERA_LOOKAT = (1.1, 0.0, 0.85)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )

    if args_cli.solution == "mine":
        action_spec_json = json.dumps(None)
        env_cfg = apply_safe_action_spec(env_cfg, action_spec_json)
    elif args_cli.solution == "refactored":
        action_spec_json = json.dumps(None)
        env_cfg = apply_safe_action_spec(env_cfg, action_spec_json)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Override viewport camera to get close-up view
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "viewport_camera_controller") and unwrapped.viewport_camera_controller is not None:
        unwrapped.viewport_camera_controller.update_view_location(
            eye=CAMERA_EYE, lookat=CAMERA_LOOKAT
        )
        print(f"[record] Camera set: eye={CAMERA_EYE}, lookat={CAMERA_LOOKAT}")

    video_folder = os.path.abspath(os.path.join("logs", "videos", f"TaskE-{args_cli.solution}", "play"))
    video_kwargs = {
        "video_folder": video_folder,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print(f"[record] Video folder: {video_folder}")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()

    # Initialize controller
    if args_cli.solution == "senior":
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "tasks", "task_e")
        controller = DrakeBatchCycleController(base_dir)
    else:
        solution = AlgSolution()

    total_reward = 0.0
    total_time = 0.0

    for step in range(args_cli.video_length):
        with torch.inference_mode():
            # Update camera each frame to keep close-up
            if hasattr(unwrapped, "viewport_camera_controller") and unwrapped.viewport_camera_controller is not None:
                unwrapped.viewport_camera_controller.update_view_location(
                    eye=CAMERA_EYE, lookat=CAMERA_LOOKAT
                )

            if args_cli.solution == "senior":
                resp = controller.act(obs, total_reward)
            else:
                resp = solution.predicts(obs, total_reward)

            giveup = resp.get("giveup", False)
            if giveup:
                break

            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device='cuda').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)

            sim_dt = info["Step_dt"]
            if isinstance(reward, torch.Tensor):
                total_reward += reward.mean().item() / sim_dt
            else:
                total_reward += float(reward) / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]
                total_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)

            if args_cli.debug and step % 100 == 0:
                print(f"  step={step} reward={total_reward:.2f} time={total_time:.2f}", flush=True)

            if terminated.item() or truncated.item():
                break

    env.close()
    print(f"\n[record] DONE: score={total_reward:.2f}, time={total_time:.2f}s", flush=True)
    print(f"[record] Video saved to: {video_folder}/", flush=True)
    return total_reward, total_time


if __name__ == "__main__":
    score, elapsed = main()
    simulation_app.close()
