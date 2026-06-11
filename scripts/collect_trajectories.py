"""Collect expert trajectories from senior's controller for Task E.

Runs DrakeBatchCycleController for N episodes, saves only perfect-score (18.0) ones.
Each saved episode contains:
    - key: 6-dim vector [mustard_x, mustard_y, sugar_x, sugar_y, banana_x, banana_y]
    - actions: (T, 8) full action sequence

IMPORTANT: Task E's env has no reset events for objects/robot — env.reset() does NOT
restore positions. We spawn a separate subprocess per episode for clean randomization
(same pattern as record_task_e.py, which runs 1 episode per process).

Usage (launcher mode — spawns subprocesses):
    PYTHONPATH="$(pwd):$(pwd)/scripts:$PYTHONPATH" python scripts/collect_trajectories.py \
        --num_episodes 100 --output_dir demo/my_solution/data/trajectories

Usage (single-episode worker — called automatically by launcher):
    PYTHONPATH="$(pwd):$(pwd)/scripts:$PYTHONPATH" python scripts/collect_trajectories.py \
        --worker --ep_idx 0 --output_dir demo/my_solution/data/trajectories
"""
import argparse
import os
import sys
import subprocess


def run_launcher():
    """Spawn one subprocess per episode to collect trajectories."""
    parser = argparse.ArgumentParser(description="Collect expert trajectories for Task E.")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="demo/my_solution/data/trajectories")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--perfect_score", type=float, default=18.0)
    parser.add_argument("--score_tol", type=float, default=0.1)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(args.output_dir) if f.endswith('.npz')])
    saved_count = existing

    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(script_path))
    env_vars = os.environ.copy()
    pythonpath = f"{project_root}:{os.path.join(project_root, 'scripts')}"
    if "PYTHONPATH" in env_vars:
        pythonpath += ":" + env_vars["PYTHONPATH"]
    env_vars["PYTHONPATH"] = pythonpath

    print(f"[collect] Starting collection: {args.num_episodes} episodes, "
          f"existing={existing}, output_dir={args.output_dir}")

    for ep_idx in range(args.num_episodes):
        cmd = [
            sys.executable, script_path,
            "--worker",
            "--ep_idx", str(ep_idx),
            "--save_idx", str(saved_count),
            "--output_dir", args.output_dir,
            "--max_steps", str(args.max_steps),
            "--perfect_score", str(args.perfect_score),
            "--score_tol", str(args.score_tol),
            "--headless",
            "--enable_cameras",
        ]
        result = subprocess.run(cmd, env=env_vars, capture_output=True, text=True)

        # Print worker stdout (contains score/key info)
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if line.startswith("[collect]"):
                    print(line)

        if result.returncode != 0:
            # Print last few lines of stderr for debugging
            stderr_lines = result.stderr.strip().split('\n')
            err_tail = '\n'.join(stderr_lines[-5:]) if stderr_lines else ""
            print(f"[collect] ep={ep_idx} WORKER FAILED (exit={result.returncode})")
            if err_tail:
                print(f"  stderr: {err_tail}")
            continue

        # Check if worker saved a file
        expected_file = os.path.join(args.output_dir, f"ep_{saved_count:04d}.npz")
        if os.path.exists(expected_file):
            saved_count += 1

    total_new = saved_count - existing
    print(f"\n[collect] Done: {total_new} new trajectories saved "
          f"({saved_count} total), {args.num_episodes} attempted")


def run_worker():
    """Run a single episode in Isaac Lab and optionally save the trajectory."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ep_idx", type=int, required=True)
    parser.add_argument("--save_idx", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--perfect_score", type=float, default=18.0)
    parser.add_argument("--score_tol", type=float, default=0.1)
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    args_cli.task = "ATEC-TaskE-Piper"
    args_cli.headless = True
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    import numpy as np

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    import atec_rl_lab.tasks
    from isaaclab_tasks.utils import parse_env_cfg

    from demo.tasks.task_e.grasp_drake_batch_cycle import DrakeBatchCycleController
    from demo.my_solution.perception import Perception

    # Create env
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "demo", "tasks", "task_e"
    )
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "demo", "my_solution", "data"
    )
    camera_pose = np.load(os.path.join(data_dir, "camera_pose.npy"), allow_pickle=True).item()
    perception = Perception(camera_pose)

    controller = DrakeBatchCycleController(base_dir)
    obs, _ = env.reset()

    total_reward = 0.0
    actions_list = []
    object_key = None
    prev_needs_obs = True
    ep_idx = args_cli.ep_idx

    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            resp = controller.act(obs, total_reward)
            if resp.get("giveup", False):
                break

            action_raw = resp["action"]
            actions_list.append(np.array(action_raw, dtype=np.float32).flatten())

            if object_key is None and prev_needs_obs and not controller.needs_initial_observation:
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
                depth = depth.astype(np.float32)

                positions = perception.locate_all_objects(depth)
                key_order = ("mustard", "sugar", "banana")
                key_parts = []
                valid = True
                for obj_name in key_order:
                    pos = positions.get(obj_name)
                    if pos is None:
                        valid = False
                        break
                    key_parts.extend([float(pos[0]), float(pos[1])])
                if valid:
                    object_key = np.array(key_parts, dtype=np.float32)
                    print(f"[collect] ep={ep_idx} step={step} key extracted: {object_key.round(3)}")
                else:
                    print(f"[collect] ep={ep_idx} step={step} key extraction FAILED")
            prev_needs_obs = controller.needs_initial_observation

            actions_tensor = torch.tensor(action_raw, dtype=torch.float32, device='cuda').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions_tensor)

            sim_dt = info["Step_dt"]
            if isinstance(reward, torch.Tensor):
                total_reward += reward.mean().item() / sim_dt
            else:
                total_reward += float(reward) / sim_dt

            if terminated.item() or truncated.item():
                break

    env.close()

    num_steps = len(actions_list)
    is_perfect = abs(total_reward - args_cli.perfect_score) < args_cli.score_tol

    if is_perfect and object_key is not None and num_steps > 0:
        actions_array = np.stack(actions_list, axis=0)
        filename = f"ep_{args_cli.save_idx:04d}.npz"
        np.savez(
            os.path.join(args_cli.output_dir, filename),
            key=object_key,
            actions=actions_array,
        )
        print(f"[collect] ep={ep_idx} score={total_reward:.2f} SAVED as {filename} "
              f"(steps={num_steps}, key={object_key.round(3)})")
    else:
        reason = "score" if not is_perfect else ("no_key" if object_key is None else "no_actions")
        print(f"[collect] ep={ep_idx} score={total_reward:.2f} steps={num_steps} SKIP ({reason})")

    simulation_app.close()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker()
    else:
        run_launcher()
