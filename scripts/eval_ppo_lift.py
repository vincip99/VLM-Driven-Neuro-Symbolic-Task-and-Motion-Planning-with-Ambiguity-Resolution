#!/usr/bin/env python
"""
Evaluation Script for Trained PPO Hybrid Pick-Up Policy
======================================================
Evaluates the performance of the trained PPO grasp & lift policy across
different target objects (cubes and cylinders) and measures success rate,
time-to-lift, and physical stability.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Ensure workspace root is in sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.envs.grasp_lift_env import ALL_TARGET_OBJECTS, GraspLiftEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO grasp and lift model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_hybrid_lift.zip",
        help="Path to trained PPO model zip",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20)",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="yellow_cube",
        help="Specific target object to evaluate (default: yellow_cube). Pass 'all' to cycle through all objects.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record rollout frames and save as video in videos/ppo_eval.mp4",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("Evaluating PPO Grasp & Lift Policy")
    print("=" * 70)
    print(f"Model Path:  {args.model_path}")
    print(f"Episodes:    {args.episodes}")
    print(f"Target Name: {args.target_name if args.target_name else 'All Objects (cubes & cans)'}")
    print("=" * 70)

    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint '{args.model_path}' does not exist!")
        sys.exit(1)

    # Load model
    model = PPO.load(args.model_path, device="cpu")

    render_mode = "rgb_array" if args.record_video else None
    env = GraspLiftEnv(target_name=args.target_name, render_mode=render_mode)

    results = []
    recorded_frames = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        target = info["target_name"]
        is_cylinder = info["is_cylinder"]

        ep_reward = 0.0
        ep_steps = 0
        success = False
        terminated = False
        truncated = False

        if args.record_video and ep < 5:
            frame = env.render()
            if frame is not None:
                recorded_frames.append(frame)

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if step_info.get("is_success", False):
                success = True

            if args.record_video and ep < 5:
                frame = env.render()
                if frame is not None:
                    recorded_frames.append(frame)

        results.append({
            "episode": ep + 1,
            "target": target,
            "is_cylinder": is_cylinder,
            "success": success,
            "reward": ep_reward,
            "steps": ep_steps,
            "final_lift": step_info.get("lift_height", 0.0),
        })

        status_str = "SUCCESS" if success else "FAILED"
        print(
            f"Ep {ep+1:02d}/{args.episodes:02d} | Target: {target:12s} "
            f"({'cylinder' if is_cylinder else 'cube    '}) | Status: {status_str:7s} | "
            f"Steps: {ep_steps:2d} | Lift: {step_info.get('lift_height', 0.0)*100:+.1f}cm | Reward: {ep_reward:+6.2f}"
        )

    env.close()

    # Summary metrics
    total_eps = len(results)
    successes = sum(1 for r in results if r["success"])
    cube_eps = [r for r in results if not r["is_cylinder"]]
    can_eps = [r for r in results if r["is_cylinder"]]

    cube_succ = sum(1 for r in cube_eps if r["success"])
    can_succ = sum(1 for r in can_eps if r["success"])

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Overall Success Rate: {successes}/{total_eps} ({successes/total_eps*100:.1f}%)")
    if cube_eps:
        print(f"Cube Success Rate:    {cube_succ}/{len(cube_eps)} ({cube_succ/len(cube_eps)*100:.1f}%)")
    if can_eps:
        print(f"Can Success Rate:     {can_succ}/{len(can_eps)} ({can_succ/len(can_eps)*100:.1f}%)")
    avg_reward = np.mean([r["reward"] for r in results])
    print(f"Average Reward:       {avg_reward:+.2f}")
    print("=" * 70)

    # Save video if requested
    if args.record_video and recorded_frames:
        import imageio
        videos_dir = os.path.join(repo_root, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        video_path = os.path.join(videos_dir, "ppo_eval_rollout.mp4")
        imageio.mimsave(
            video_path,
            recorded_frames,
            fps=20,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=["-movflags", "+faststart"],
        )
        print(f"Recorded evaluation video (upright unmirrored) saved to: {video_path}")


if __name__ == "__main__":
    main()
