#!/usr/bin/env python
"""
Comparative Benchmark: Heuristic Pick vs PPO Pick Approach (Cubes Evaluation)
=============================================================================
Runs a controlled comparison between:
1. Analytical Heuristic State Machine Controller (`get_expert_action()`)
2. Learned Closed-Loop PPO Policy (`model.predict()`)

Focuses strictly on the cube objects (Yellow Cube and Purple Cube) for which
the PPO policy was trained and specialized.

Measures:
- Task Success Rate (%) across yellow and purple cubes
- Execution Speed / Latency (mean steps to successful lift)
- Episode Cumulative Return (efficiency and stability)

Generates:
- `images/heuristic_vs_ppo.png` and `.pdf`
- `Docs/pictures/heuristic_vs_ppo.png` and `.pdf`
- Clean quantitative comparison summary for presentation
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.envs.grasp_lift_env import GraspLiftEnv

CUBE_TARGET_OBJECTS = [
    "yellow_cube",
    "purple_cube",
]

output_dirs = [
    os.path.join(repo_root, "Docs", "pictures"),
]
for d in output_dirs:
    os.makedirs(d, exist_ok=True)

# Styling palette matching presentation
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

COLOR_HEURISTIC = "#7f7f7f"  # Gray for heuristic baseline
COLOR_PPO = "#9B1115"        # myred for PPO
COLOR_BLUE = "#002060"       # darkblue


def evaluate_method(method_name: str, model, episodes_per_object: int = 20, seed_offset: int = 100):
    env = GraspLiftEnv(target_name=None)
    records = []

    for obj in CUBE_TARGET_OBJECTS:
        for ep in range(episodes_per_object):
            seed = seed_offset + ep * 7
            obs, info = env.reset(seed=seed, options={"target_name": obj})
            done = False
            steps = 0
            total_rew = 0.0
            success = False

            while not done and steps < env.max_steps:
                steps += 1
                if method_name == "Heuristic":
                    action = env.get_expert_action()
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, r, term, trunc, step_info = env.step(action)
                total_rew += r
                if step_info.get("is_success", False):
                    success = True
                done = term or trunc

            records.append({
                "method": method_name,
                "target": obj,
                "display_name": "Yellow Cube" if "yellow" in obj else "Purple Cube",
                "success": success,
                "steps": steps,
                "reward": total_rew,
                "final_lift": step_info.get("lift_height", 0.0),
            })

    env.close()
    return pd.DataFrame(records)


def run_benchmark():
    print("=" * 70)
    print("Benchmarking: Heuristic Pick vs PPO Pick Approach (Cubes)")
    print("=" * 70)

    model_path = os.path.join(repo_root, "models", "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(repo_root, "models", "ppo_hybrid_lift.zip")
    print(f"Loading PPO Model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    episodes_per_obj = 20  # 2 cube types * 20 episodes = 40 episodes per method
    print(f"Running {episodes_per_obj} episodes per cube ({episodes_per_obj * len(CUBE_TARGET_OBJECTS)} total per method)...")

    print("\nEvaluating Heuristic State Machine Controller on Cubes...")
    df_heur = evaluate_method("Heuristic", model=None, episodes_per_object=episodes_per_obj, seed_offset=100)

    print("Evaluating PPO Closed-Loop Policy on Cubes...")
    df_ppo = evaluate_method("PPO", model=model, episodes_per_object=episodes_per_obj, seed_offset=100)

    df = pd.concat([df_heur, df_ppo], ignore_index=True)

    # ── Print Quantitative Comparison Table ───────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'Metric':<30} | {'Heuristic Controller':<18} | {'PPO Policy':<18}")
    print("-" * 75)

    succ_heur = df_heur["success"].mean() * 100
    succ_ppo = df_ppo["success"].mean() * 100
    print(f"{'Overall Cube Success Rate (%)':<30} | {succ_heur:17.1f}% | {succ_ppo:17.1f}%")

    yellow_heur = df_heur[df_heur["target"] == "yellow_cube"]["success"].mean() * 100
    yellow_ppo = df_ppo[df_ppo["target"] == "yellow_cube"]["success"].mean() * 100
    print(f"{'Yellow Cube Success (%)':<30} | {yellow_heur:17.1f}% | {yellow_ppo:17.1f}%")

    purple_heur = df_heur[df_heur["target"] == "purple_cube"]["success"].mean() * 100
    purple_ppo = df_ppo[df_ppo["target"] == "purple_cube"]["success"].mean() * 100
    print(f"{'Purple Cube Success (%)':<30} | {purple_heur:17.1f}% | {purple_ppo:17.1f}%")

    steps_heur = df_heur[df_heur["success"]]["steps"].mean()
    steps_ppo = df_ppo[df_ppo["success"]]["steps"].mean()
    print(f"{'Mean Steps to Lift (Successes)':<30} | {steps_heur:18.1f} | {steps_ppo:18.1f}")

    time_heur = steps_heur / 20.0  # control freq = 20 Hz
    time_ppo = steps_ppo / 20.0
    print(f"{'Mean Physical Time (s)':<30} | {time_heur:17.2f}s | {time_ppo:17.2f}s")

    rew_heur = df_heur["reward"].mean()
    rew_ppo = df_ppo["reward"].mean()
    print(f"{'Mean Episode Return':<30} | {rew_heur:18.1f} | {rew_ppo:18.1f}")
    print("=" * 75)

    # ── Generate Presentation Plot ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300)

    # 1. Success Rate Comparison
    ax1 = axes[0]
    categories = ["Cube Pick Success"]
    heur_vals = [succ_heur]
    ppo_vals = [succ_ppo]
    x = np.arange(len(categories))
    width = 0.28

    rects1 = ax1.bar(x - width/2, heur_vals, width, label="Heuristic", color=COLOR_HEURISTIC, edgecolor="black", alpha=0.85)
    rects2 = ax1.bar(x + width/2, ppo_vals, width, label="PPO Policy", color=COLOR_PPO, edgecolor="black", alpha=0.9)

    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Cube Grasp & Lift Success Rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right", fontsize=9.5)

    for r in rects1:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 2, f"{h:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#333333")
    for r in rects2:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2., h + 2, f"{h:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=COLOR_PPO)

    # 2. Execution Latency (Steps to Successful Lift)
    ax2 = axes[1]
    succ_data = [df_heur[df_heur["success"]]["steps"].values, df_ppo[df_ppo["success"]]["steps"].values]
    box = ax2.boxplot(
        succ_data,
        labels=["Heuristic", "PPO Policy"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2.0)
    )
    box["boxes"][0].set_facecolor(COLOR_HEURISTIC)
    box["boxes"][0].set_alpha(0.7)
    box["boxes"][1].set_facecolor(COLOR_PPO)
    box["boxes"][1].set_alpha(0.85)

    ax2.set_ylabel("Execution Steps (20 Hz)")
    ax2.set_title("Execution Speed (Steps to Lift)")
    ax2.grid(True, axis="y")

    # 3. Episode Return Distribution
    ax3 = axes[2]
    ret_data = [df_heur["reward"].values, df_ppo["reward"].values]
    box3 = ax3.boxplot(
        ret_data,
        labels=["Heuristic", "PPO Policy"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2.0)
    )
    box3["boxes"][0].set_facecolor(COLOR_HEURISTIC)
    box3["boxes"][0].set_alpha(0.7)
    box3["boxes"][1].set_facecolor(COLOR_PPO)
    box3["boxes"][1].set_alpha(0.85)

    ax3.set_ylabel("Cumulative Episode Return")
    ax3.set_title("Reward & Contact Quality")
    ax3.grid(True, axis="y")

    plt.tight_layout()

    for d in output_dirs:
        png_p = os.path.join(d, "heuristic_vs_ppo.png")
        pdf_p = os.path.join(d, "heuristic_vs_ppo.pdf")
        fig.savefig(png_p, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_p, bbox_inches="tight")
        print(f"Saved: {png_p}")
        print(f"Saved: {pdf_p}")

    plt.close(fig)
    print("\nCube benchmark completed successfully!")


if __name__ == "__main__":
    run_benchmark()
