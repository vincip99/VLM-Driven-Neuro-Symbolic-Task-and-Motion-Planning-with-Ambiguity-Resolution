#!/usr/bin/env python
"""
Generate Publication-Quality Training and Evaluation Plots for PPO Manipulation
================================================================================
Generates high-resolution figures into the `images/` and `Docs/pictures/` directories:
1. `ppo_training_results.png` / `.pdf`: 2-panel combined training and eval
2. `ppo_training_curve.png` / `.pdf`: Single panel training return evolution
3. `ppo_eval_curve.png` / `.pdf`: Single panel periodic evaluation curve
4. `ppo_bc_ablation.png` / `.pdf`: Comparison between Pure RL and BC Warm-Start
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(repo_root, "logs", "ppo_training")
output_dirs = [
    os.path.join(repo_root, "Docs", "pictures"),
]
for d in output_dirs:
    os.makedirs(d, exist_ok=True)

# ── Aesthetic Configuration (Matching Presentation Palette) ──────────────────
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

COLOR_DARKBLUE = "#002060"  # matches presentation darkblue
COLOR_RED = "#9B1115"       # matches presentation myred
COLOR_ACCENT = "#2A7B9B"    # teal accent
COLOR_GRAY = "#7f7f7f"


def save_figure(fig, filename_base):
    for d in output_dirs:
        png_path = os.path.join(d, f"{filename_base}.png")
        pdf_path = os.path.join(d, f"{filename_base}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")


def plot_main_training_curves():
    train_csv = os.path.join(log_dir, "train_monitor.csv")
    eval_npz = os.path.join(log_dir, "evaluations.npz")

    df = pd.read_csv(train_csv, skiprows=1)
    df["timesteps"] = df["l"].cumsum()
    evals = np.load(eval_npz)

    # 1. Combined 2-Panel Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)

    # Panel 1: Training Episode Returns
    ax1 = axes[0]
    ax1.scatter(
        df["timesteps"], df["r"],
        color=COLOR_DARKBLUE, alpha=0.22, s=14, edgecolors="none",
        label="Raw Episode Return"
    )
    rolling_w = 20
    roll_mean = df["r"].rolling(window=rolling_w, min_periods=5).mean()
    roll_std = df["r"].rolling(window=rolling_w, min_periods=5).std()

    ax1.plot(
        df["timesteps"], roll_mean,
        color=COLOR_RED, linewidth=2.4,
        label=f"Rolling Mean (w={rolling_w})"
    )
    ax1.fill_between(
        df["timesteps"],
        roll_mean - roll_std,
        roll_mean + roll_std,
        color=COLOR_RED, alpha=0.18,
        label="±1 Std Dev"
    )
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax1.set_xlabel("Environment Timesteps")
    ax1.set_ylabel("Episode Return")
    ax1.set_title("PPO Training Return Evolution")
    ax1.set_xlim(0, 26000)
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)

    # Panel 2: Deterministic Periodic Evaluation
    ax2 = axes[1]
    eval_timesteps = evals["timesteps"]
    eval_returns = evals["results"]
    eval_mean = eval_returns.mean(axis=1)
    eval_std = eval_returns.std(axis=1)

    for step, rets in zip(eval_timesteps, eval_returns):
        ax2.scatter(
            [step] * len(rets), rets,
            color=COLOR_ACCENT, alpha=0.45, s=24, edgecolors="none"
        )

    ax2.plot(
        eval_timesteps, eval_mean,
        color=COLOR_DARKBLUE, marker="o", markersize=7, linewidth=2.4,
        label="Mean Eval Return (10 episodes)"
    )
    ax2.fill_between(
        eval_timesteps,
        eval_mean - eval_std,
        eval_mean + eval_std,
        color=COLOR_DARKBLUE, alpha=0.15,
        label="±1 Std Dev"
    )
    ax2.set_xlabel("Environment Timesteps")
    ax2.set_ylabel("Evaluation Return")
    ax2.set_title("Periodic Policy Evaluation (Greedy)")
    ax2.set_xlim(3000, 27000)
    ax2.set_xticks(eval_timesteps)
    ax2.set_xticklabels([f"{int(s/1000)}k" for s in eval_timesteps])
    ax2.grid(True)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    save_figure(fig, "ppo_training_results")
    plt.close(fig)

    # 2. Standalone Training Return Plot
    fig_train, ax_tr = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    ax_tr.scatter(
        df["timesteps"], df["r"],
        color=COLOR_DARKBLUE, alpha=0.22, s=14, edgecolors="none",
        label="Raw Episode Return"
    )
    ax_tr.plot(
        df["timesteps"], roll_mean,
        color=COLOR_RED, linewidth=2.4,
        label=f"Rolling Mean (w={rolling_w})"
    )
    ax_tr.fill_between(
        df["timesteps"],
        roll_mean - roll_std,
        roll_mean + roll_std,
        color=COLOR_RED, alpha=0.18,
        label="±1 Std Dev"
    )
    ax_tr.set_xlabel("Environment Timesteps")
    ax_tr.set_ylabel("Episode Return")
    ax_tr.set_title("PPO Training Return Evolution")
    ax_tr.set_xlim(0, 26000)
    ax_tr.grid(True)
    ax_tr.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
    plt.tight_layout()
    save_figure(fig_train, "ppo_training_curve")
    plt.close(fig_train)

    # 3. Standalone Evaluation Return Plot
    fig_eval, ax_ev = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    for step, rets in zip(eval_timesteps, eval_returns):
        ax_ev.scatter(
            [step] * len(rets), rets,
            color=COLOR_ACCENT, alpha=0.45, s=24, edgecolors="none"
        )
    ax_ev.plot(
        eval_timesteps, eval_mean,
        color=COLOR_DARKBLUE, marker="o", markersize=7, linewidth=2.4,
        label="Mean Eval Return (10 episodes)"
    )
    ax_ev.fill_between(
        eval_timesteps,
        eval_mean - eval_std,
        eval_mean + eval_std,
        color=COLOR_DARKBLUE, alpha=0.15,
        label="±1 Std Dev"
    )
    ax_ev.set_xlabel("Environment Timesteps")
    ax_ev.set_ylabel("Evaluation Return")
    ax_ev.set_title("Periodic Policy Evaluation (Greedy)")
    ax_ev.set_xlim(3000, 27000)
    ax_ev.set_xticks(eval_timesteps)
    ax_ev.set_xticklabels([f"{int(s/1000)}k" for s in eval_timesteps])
    ax_ev.grid(True)
    ax_ev.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
    plt.tight_layout()
    save_figure(fig_eval, "ppo_eval_curve")
    plt.close(fig_eval)


def plot_bc_vs_pure_rl():
    """Compares PPO_2 (pure RL, 120k steps) with PPO_3 (BC warm-start, 25k steps)."""
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea2 = event_accumulator.EventAccumulator(os.path.join(log_dir, "PPO_2"))
        ea2.Reload()
        r2 = ea2.Scalars("rollout/ep_rew_mean")
        steps2 = np.array([x.step for x in r2])
        vals2 = np.array([x.value for x in r2])

        ea3 = event_accumulator.EventAccumulator(os.path.join(log_dir, "PPO_3"))
        ea3.Reload()
        r3 = ea3.Scalars("rollout/ep_rew_mean")
        steps3 = np.array([x.step for x in r3])
        vals3 = np.array([x.value for x in r3])

        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)

        ax.plot(
            steps2 / 1000.0, vals2,
            color=COLOR_GRAY, linewidth=2.0, linestyle="--",
            label="Pure RL (No BC, 120k steps - Reaching plateau)"
        )
        ax.plot(
            steps3 / 1000.0, vals3,
            color=COLOR_RED, linewidth=2.6,
            label="With BC Warm-Start (100 demos, 25k steps)"
        )

        ax.set_xlabel("Environment Timesteps (x1000)")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title("Ablation: Behavioral Cloning Warm-Start vs Pure RL")
        ax.grid(True)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)

        plt.tight_layout()
        save_figure(fig, "ppo_bc_ablation")
        plt.close(fig)
    except Exception as e:
        print(f"Could not generate BC ablation plot: {e}")


if __name__ == "__main__":
    plot_main_training_curves()
    plot_bc_vs_pure_rl()
