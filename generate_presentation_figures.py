#!/usr/bin/env python3
"""Generate presentation-ready MuZero result figures without retraining."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "presentation_figures"


def portable_path(path: Path) -> str:
    """Reference files by repository-relative path, never a personal absolute path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return resolved.name
DEFAULT_TRAINING_LOG_NAME = "training_history.json"
DEFAULT_WEIGHTS_NAME = "muzero_weights_optimized.pkl"
MOVING_AVERAGE_WINDOW = 50
BENCHMARK_EPISODES = 1000

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/private/tmp")) / "project2it3105-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def find_latest_run_dir(project_dir: Path) -> Path:
    run_dirs = [Path(path) for path in glob.glob(str(project_dir / "run_*")) if Path(path).is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directory found under {project_dir}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def trailing_moving_average(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    if not values:
        return np.asarray([], dtype=np.float32)

    arr = np.asarray(values, dtype=np.float32)
    cumulative = np.cumsum(arr, dtype=np.float64)
    averages = np.empty_like(arr, dtype=np.float32)

    for idx in range(len(arr)):
        start = max(0, idx - window + 1)
        window_sum = cumulative[idx] - (cumulative[start - 1] if start > 0 else 0.0)
        averages[idx] = window_sum / (idx - start + 1)

    return averages


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_training_history(training_log_path: Path) -> tuple[list[float], list[float]]:
    with training_log_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    try:
        losses = payload["losses"]
        episode_rewards = payload["episode_rewards"]
    except KeyError as exc:
        raise KeyError(
            f"Training log {training_log_path} is missing the required key {exc!s}"
        ) from exc

    return losses, episode_rewards


def create_training_dynamics_figure(
    losses: list[float], episode_rewards: list[float], output_path: Path
) -> None:
    loss_ma = trailing_moving_average(losses, MOVING_AVERAGE_WINDOW)
    reward_ma = trailing_moving_average(episode_rewards, MOVING_AVERAGE_WINDOW)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_loss, ax_reward) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    ax_loss.plot(
        range(len(losses)),
        losses,
        color="#9ecae1",
        linewidth=1.3,
        alpha=0.55,
        label="Raw Loss",
    )
    ax_loss.plot(
        range(len(loss_ma)),
        loss_ma,
        color="#08519c",
        linewidth=2.6,
        label=f"Moving Average ({MOVING_AVERAGE_WINDOW})",
    )
    ax_loss.set_title("MuZero Training Loss", fontsize=17, pad=12)
    ax_loss.set_xlabel("Training Step", fontsize=13)
    ax_loss.set_ylabel("Loss", fontsize=13)
    ax_loss.tick_params(labelsize=11)
    ax_loss.legend(fontsize=11)

    ax_reward.plot(
        range(len(episode_rewards)),
        episode_rewards,
        color="#fdae6b",
        linewidth=1.2,
        alpha=0.45,
        label="Raw Episode Reward",
    )
    ax_reward.plot(
        range(len(reward_ma)),
        reward_ma,
        color="#e6550d",
        linewidth=2.6,
        label=f"Moving Average ({MOVING_AVERAGE_WINDOW})",
    )
    ax_reward.axhline(0.0, color="#222222", linewidth=1.4, linestyle="--", label="Reward Baseline")
    ax_reward.set_title("Training Reward per Episode", fontsize=17, pad=12)
    ax_reward.set_xlabel("Episode", fontsize=13)
    ax_reward.set_ylabel("Total Reward", fontsize=13)
    ax_reward.tick_params(labelsize=11)
    ax_reward.legend(fontsize=11)

    save_figure(fig, output_path)


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def parse_training_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as file:
        text = file.read()

    max_reward_match = re.search(r"Max Episode Reward:\s*([-\d.]+)", text)
    loss_match = re.search(r"Loss:\s*([-\d.]+)", text)
    if not max_reward_match or not loss_match:
        raise ValueError(f"Could not parse training summary metrics from {summary_path}")

    return {
        "max_reward": int(round(float(max_reward_match.group(1)))),
        "final_loss": float(loss_match.group(1)),
    }


def get_benchmark_path(run_dir: Path, weights_path: Path, episodes: int) -> Path:
    return run_dir / f"benchmark_{weights_path.stem}_{episodes}eps.json"


def evaluate_actor_only(weights_path: Path, episodes: int, benchmark_path: Path) -> dict:
    from config import GLOBAL_SEED
    from game_manager import GameStateManager
    from main import evaluate_actor, get_config_snapshot, set_global_seeds
    from neural_networks import MuZeroNet
    from rl_manager import ReinforcementLearningManager

    set_global_seeds(GLOBAL_SEED)

    gsm = GameStateManager(seed=GLOBAL_SEED)
    net = MuZeroNet(seed=GLOBAL_SEED)
    rl_manager = ReinforcementLearningManager(net)
    rl_manager.params = net.load_params(str(weights_path))

    stats = evaluate_actor(gsm, rl_manager, episodes, render=False)
    result = {
        "timestamp": datetime.now().isoformat(),
        "weights_path": portable_path(weights_path),
        "episodes": int(episodes),
        "render": False,
        "global_seed": int(GLOBAL_SEED),
        "config_snapshot": get_config_snapshot(),
        **stats,
    }
    save_json(result, benchmark_path)
    return result


def load_or_run_actor_benchmark(run_dir: Path, weights_path: Path, episodes: int) -> tuple[dict, Path, bool]:
    benchmark_path = get_benchmark_path(run_dir, weights_path, episodes)
    if benchmark_path.exists():
        payload = load_json(benchmark_path)
        rewards = payload.get("all_rewards", [])
        if payload.get("episodes") == episodes and len(rewards) == episodes:
            return payload, benchmark_path, True

    payload = evaluate_actor_only(weights_path, episodes, benchmark_path)
    return payload, benchmark_path, False


def summarize_rewards(rewards: list[float] | np.ndarray) -> dict:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    return {
        "episodes": int(rewards_np.size),
        "mean": float(np.mean(rewards_np)),
        "median": float(np.median(rewards_np)),
        "std": float(np.std(rewards_np)),
        "min": float(np.min(rewards_np)),
        "max": float(np.max(rewards_np)),
        "positive_episodes": int(np.sum(rewards_np > 0)),
        "non_negative_episodes": int(np.sum(rewards_np >= 0)),
    }


def create_actor_distribution_figure(
    rewards: list[float], summary: dict, output_path: Path, compact: bool = False
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    if compact:
        fig, ax = plt.subplots(figsize=(11.0, 5.5), constrained_layout=False)
    else:
        fig, ax = plt.subplots(figsize=(12, 6.75), constrained_layout=True)

    bins = min(40, max(18, int(np.sqrt(len(rewards)))))
    ax.hist(rewards, bins=bins, color="#3182bd", edgecolor="white", alpha=0.88)
    ax.axvline(0.0, color="#222222", linewidth=1.6, linestyle="--", label="Reward = 0")
    ax.axvline(summary["mean"], color="#cb181d", linewidth=2.2, label=f"Mean = {summary['mean']:.2f}")

    title_fontsize = 14 if compact else 18
    axis_label_fontsize = 16 if compact else 13
    tick_fontsize = 12 if compact else 11
    legend_fontsize = 11
    stats_fontsize = 10 if compact else 11

    ax.set_title("Actor-Only Reward Distribution (1000 Episodes)", fontsize=title_fontsize, pad=7 if compact else 12)
    ax.set_xlabel("Total Reward per Episode", fontsize=axis_label_fontsize)
    ax.set_ylabel("Episode Count", fontsize=axis_label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, loc="upper left", frameon=True, borderpad=0.4, handlelength=2.0)

    stats_text = "\n".join(
        [
            f"Mean: {summary['mean']:.2f}",
            f"Median: {summary['median']:.2f}",
            f"Std: {summary['std']:.2f}",
            f"Min: {summary['min']:.0f}",
            f"Max: {summary['max']:.0f}",
            f"Positive: {summary['positive_episodes']}",
            f"Non-negative: {summary['non_negative_episodes']}",
        ]
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=stats_fontsize,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d9d9d9", "alpha": 0.95},
    )

    if compact:
        fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.14)

    save_figure(fig, output_path)


def print_summary(summary: dict) -> None:
    print("Actor-only evaluation summary")
    print(f"  Mean: {summary['mean']:.4f}")
    print(f"  Median: {summary['median']:.4f}")
    print(f"  Std: {summary['std']:.4f}")
    print(f"  Min: {summary['min']:.0f}")
    print(f"  Max: {summary['max']:.0f}")
    print(f"  Positive episodes: {summary['positive_episodes']}")
    print(f"  Non-negative episodes: {summary['non_negative_episodes']}")


def create_final_results_slide(
    histogram_path: Path,
    training_plot_path: Path,
    output_path: Path,
    actor_summary: dict,
    training_summary: dict,
) -> None:
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")

    title_color = "#16324f"
    accent_color = "#2a6f97"
    light_panel = "#f5f8fb"
    border_color = "#d8e1ea"
    note_bg = "#eef5fb"
    body_color = "#22313f"

    fig.text(0.05, 0.94, "Final Results", fontsize=26, fontweight="bold", color=title_color)
    fig.text(
        0.05,
        0.9,
        "Training: 1500 episodes | Final benchmark: actor-only over 1000 episodes",
        fontsize=14,
        color=accent_color,
        fontweight="semibold",
    )
    fig.lines.append(
        plt.Line2D([0.05, 0.95], [0.885, 0.885], transform=fig.transFigure, color=border_color, linewidth=1.4)
    )

    histogram_ax = fig.add_axes([0.05, 0.26, 0.6, 0.57])
    histogram_ax.imshow(plt.imread(histogram_path))
    histogram_ax.set_axis_off()

    metrics_ax = fig.add_axes([0.685, 0.34, 0.27, 0.49])
    metrics_ax.set_facecolor(light_panel)
    for spine in metrics_ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.2)
    metrics_ax.set_xticks([])
    metrics_ax.set_yticks([])
    metrics_ax.text(0.06, 0.93, "Key Results", fontsize=17, fontweight="bold", color=title_color, va="top")
    metrics_lines = [
        f"Mean reward: {actor_summary['mean']:.2f}",
        f"Median: {actor_summary['median']:.0f}",
        f"Std: {actor_summary['std']:.2f}",
        f"Min reward: {actor_summary['min']:.0f}",
        f"Max reward: {actor_summary['max']:.0f}",
        f"Positive episodes: {actor_summary['positive_episodes']} / {actor_summary['episodes']}",
        f"Non-negative episodes: {actor_summary['non_negative_episodes']} / {actor_summary['episodes']}",
        f"Training: max reward {training_summary['max_reward']}",
        f"Final loss ≈ {training_summary['final_loss']:.2f}",
    ]
    metrics_ax.text(
        0.06,
        0.84,
        "\n".join(metrics_lines),
        fontsize=13.2,
        color=body_color,
        va="top",
        linespacing=1.52,
    )

    interpretation_ax = fig.add_axes([0.685, 0.19, 0.27, 0.11])
    interpretation_ax.set_facecolor(note_bg)
    for spine in interpretation_ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.2)
    interpretation_ax.set_xticks([])
    interpretation_ax.set_yticks([])
    interpretation_ax.text(0.06, 0.8, "Interpretation", fontsize=14.5, fontweight="bold", color=title_color, va="top")
    interpretation_ax.text(
        0.06,
        0.48,
        "Mostly positive, fairly stable, but high variance —\nnot solved, consistently above zero.",
        fontsize=12.5,
        color=body_color,
        va="top",
        linespacing=1.3,
    )

    training_ax = fig.add_axes([0.05, 0.06, 0.34, 0.13])
    training_ax.imshow(plt.imread(training_plot_path))
    training_ax.set_axis_off()

    training_note_ax = fig.add_axes([0.41, 0.06, 0.24, 0.13])
    training_note_ax.set_facecolor("white")
    for spine in training_note_ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.0)
    training_note_ax.set_xticks([])
    training_note_ax.set_yticks([])
    training_note_ax.text(0.06, 0.72, "Training evidence", fontsize=13.5, fontweight="bold", color=title_color, va="top")
    training_note_ax.text(
        0.06,
        0.42,
        "Noisy rewards, bounded loss around 13–14.",
        fontsize=12.2,
        color=body_color,
        va="top",
    )

    fig.text(0.05, 0.835, "Main evidence: actor-only reward distribution over 1000 episodes", fontsize=12.5, color=body_color)

    save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation-ready MuZero result figures.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing the checkpoint and saved evaluation artifacts.")
    parser.add_argument("--weights", default=None, help="Optional checkpoint path. Defaults to muzero_weights_optimized.pkl inside the run directory.")
    parser.add_argument("--training-log", default=None, help="Optional raw training-history JSON path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for PNG and JSON artifacts.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else find_latest_run_dir(PROJECT_DIR).resolve()
    weights_path = Path(args.weights).resolve() if args.weights else (run_dir / DEFAULT_WEIGHTS_NAME).resolve()
    training_log_path = Path(args.training_log).resolve() if args.training_log else (run_dir / DEFAULT_TRAINING_LOG_NAME).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    training_figure_path = output_dir / "training_dynamics.png"
    distribution_figure_path = output_dir / "actor_only_reward_distribution.png"
    compact_distribution_figure_path = output_dir / "actor_only_reward_distribution_compact.png"
    final_results_slide_path = output_dir / "final_results_slide.png"
    summary_json_path = output_dir / "evaluation_summary.json"
    training_summary_path = run_dir / "historical_training_summary_with_50ep_actor_eval.txt"

    training_status = {
        "status": "missing_log",
        "missing_log_file": portable_path(training_log_path),
        "output_path": portable_path(training_figure_path),
    }
    if training_log_path.exists():
        losses, episode_rewards = load_training_history(training_log_path)
        create_training_dynamics_figure(losses, episode_rewards, training_figure_path)
        training_status = {
            "status": "created",
            "source_log_file": portable_path(training_log_path),
            "loss_points": int(len(losses)),
            "reward_points": int(len(episode_rewards)),
            "moving_average_window": MOVING_AVERAGE_WINDOW,
            "output_path": portable_path(training_figure_path),
        }
        print(f"Saved training dynamics figure to {training_figure_path}")
    else:
        print(f"Missing training log file: {training_log_path}")

    benchmark_payload, benchmark_path, reused_existing = load_or_run_actor_benchmark(
        run_dir, weights_path, BENCHMARK_EPISODES
    )
    rewards = benchmark_payload["all_rewards"]
    actor_summary = summarize_rewards(rewards)
    create_actor_distribution_figure(rewards, actor_summary, distribution_figure_path)
    create_actor_distribution_figure(rewards, actor_summary, compact_distribution_figure_path, compact=True)
    training_summary_metrics = parse_training_summary(training_summary_path)
    create_final_results_slide(
        compact_distribution_figure_path,
        run_dir / "training_plot.png",
        final_results_slide_path,
        actor_summary,
        training_summary_metrics,
    )
    print_summary(actor_summary)
    print(f"Saved actor-only reward distribution to {distribution_figure_path}")
    print(f"Saved compact actor-only reward distribution to {compact_distribution_figure_path}")
    print(f"Saved final results slide to {final_results_slide_path}")

    evaluation_summary = {
        "run_dir": portable_path(run_dir),
        "weights_path": portable_path(weights_path),
        "actor_only_evaluation": {
            "benchmark_json": portable_path(benchmark_path),
            "used_existing_1000_episode_results": reused_existing,
            **actor_summary,
        },
        "artifacts": {
            "actor_only_reward_distribution_png": portable_path(distribution_figure_path),
            "actor_only_reward_distribution_compact_png": portable_path(compact_distribution_figure_path),
            "final_results_slide_png": portable_path(final_results_slide_path),
            "training_dynamics_png": portable_path(training_figure_path) if training_status["status"] == "created" else None,
        },
        "training_dynamics": training_status,
    }
    save_json(evaluation_summary, summary_json_path)
    print(f"Saved evaluation summary to {summary_json_path}")


if __name__ == "__main__":
    main()
