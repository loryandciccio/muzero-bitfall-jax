import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from config import EVAL_EPISODES, GLOBAL_SEED, RENDER_GAME, WEIGHTS_PATH
from game_manager import GameStateManager
from main import evaluate_actor, get_config_snapshot, resolve_weights_path, set_global_seeds
from neural_networks import MuZeroNet
from rl_manager import ReinforcementLearningManager


def portable_weights_path(resolved_weights_path):
    """Keep personal filesystem locations out of persisted benchmark metadata."""
    resolved = Path(resolved_weights_path).resolve()
    try:
        return resolved.relative_to(Path(__file__).resolve().parent).as_posix()
    except ValueError:
        return resolved.name


def default_output_path(resolved_weights_path, episodes):
    run_dir = os.path.dirname(resolved_weights_path) or "."
    weights_stem = os.path.splitext(os.path.basename(resolved_weights_path))[0]
    return os.path.join(run_dir, f"benchmark_{weights_stem}_{episodes}eps.json")


def run_benchmark(weights_path, episodes, render=False, output_path=None):
    set_global_seeds(GLOBAL_SEED)

    resolved_weights = resolve_weights_path(weights_path)
    gsm = GameStateManager(seed=GLOBAL_SEED)
    net = MuZeroNet(seed=GLOBAL_SEED)
    rl_manager = ReinforcementLearningManager(net)
    rl_manager.params = net.load_params(resolved_weights)

    stats = evaluate_actor(gsm, rl_manager, episodes, render=render)
    result = {
        "timestamp": datetime.now().isoformat(),
        "weights_path": portable_weights_path(resolved_weights),
        "episodes": int(episodes),
        "render": bool(render),
        "global_seed": int(GLOBAL_SEED),
        "config_snapshot": get_config_snapshot(),
        **stats,
    }

    final_output_path = output_path or default_output_path(resolved_weights, episodes)
    output_dir = os.path.dirname(final_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"Benchmark complete for {resolved_weights}")
    print(f"Episodes: {result['episodes']}")
    print(f"Mean Reward: {result['mean_reward']:.2f}")
    print(f"Std Reward: {result['std_reward']:.2f}")
    print(f"Min Reward: {result['min_reward']}")
    print(f"Max Reward: {result['max_reward']}")
    print(f"Saved benchmark summary to {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actor-only benchmark evaluation for a trained MuZero Knockoff checkpoint.")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="Checkpoint path or checkpoint basename inside run_* directories.")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES, help="Number of actor-only evaluation episodes.")
    parser.add_argument("--output", default=None, help="Optional JSON output path for the benchmark summary.")
    parser.add_argument("--render", action="store_true", default=RENDER_GAME, help="Render the first evaluation episode.")
    args = parser.parse_args()

    run_benchmark(args.weights, args.episodes, render=args.render, output_path=args.output)
