import argparse
import glob
import json
import os
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm

from buffer import EpisodeBuffer
from config import (
    INITIAL_RECEPTOR_OFFSET,
    INITIAL_RECEPTOR_SIZE,
    ABSTRACT_STATE_DIM,
    BUFFER_CAPACITY,
    DISCOUNT_FACTOR,
    EVAL_EPISODES,
    GLOBAL_SEED,
    GRID_HEIGHT,
    GRID_WIDTH,
    LOOK_BACK,
    MAX_SEARCH_DEPTH,
    MCTS_SIMULATIONS,
    MINIBATCH_SIZE,
    NUM_EPISODES,
    POLICY_DIM,
    RENDER_GAME,
    RUN_VERIFICATION_CHECKS,
    SHOW_PLOTS,
    TRAINING_INTERVAL,
    WEIGHTS_PATH,
)
from game_manager import AbstractStateManager, GameStateManager
from mcts import uMCTS
from neural_networks import MuZeroNet
from representations import get_blank_state, to_code
from rl_manager import ReinforcementLearningManager
from simulator import BitFall

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_config_snapshot():
    """Return a serializable snapshot of the current uppercase config constants."""
    import config as config_module

    snapshot = {}
    for name in dir(config_module):
        if not name.isupper():
            continue
        value = getattr(config_module, name)
        if isinstance(value, (int, float, bool, str, list, dict, tuple)) or value is None:
            snapshot[name] = value
    return snapshot


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def evaluate_actor(gsm, rl_manager, episodes, render=False):
    """Evaluate actor-only mode: Representation + Prediction (no MCTS/Dynamics)."""
    rewards = []
    for ep in range(episodes):
        s = gsm.produce_initial_state()
        state_history = [get_blank_state(GRID_WIDTH, GRID_HEIGHT) for _ in range(LOOK_BACK - 1)] + [s]
        total_reward = 0

        while not gsm.is_final_state():
            phi = state_history[-LOOK_BACK:]
            action = rl_manager.play_as_actor(phi)
            s, reward = gsm.step(action)
            total_reward += reward
            state_history.append(s)
            if render and ep == 0:
                from visualization import visualize_state
                visualize_state(s)

        rewards.append(total_reward)

    rewards_np = np.asarray(rewards, dtype=np.float32)
    return {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(rewards_np)),
        "std_reward": float(np.std(rewards_np)),
        "min_reward": int(np.min(rewards_np)),
        "max_reward": int(np.max(rewards_np)),
        "all_rewards": rewards,
    }


def run_verification_checks():
    """Rigorous checks for simulator logic and MuZero architecture compliance."""
    print("Running comprehensive verification checks...")

    # 1. Configurable initial receptor setup check.
    sim = BitFall(seed=GLOBAL_SEED)
    expected_indices = sorted(
        [((INITIAL_RECEPTOR_OFFSET % GRID_WIDTH) + i) % GRID_WIDTH for i in range(INITIAL_RECEPTOR_SIZE)]
    )
    actual_indices = sorted([idx for seg in sim._get_receptor_segments() for idx in seg])
    assert actual_indices == expected_indices, (
        f"Initial receptor configuration mismatch: expected {expected_indices}, got {actual_indices}"
    )

    # 2. BitFall transition check: debris shifts down exactly one row per step.
    sim = BitFall(seed=GLOBAL_SEED)
    sim.grid[:] = 0
    test_row = GRID_HEIGHT // 2
    test_start = GRID_WIDTH // 3
    test_len = 2
    sim.grid[test_row, test_start:test_start+test_len] = 1 # Put some debris
    before_grid = sim.grid.copy()
    sim.step(0) # Action 0 = Stay
    assert np.array_equal(sim.grid[test_row+1, test_start:test_start+test_len], before_grid[test_row, test_start:test_start+test_len]), "Debris movement sequence failed."

    # 3. BitFall reward checks: Larger, Covers, Excess rule.
    sim = BitFall(seed=GLOBAL_SEED)
    sim.grid[:] = 0
    sim.receptor_offset = GRID_WIDTH // 4
    sim.receptor_size = 3 # Receptor [offset, offset+1, offset+2]
    off = sim.receptor_offset
    sim.grid[-1, off:off+2] = 1 # Debris [off, off+1]
    # Receptor (3) > Debris (2), covers it -> +2
    assert sim._calculate_reward() == 2, f"Reward error: expected +2, got {sim._calculate_reward()}"

    sim.grid[-1, off:off+3] = 1 # Debris [off, off+1, off+2]
    # Equal size -> 0
    assert sim._calculate_reward() == 0, "Reward error: equal size should yield 0 points."

    sim.grid[-1, off-1:off+3] = 1 # Debris [off-1, off, off+1, off+2]
    # Debris (4) > Receptor (3), covers it -> -3
    assert sim._calculate_reward() == -3, f"Reward error: expected -3, got {sim._calculate_reward()}"

    # 4. Split-wrap behavior check (The 'essence' of BitFall complexity).
    sim.receptor_size = 4
    sim.receptor_offset = GRID_WIDTH - 1 # Wrap at the very end
    segments = sim._get_receptor_segments()
    # Indices: [GRID_WIDTH-1, 0, 1, 2]
    assert len(segments) == 2, f"Split-wrap failed: expected 2 segments, got {len(segments)}"
    sizes = sorted([len(s) for s in segments])
    assert sizes == [1, 3], f"Split-wrap sizes mismatch: expected [1, 3], got {sizes}"

    # 5. Neural Network / Tensor shape compliance.
    net = MuZeroNet(seed=GLOBAL_SEED)
    blank = get_blank_state(GRID_WIDTH, GRID_HEIGHT)
    code = to_code([blank for _ in range(LOOK_BACK)])
    sigma = net.representation(net.get_all_params()["repr"], code)
    assert sigma.shape == (ABSTRACT_STATE_DIM,), f"Repr shape {sigma.shape} mismatch."
    
    policy_logits, value = net.prediction(net.get_all_params()["pred"], sigma)
    assert policy_logits.shape == (POLICY_DIM,), "Policy logits shape mismatch."
    
    next_sigma, reward = net.dynamics(net.get_all_params()["dyna"], sigma, 1)
    assert next_sigma.shape == (ABSTRACT_STATE_DIM,), "Dynamics next-state shape mismatch."
    assert np.asarray(reward).shape == (), "Predicted reward must be scalar."

    # 6. EpisodeBuffer target values must preserve real rewards near episode end
    # instead of replacing them with an MCTS bootstrap too early.
    uniform_policy = np.ones(POLICY_DIM, dtype=np.float32) / POLICY_DIM
    dummy_state = np.zeros((2, 2), dtype=int)
    dummy_episode = [
        [dummy_state, 10.0, uniform_policy, 0, 1.0],
        [dummy_state, 20.0, uniform_policy, 1, 2.0],
        [dummy_state, 30.0, uniform_policy, 2, 3.0],
    ]
    test_buffer = EpisodeBuffer(capacity=1)
    assert np.isclose(
        test_buffer._calculate_target_value(dummy_episode, 0),
        1.0 + DISCOUNT_FACTOR * 2.0 + DISCOUNT_FACTOR**2 * 3.0,
    )
    assert np.isclose(test_buffer._calculate_target_value(dummy_episode, 1), 2.0 + DISCOUNT_FACTOR * 3.0)
    assert np.isclose(test_buffer._calculate_target_value(dummy_episode, 2), 3.0)

    print("Verification checks passed: Simulator and Networks are compliant.")


def resolve_weights_path(weights_path):
    """Resolve an evaluation checkpoint from an explicit path or the latest run directory."""
    candidate = os.path.expanduser(weights_path)

    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate

    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    project_candidate = os.path.join(PROJECT_DIR, candidate)
    if os.path.exists(project_candidate):
        return project_candidate

    basename = os.path.basename(candidate)
    run_matches = glob.glob(os.path.join(PROJECT_DIR, "run_*", basename))
    if run_matches:
        return max(run_matches, key=os.path.getmtime)

    # Fallback for the common case where eval is launched with the default
    # basename but checkpoints live only inside run_* folders.
    if weights_path == WEIGHTS_PATH:
        generic_matches = glob.glob(os.path.join(PROJECT_DIR, "run_*", "*.pkl"))
        if generic_matches:
            return max(generic_matches, key=os.path.getmtime)

    raise FileNotFoundError(
        f"Parameter file not found: {weights_path}. "
        "Pass --weights explicitly or place a checkpoint inside a run_* directory."
    )


def save_run_results(run_dir, start_time, rl_manager, net, losses, episode_rewards, eval_stats):
    """Saves weights, metrics, and plots for a training run."""
    os.makedirs(run_dir, exist_ok=True)
    weights_path = os.path.join(run_dir, WEIGHTS_PATH)
    net.save_params(rl_manager.params, weights_path)

    config_path = os.path.join(run_dir, "config_snapshot.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(get_config_snapshot(), f, indent=2, sort_keys=True)

    metrics_path = os.path.join(run_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"MuZero Training Run Metrics Report\nStart: {start_time}\n")
        f.write(f"Actor-Only Mean Reward: {eval_stats['mean_reward']:.2f}\n")
        f.write(f"Max Episode Reward During Training: {max(episode_rewards) if episode_rewards else 0}\n")
        f.write(f"Final Training Loss: {losses[-1] if losses else 0}\n")

    evaluation_path = os.path.join(run_dir, "evaluation_summary.txt")
    with open(evaluation_path, "w", encoding="utf-8") as f:
        f.write("Actor-Only Evaluation Summary\n")
        f.write(f"Episodes: {eval_stats['episodes']}\n")
        f.write(f"Mean Reward: {eval_stats['mean_reward']:.2f}\n")
        f.write(f"Std Reward: {eval_stats['std_reward']:.2f}\n")
        f.write(f"Min Reward: {eval_stats['min_reward']}\n")
        f.write(f"Max Reward: {eval_stats['max_reward']}\n")

    plot_path = os.path.join(run_dir, "training_plot.png")
    from visualization import plot_training_results
    plot_training_results(losses, episode_rewards, save_path=plot_path, show=SHOW_PLOTS)
    print(f"Results saved to {run_dir}")


def run_training():
    """Main training loop for MuZero with robust result persistence."""
    set_global_seeds(GLOBAL_SEED)
    if RUN_VERIFICATION_CHECKS:
        run_verification_checks()

    start_time = datetime.now()
    start_str = start_time.strftime("%Y%m%d_%H%M")
    
    gsm = GameStateManager(seed=GLOBAL_SEED)
    net = MuZeroNet(seed=GLOBAL_SEED)
    rl_manager = ReinforcementLearningManager(net)
    asm = AbstractStateManager(net, rl_manager.params)
    buffer = EpisodeBuffer(capacity=BUFFER_CAPACITY)
    rng = np.random.default_rng(GLOBAL_SEED)

    losses, episode_rewards = [], []

    pbar = tqdm(range(NUM_EPISODES), desc="Training MuZero")
    try:
        for ep in pbar:
            asm.params = rl_manager.params
            s = gsm.produce_initial_state()
            episode_data, total_reward = [], 0
            state_history = [get_blank_state(GRID_WIDTH, GRID_HEIGHT) for _ in range(LOOK_BACK - 1)] + [s]

            while not gsm.is_final_state():
                phi_k = state_history[-LOOK_BACK:]
                sigma_k = asm.get_abstract_state(phi_k)
                mcts = uMCTS(asm, rng=rng)
                root = mcts.search(sigma_k, MCTS_SIMULATIONS, MAX_SEARCH_DEPTH, legal_actions=gsm.get_legal_actions(s))

                pi_k = np.zeros(POLICY_DIM, dtype=np.float32)
                for a, child in root.children.items(): pi_k[a] = child.visit_count
                total_visits = np.sum(pi_k)
                pi_k = pi_k / total_visits if total_visits > 0 else np.ones(POLICY_DIM) / POLICY_DIM

                v_k = float(root.value())
                action = int(rng.choice(np.arange(POLICY_DIM), p=pi_k))
                next_s, reward = gsm.step(action)
                total_reward += reward
                episode_data.append([s, v_k, pi_k, action, reward])
                s, state_history = next_s, state_history + [next_s]

            buffer.save_episode(episode_data)
            episode_rewards.append(total_reward)

            if (ep + 1) % TRAINING_INTERVAL == 0:
                loss = rl_manager.train_step(buffer, MINIBATCH_SIZE)
                if loss is not None:
                    losses.append(loss)
                    pbar.set_postfix({"Loss": f"{loss:.4f}", "Avg Reward": f"{np.mean(episode_rewards[-10:]):.2f}"})

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving available results...")
    finally:
        end_str = datetime.now().strftime("%H%M")
        # Use a fresh simulator manager so the post-training actor evaluation is
        # reproducible and independent from the RNG state consumed during training.
        eval_gsm = GameStateManager(seed=GLOBAL_SEED)
        eval_stats = evaluate_actor(eval_gsm, rl_manager, EVAL_EPISODES, render=False)
        save_run_results(f"run_{start_str}_to_{end_str}", start_time, rl_manager, net, losses, episode_rewards, eval_stats)


def run_eval_only(weights_path, episodes, render=False):
    """Evaluation entrypoint in Pure Actor mode."""
    set_global_seeds(GLOBAL_SEED)
    gsm = GameStateManager(seed=GLOBAL_SEED)
    net = MuZeroNet(seed=GLOBAL_SEED)
    rl_manager = ReinforcementLearningManager(net)
    resolved_weights = resolve_weights_path(weights_path)
    rl_manager.params = net.load_params(resolved_weights)
    stats = evaluate_actor(gsm, rl_manager, episodes, render=render)
    print(f"Eval-Only ({resolved_weights})")
    print(f"Episodes: {stats['episodes']}")
    print(f"Mean Reward: {stats['mean_reward']:.2f}")
    print(f"Std Reward: {stats['std_reward']:.2f}")
    print(f"Min Reward: {stats['min_reward']}")
    print(f"Max Reward: {stats['max_reward']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the MuZero Knockoff agent for BitFall.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="Checkpoint path or checkpoint basename inside run_* directories.")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES, help="Number of evaluation episodes when --mode eval is used.")
    parser.add_argument("--render", action="store_true", default=RENDER_GAME, help="Render the first evaluation episode when --mode eval is used.")
    args = parser.parse_args()
    if args.mode == "train":
        run_training()
    else:
        run_eval_only(args.weights, args.episodes, render=args.render)
