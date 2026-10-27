import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    GLOBAL_SEED,
    GRID_HEIGHT,
    GRID_WIDTH,
    LOOK_BACK,
    MAX_EPISODE_STEPS,
    MAX_SEARCH_DEPTH,
    MCTS_SIMULATIONS,
    POLICY_DIM,
    WEIGHTS_PATH,
)
from game_manager import AbstractStateManager, GameStateManager
from main import resolve_weights_path, set_global_seeds
from mcts import uMCTS
from neural_networks import MuZeroNet
from representations import get_blank_state
from rl_manager import ReinforcementLearningManager


CELL_SIZE = 24
INFO_BAR_HEIGHT = 56
REWARD_HISTORY_HEIGHT = 24
GRID_LINE_WIDTH = 1
CHANGE_HIGHLIGHT_WIDTH = 3
RECEPTOR_OUTLINE_WIDTH = 3
REWARD_HOLD_FRAMES = 4

COLOR_EMPTY = (25, 28, 36)
COLOR_DEBRIS = (230, 179, 0)
COLOR_RECEPTOR = (70, 155, 255)
COLOR_GRID = (55, 60, 70)
COLOR_TEXT = (240, 240, 240)
COLOR_BG = (15, 18, 24)
COLOR_CHANGE = (255, 255, 255)
COLOR_REWARD_POS = (88, 214, 141)
COLOR_REWARD_NEG = (255, 111, 111)

ACTION_NAMES = {
    0: "Stay",
    1: "Left",
    2: "Right",
}


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def default_output_path(resolved_weights_path):
    run_dir = os.path.dirname(resolved_weights_path) or "."
    return os.path.join(run_dir, "bitfall_eval_actor.gif")


def simulate_eval_episode(weights_path, seed, max_steps=None, manual_actions=None):
    set_global_seeds(seed)

    resolved_weights = resolve_weights_path(weights_path)
    gsm = GameStateManager(seed=seed)
    net = MuZeroNet(seed=GLOBAL_SEED)
    rl_manager = ReinforcementLearningManager(net)
    rl_manager.params = net.load_params(resolved_weights)

    state = gsm.produce_initial_state()
    state_history = [get_blank_state(GRID_WIDTH, GRID_HEIGHT) for _ in range(LOOK_BACK - 1)] + [state]

    steps = [{
        "state": state,
        "step_idx": 0,
        "action": None,
        "reward": None,
        "total_reward": 0,
    }]

    total_reward = 0
    step_idx = 0
    step_limit = max_steps if max_steps is not None else MAX_EPISODE_STEPS

    while not gsm.is_final_state() and step_idx < step_limit:
        phi = state_history[-LOOK_BACK:]
        if manual_actions and step_idx < len(manual_actions):
            action = manual_actions[step_idx]
        elif manual_actions:
            action = 0
        else:
            action = rl_manager.play_as_actor(phi)
        next_state, reward = gsm.step(action)
        total_reward += reward
        step_idx += 1
        state_history.append(next_state)
        steps.append({
            "state": next_state,
            "step_idx": step_idx,
            "action": action,
            "reward": reward,
            "total_reward": total_reward,
        })

    return resolved_weights, steps


def simulate_train_episode(weights_path, seed, max_steps=None):
    set_global_seeds(seed)

    resolved_weights = resolve_weights_path(weights_path)
    gsm = GameStateManager(seed=seed)
    net = MuZeroNet(seed=GLOBAL_SEED)
    rl_manager = ReinforcementLearningManager(net)
    rl_manager.params = net.load_params(resolved_weights)
    asm = AbstractStateManager(net, rl_manager.params)
    rng = np.random.default_rng(seed)

    state = gsm.produce_initial_state()
    state_history = [get_blank_state(GRID_WIDTH, GRID_HEIGHT) for _ in range(LOOK_BACK - 1)] + [state]

    steps = [{
        "state": state,
        "step_idx": 0,
        "action": None,
        "reward": None,
        "total_reward": 0,
    }]

    total_reward = 0
    step_idx = 0
    step_limit = max_steps if max_steps is not None else MAX_EPISODE_STEPS

    while not gsm.is_final_state() and step_idx < step_limit:
        phi = state_history[-LOOK_BACK:]
        sigma_k = asm.get_abstract_state(phi)
        mcts = uMCTS(asm, rng=rng)
        root = mcts.search(
            sigma_k,
            MCTS_SIMULATIONS,
            MAX_SEARCH_DEPTH,
            legal_actions=gsm.get_legal_actions(state),
        )

        pi_k = np.zeros(POLICY_DIM, dtype=np.float32)
        for action, child in root.children.items():
            pi_k[action] = child.visit_count
        total_visits = np.sum(pi_k)
        if total_visits > 0:
            pi_k = pi_k / total_visits
        else:
            pi_k = np.ones(POLICY_DIM, dtype=np.float32) / POLICY_DIM

        action = int(rng.choice(np.arange(POLICY_DIM), p=pi_k))
        next_state, reward = gsm.step(action)
        total_reward += reward
        step_idx += 1
        state_history.append(next_state)
        steps.append({
            "state": next_state,
            "step_idx": step_idx,
            "action": action,
            "reward": reward,
            "total_reward": total_reward,
        })
        state = next_state

    return resolved_weights, steps


def state_to_image(
    state,
    step_idx,
    action,
    reward,
    total_reward,
    mode_label,
    prev_state=None,
    reward_history_text=None,
):
    img_w = GRID_WIDTH * CELL_SIZE
    extra_history_height = REWARD_HISTORY_HEIGHT if reward_history_text else 0
    img_h = GRID_HEIGHT * CELL_SIZE + INFO_BAR_HEIGHT + extra_history_height
    image = Image.new("RGB", (img_w, img_h), COLOR_BG)
    draw = ImageDraw.Draw(image)

    color_map = {
        0: COLOR_EMPTY,
        1: COLOR_DEBRIS,
        2: COLOR_RECEPTOR,
    }
    font_small = load_font(12)
    font_medium = load_font(13, bold=True)

    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            value = int(state[row, col])
            x0 = col * CELL_SIZE
            y0 = row * CELL_SIZE
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE
            draw.rectangle([x0, y0, x1, y1], fill=color_map[value], outline=COLOR_GRID, width=GRID_LINE_WIDTH)
            if value == 2:
                draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=COLOR_RECEPTOR, width=RECEPTOR_OUTLINE_WIDTH)
            if prev_state is not None and int(prev_state[row, col]) != value:
                draw.rectangle([x0 + 4, y0 + 4, x1 - 4, y1 - 4], outline=COLOR_CHANGE, width=CHANGE_HIGHLIGHT_WIDTH)

    info_y = GRID_HEIGHT * CELL_SIZE + 8
    reward_text = f"{reward:+.0f}" if reward is not None else "n/a"
    action_text = ACTION_NAMES.get(action, "n/a") if action is not None else "Start"
    line1 = f"Step {step_idx:03d} | Action: {action_text} | Reward: {reward_text}"
    line2 = f"Total score: {total_reward:+.0f} | Mode: {mode_label}"
    draw.text((8, info_y), line1, fill=COLOR_TEXT, font=font_small)
    draw.text((8, info_y + 22), line2, fill=COLOR_TEXT, font=font_small)
    if reward_history_text:
        draw.text((8, info_y + 40), reward_history_text, fill=COLOR_TEXT, font=font_small)

    if reward not in (None, 0):
        box_color = COLOR_REWARD_POS if reward > 0 else COLOR_REWARD_NEG
        reward_box = [img_w - 112, 10, img_w - 10, 42]
        draw.rounded_rectangle(reward_box, radius=8, fill=box_color)
        draw.text((reward_box[0] + 12, reward_box[1] + 7), f"Reward {reward:+.0f}", fill=COLOR_BG, font=font_medium)
    return image


def parse_actions(actions_text):
    if not actions_text:
        return None
    parts = [p.strip() for p in actions_text.split(",") if p.strip()]
    actions = [int(p) for p in parts]
    for a in actions:
        if a not in ACTION_NAMES:
            raise ValueError(f"Invalid manual action {a}. Use only 0=Stay, 1=Left, 2=Right.")
    return actions


def format_reward_history(steps, current_index, window):
    recent = []
    start = max(1, current_index - window + 1)
    for idx in range(start, current_index + 1):
        reward = steps[idx]["reward"]
        if reward is None:
            continue
        recent.append(f"{reward:+.0f}")
    return "Recent rewards: " + (" ".join(recent) if recent else "n/a")


def build_step_images_from_steps(steps, mode_label, show_reward_history=False, reward_history_window=10):
    step_images = []
    prev_state = None
    for idx, step in enumerate(steps):
        reward_history_text = None
        if show_reward_history:
            reward_history_text = format_reward_history(steps, idx, reward_history_window)
        image = state_to_image(
            step["state"],
            step_idx=step["step_idx"],
            action=step["action"],
            reward=step["reward"],
            total_reward=step["total_reward"],
            mode_label=mode_label,
            prev_state=prev_state,
            reward_history_text=reward_history_text,
        )
        step_images.append(
            {
                "step_idx": step["step_idx"],
                "reward": step["reward"],
                "image": image,
            }
        )
        prev_state = step["state"]
    return step_images


def expand_step_images_to_frames(step_images):
    frames = []
    for item in step_images:
        hold_count = REWARD_HOLD_FRAMES if item["reward"] not in (None, 0) else 1
        for _ in range(hold_count):
            frames.append(item["image"].copy())
    return frames


def generate_episode_frames(
    weights_path,
    seed,
    mode,
    max_steps=None,
    manual_actions=None,
    show_reward_history=False,
    reward_history_window=10,
):
    if manual_actions:
        mode_label = "Manual demo"
        resolved_weights, steps = simulate_eval_episode(weights_path, seed, max_steps=max_steps, manual_actions=manual_actions)
    elif mode == "train":
        mode_label = "Train (u-MCTS)"
        resolved_weights, steps = simulate_train_episode(weights_path, seed, max_steps=max_steps)
    else:
        mode_label = "Eval (actor-only)"
        resolved_weights, steps = simulate_eval_episode(weights_path, seed, max_steps=max_steps, manual_actions=None)
    step_images = build_step_images_from_steps(
        steps,
        mode_label,
        show_reward_history=show_reward_history,
        reward_history_window=reward_history_window,
    )
    frames = expand_step_images_to_frames(step_images)
    total_reward = steps[-1]["total_reward"]
    step_idx = steps[-1]["step_idx"]
    return resolved_weights, frames, total_reward, step_idx, step_images


def find_best_seed(weights_path, mode, seed_start, seed_end, max_steps=None):
    best = None
    for seed in range(seed_start, seed_end + 1):
        if mode == "train":
            resolved_weights, steps = simulate_train_episode(weights_path, seed, max_steps=max_steps)
        else:
            resolved_weights, steps = simulate_eval_episode(weights_path, seed, max_steps=max_steps, manual_actions=None)
        total_reward = steps[-1]["total_reward"]
        if best is None or total_reward > best["total_reward"]:
            best = {
                "weights_path": resolved_weights,
                "seed": seed,
                "total_reward": total_reward,
                "steps": steps,
            }
    return best


def save_gif(frames, output_path, fps):
    duration_ms = int(1000 / fps)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_step_images(step_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for item in step_images:
        filename = f"step_{item['step_idx']:03d}.png"
        item["image"].save(os.path.join(output_dir, filename))


def main():
    parser = argparse.ArgumentParser(description="Generate a BitFall GIF from a trained checkpoint for train/u-MCTS or eval/actor-only mode.")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="Checkpoint path or checkpoint basename inside run_* directories.")
    parser.add_argument("--output", default=None, help="Optional GIF output path.")
    parser.add_argument("--fps", type=int, default=6, help="Frames per second for the GIF.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on the number of steps to include.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Episode seed used for the game simulation.")
    parser.add_argument("--mode", choices=["eval", "train"], default="eval", help="Render actor-only final evaluation or training-time u-MCTS gameplay.")
    parser.add_argument("--auto-best-seed", action="store_true", help="Search a seed range and render the best episode automatically for the selected mode.")
    parser.add_argument("--seed-start", type=int, default=0, help="Start of the seed range for --auto-best-seed.")
    parser.add_argument("--seed-end", type=int, default=100, help="End of the seed range for --auto-best-seed.")
    parser.add_argument("--show-reward-history", action="store_true", help="Show a short recent reward history under the GIF info bar.")
    parser.add_argument("--reward-history-window", type=int, default=10, help="How many recent rewards to show when --show-reward-history is enabled.")
    parser.add_argument("--export-step-images", action="store_true", help="Also save one PNG image for each logical step.")
    parser.add_argument("--steps-dir", default=None, help="Optional directory for exported step images.")
    parser.add_argument(
        "--actions",
        default=None,
        help="Optional manual action sequence such as '2,2,2,1,1,0'. If provided, this is used instead of the actor.",
    )
    args = parser.parse_args()

    manual_actions = parse_actions(args.actions)
    if args.auto_best_seed and manual_actions:
        raise ValueError("--auto-best-seed cannot be combined with --actions.")
    if args.mode == "train" and manual_actions:
        raise ValueError("--actions cannot be combined with --mode train.")

    if args.auto_best_seed:
        best = find_best_seed(args.weights, args.mode, args.seed_start, args.seed_end, max_steps=args.max_steps)
        resolved_weights = best["weights_path"]
        chosen_seed = best["seed"]
        mode_label = "Train (u-MCTS)" if args.mode == "train" else "Eval (actor-only)"
        step_images = build_step_images_from_steps(
            best["steps"],
            mode_label,
            show_reward_history=args.show_reward_history,
            reward_history_window=args.reward_history_window,
        )
        frames = expand_step_images_to_frames(step_images)
        total_reward = best["total_reward"]
        num_steps = best["steps"][-1]["step_idx"]
    else:
        chosen_seed = args.seed
        resolved_weights, frames, total_reward, num_steps, step_images = generate_episode_frames(
            args.weights,
            seed=args.seed,
            mode=args.mode,
            max_steps=args.max_steps,
            manual_actions=manual_actions,
            show_reward_history=args.show_reward_history,
            reward_history_window=args.reward_history_window,
        )
    if args.output:
        output_path = args.output
    elif manual_actions:
        output_path = os.path.join(os.path.dirname(resolved_weights) or ".", "bitfall_manual_demo.gif")
    elif args.mode == "train":
        output_path = os.path.join(os.path.dirname(resolved_weights) or ".", "bitfall_train_umcts.gif")
    else:
        output_path = default_output_path(resolved_weights)
    save_gif(frames, output_path, fps=args.fps)
    if args.export_step_images:
        steps_dir = args.steps_dir
        if steps_dir is None:
            gif_stem = os.path.splitext(os.path.basename(output_path))[0]
            steps_dir = os.path.join(os.path.dirname(output_path) or ".", f"{gif_stem}_steps")
        save_step_images(step_images, steps_dir)

    print(f"Generated GIF from {resolved_weights}")
    print(f"Episode seed: {chosen_seed}")
    if manual_actions:
        mode_name = "manual demo"
    else:
        mode_name = "train/u-MCTS" if args.mode == "train" else "eval/actor-only"
    print(f"Mode: {mode_name}")
    print(f"Steps captured: {num_steps}")
    print(f"Episode total reward: {total_reward}")
    print(f"Saved GIF to {output_path}")
    if args.export_step_images:
        print(f"Saved step images to {steps_dir}")


if __name__ == "__main__":
    main()
