from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from hw1_imitation.data import Normalizer, load_pusht_zarr
from hw1_imitation.evaluation import ENV_ID
from hw1_imitation.model import BasePolicy


def pick_best_mse_run(exp_dir: Path) -> Path:
    best_reward = -float("inf")
    best_run: Path | None = None
    for run_dir in sorted(exp_dir.glob("seed_*")):
        cfg_path = run_dir / "wandb" / "files" / "config.yaml"
        summary_path = run_dir / "wandb" / "files" / "wandb-summary.json"
        if not cfg_path.exists() or not summary_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg.get("policy_type", {}).get("value") != "mse":
            continue
        summary = json.loads(summary_path.read_text())
        reward = float(summary.get("eval/mean_reward", -float("inf")))
        if reward >= best_reward:
            best_reward = reward
            best_run = run_dir
    if best_run is None:
        raise RuntimeError("No MSE run with wandb summary found in exp/")
    return best_run


def read_loss_curve(log_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, losses = [], []
    with log_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss = row.get("train/loss", "")
            if loss:
                steps.append(int(row["step"]))
                losses.append(float(loss))
    return np.asarray(steps), np.asarray(losses)


def evaluate_checkpoint(
    model: BasePolicy,
    normalizer: Normalizer,
    device: torch.device,
    *,
    chunk_size: int,
    num_episodes: int,
    flow_num_steps: int,
) -> float:
    env = gym.make(ENV_ID, obs_type="state")
    action_low = env.action_space.low
    action_high = env.action_space.high
    rewards = []
    model.eval()
    for ep_idx in range(num_episodes):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_idx = chunk_size
        action_chunk: np.ndarray | None = None
        best = 0.0
        while not done:
            if action_chunk is None or chunk_idx >= chunk_size:
                state = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                with torch.no_grad():
                    pred = model.sample_actions(state.unsqueeze(0), num_steps=flow_num_steps).cpu().numpy()[0]
                action_chunk = np.clip(normalizer.denormalize_action(pred), action_low, action_high)
                chunk_idx = 0
            action = action_chunk[chunk_idx]
            obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
            best = max(best, float(reward))
            done = terminated or truncated
            chunk_idx += 1
        rewards.append(best)
    env.close()
    return float(np.mean(rewards))


def draw_curve(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    x: np.ndarray,
    y: np.ndarray,
    title: str,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline="black", width=2)
    if len(x) == 0:
        draw.text((x0 + 8, y0 + 8), f"{title}: empty", fill="black")
        return
    left, right, top, bottom = x0 + 55, x1 - 20, y0 + 30, y1 - 45
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    def px(xx: float) -> float:
        return left + (xx - xmin) / (xmax - xmin) * (right - left)

    def py(yy: float) -> float:
        return bottom - (yy - ymin) / (ymax - ymin) * (bottom - top)

    for i in range(5):
        tx = i / 4
        xx = xmin + tx * (xmax - xmin)
        xi = int(left + tx * (right - left))
        draw.line([(xi, top), (xi, bottom)], fill="#e0e0e0", width=1)
        draw.text((xi - 22, bottom + 8), f"{int(xx)}", fill="black")
    for i in range(5):
        ty = i / 4
        yy = ymin + ty * (ymax - ymin)
        yi = int(bottom - ty * (bottom - top))
        draw.line([(left, yi), (right, yi)], fill="#e0e0e0", width=1)
        draw.text((x0 + 8, yi - 7), f"{yy:.3f}", fill="black")

    pts = [(px(float(xx)), py(float(yy))) for xx, yy in zip(x, y, strict=True)]
    if len(pts) == 1:
        draw.ellipse((pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2), fill="#1f77b4")
    else:
        draw.line(pts, fill="#1f77b4", width=2)
    draw.text((x0 + 8, y0 + 8), title, fill="black")
    draw.text((right - 70, bottom + 8), "step", fill="black")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, default=Path("exp"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--data-zarr", type=Path, default=Path("data/pusht/pusht_cchi_v7_replay.zarr"))
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir is not None else pick_best_mse_run(args.exp_dir)
    cfg = yaml.safe_load((run_dir / "wandb" / "files" / "config.yaml").read_text())
    chunk_size = int(cfg["chunk_size"]["value"])
    flow_num_steps = int(cfg["flow_num_steps"]["value"])

    loss_steps, losses = read_loss_curve(run_dir / "log.csv")
    states, actions, _ = load_pusht_zarr(args.data_zarr)
    normalizer = Normalizer.from_data(states, actions)
    device = torch.device(args.device)

    ckpt_dir = run_dir / "wandb" / "files" / "checkpoints"
    ckpts = sorted(
        ckpt_dir.glob("checkpoint_step_*.pkl"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    reward_steps, rewards = [], []
    for ckpt in ckpts:
        step = int(ckpt.stem.split("_")[-1])
        model = torch.load(ckpt, map_location=device, weights_only=False)
        reward = evaluate_checkpoint(
            model,
            normalizer,
            device,
            chunk_size=chunk_size,
            num_episodes=args.num_eval_episodes,
            flow_num_steps=flow_num_steps,
        )
        reward_steps.append(step)
        rewards.append(reward)
        print(f"step={step} mean_reward={reward:.6f}")

    reward_steps_arr = np.asarray(reward_steps)
    rewards_arr = np.asarray(rewards)
    reward_csv = run_dir / "reward_curve.csv"
    with reward_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "eval/mean_reward"])
        writer.writerows(zip(reward_steps, rewards, strict=True))

    img = Image.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(img)
    draw_curve(draw, (20, 20, 690, 600), loss_steps, losses, "train/loss vs step")
    draw_curve(
        draw,
        (710, 20, 1380, 600),
        reward_steps_arr,
        rewards_arr,
        "eval/mean_reward vs step",
    )
    out_path = run_dir / "training_curves.png"
    img.save(out_path)
    print(f"run_dir={run_dir}")
    print(f"saved={out_path}")
    print(f"saved={reward_csv}")


if __name__ == "__main__":
    main()
