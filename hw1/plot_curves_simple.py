from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from hw1_imitation.data import Normalizer, load_pusht_zarr
from hw1_imitation.evaluation import ENV_ID


def eval_reward(model: torch.nn.Module, normalizer: Normalizer, chunk_size: int, num_eps: int) -> float:
    env = gym.make(ENV_ID, obs_type="state")
    low, high = env.action_space.low, env.action_space.high
    rewards = []
    model.eval()
    device = next(model.parameters()).device

    for seed in range(num_eps):
        obs, _ = env.reset(seed=seed)
        done = False
        chunk = None
        i = chunk_size
        best = 0.0
        while not done:
            if chunk is None or i >= chunk_size:
                s = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                with torch.no_grad():
                    chunk = model.sample_actions(s[None]).cpu().numpy()[0]
                chunk = np.clip(normalizer.denormalize_action(chunk), low, high)
                i = 0
            obs, r, term, trunc, _ = env.step(chunk[i].astype(np.float32))
            best = max(best, float(r))
            done = term or trunc
            i += 1
        rewards.append(best)
    env.close()
    return float(np.mean(rewards))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("exp/seed_42_20260211_231548"))
    p.add_argument("--zarr", type=Path, default=Path("data/pusht/pusht_cchi_v7_replay.zarr"))
    p.add_argument("--num-eps", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    loss_steps, losses = [], []
    with (args.run_dir / "log.csv").open() as f:
        for row in csv.DictReader(f):
            if row["train/loss"] == "":
                continue
            loss_steps.append(int(row["step"]))
            losses.append(float(row["train/loss"]))
    loss_steps = np.asarray(loss_steps)
    losses = np.asarray(losses)

    states, actions, _ = load_pusht_zarr(args.zarr)
    normalizer = Normalizer.from_data(states, actions)

    ckpts = sorted(
        (args.run_dir / "wandb" / "files" / "checkpoints").glob("checkpoint_step_*.pkl"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    reward_steps, rewards = [], []
    for ckpt in ckpts:
        step = int(ckpt.stem.split("_")[-1])
        model = torch.load(ckpt, map_location=args.device, weights_only=False)
        rew = eval_reward(model, normalizer, chunk_size=model.chunk_size, num_eps=args.num_eps)
        reward_steps.append(step)
        rewards.append(rew)
        print(step, rew)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(loss_steps, losses)
    ax[0].set_title("Train Loss")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("MSE")
    ax[1].plot(reward_steps, rewards, marker="o")
    ax[1].set_title("Eval Mean Reward")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Reward")
    fig.tight_layout()
    out = args.run_dir / "training_curves.png"
    fig.savefig(out, dpi=200)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
