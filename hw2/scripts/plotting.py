from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

RUN_DIR_PATTERN = re.compile(r"^[^_]+_(?P<exp_name>.+)_sd\d+_(?P<date>\d{8})_(?P<time>\d{6})$")
STEP_FIELD_CANDIDATES = ("Train_EnvstepsSoFar", "Train EnvstepsSoFar")
RETURN_FIELD_CANDIDATES = (
    "Eval_AverageReturn",
    "Eval AverageReturn",
    "Train_AverageReturn",
    "Train AverageReturn",
)
PREFERRED_ORDER = (
    "cartpole",
    "cartpole_na",
    "cartpole_rtg",
    "cartpole_rtg_na",
    "cartpole_lb",
    "cartpole_lb_na",
    "cartpole_lb_rtg",
    "cartpole_lb_rtg_na",
)


@dataclass
class RunCurve:
    exp_name: str
    run_dir: Path
    timestamp: str
    env_steps: list[float]
    avg_returns: list[float]


def parse_run_name(run_dir_name: str) -> tuple[str, str] | None:
    match = RUN_DIR_PATTERN.match(run_dir_name)
    if match is None:
        return None
    exp_name = match.group("exp_name")
    timestamp = f"{match.group('date')}_{match.group('time')}"
    return exp_name, timestamp


def find_first_existing_key(candidates: tuple[str, ...], fieldnames: list[str] | None) -> str | None:
    if fieldnames is None:
        return None
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def load_curve_from_csv(log_path: Path) -> tuple[list[float], list[float]]:
    with log_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        step_key = find_first_existing_key(STEP_FIELD_CANDIDATES, reader.fieldnames)
        return_key = find_first_existing_key(RETURN_FIELD_CANDIDATES, reader.fieldnames)
        if step_key is None or return_key is None:
            raise ValueError(f"Missing required fields in {log_path}")

        env_steps: list[float] = []
        avg_returns: list[float] = []
        for row in reader:
            step_value = row.get(step_key, "")
            return_value = row.get(return_key, "")
            if not step_value or not return_value:
                continue
            env_steps.append(float(step_value))
            avg_returns.append(float(return_value))

    if not env_steps:
        raise ValueError(f"No valid curve data in {log_path}")

    paired_values = sorted(zip(env_steps, avg_returns), key=lambda value: value[0])
    sorted_steps = [step for step, _ in paired_values]
    sorted_returns = [avg_return for _, avg_return in paired_values]
    return sorted_steps, sorted_returns


def collect_latest_runs(exp_dir: Path) -> dict[str, RunCurve]:
    runs_by_exp_name: dict[str, list[RunCurve]] = defaultdict(list)

    for log_path in sorted(exp_dir.glob("*/log.csv")):
        run_dir = log_path.parent
        parsed_run_name = parse_run_name(run_dir.name)
        if parsed_run_name is None:
            continue

        exp_name, timestamp = parsed_run_name
        env_steps, avg_returns = load_curve_from_csv(log_path)
        runs_by_exp_name[exp_name].append(
            RunCurve(
                exp_name=exp_name,
                run_dir=run_dir,
                timestamp=timestamp,
                env_steps=env_steps,
                avg_returns=avg_returns,
            )
        )

    latest_runs: dict[str, RunCurve] = {}
    for exp_name, runs in runs_by_exp_name.items():
        latest_runs[exp_name] = max(runs, key=lambda run: run.timestamp)
    return latest_runs


def exp_sort_key(exp_name: str) -> tuple[int, str]:
    if exp_name in PREFERRED_ORDER:
        return (PREFERRED_ORDER.index(exp_name), exp_name)
    return (len(PREFERRED_ORDER), exp_name)


def plot_learning_curves(curves: dict[str, RunCurve], title: str, output_path: Path) -> None:
    if not curves:
        return

    figure, axis = plt.subplots(figsize=(9, 6))
    for exp_name in sorted(curves, key=exp_sort_key):
        curve = curves[exp_name]
        axis.plot(curve.env_steps, curve.avg_returns, label=exp_name)

    axis.set_title(title)
    axis.set_xlabel("Train EnvstepsSoFar")
    axis.set_ylabel("Average Return")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CartPole learning curves from hw2/exp logs.")
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "exp",
        help="Directory containing experiment run subdirectories.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "exp" / "plots",
        help="Directory to save generated plot images.",
    )
    args = parser.parse_args()

    latest_runs = collect_latest_runs(args.exp_dir)
    small_batch_runs = {
        exp_name: run
        for exp_name, run in latest_runs.items()
        if exp_name.startswith("cartpole") and not exp_name.startswith("cartpole_lb")
    }
    large_batch_runs = {
        exp_name: run for exp_name, run in latest_runs.items() if exp_name.startswith("cartpole_lb")
    }

    small_batch_plot_path = args.out_dir / "cartpole_small_batch_learning_curves.png"
    large_batch_plot_path = args.out_dir / "cartpole_large_batch_learning_curves.png"

    plot_learning_curves(
        curves=small_batch_runs,
        title="CartPole Small Batch (no lb): Average Return vs Env Steps",
        output_path=small_batch_plot_path,
    )
    plot_learning_curves(
        curves=large_batch_runs,
        title="CartPole Large Batch (lb): Average Return vs Env Steps",
        output_path=large_batch_plot_path,
    )

    if small_batch_runs:
        print(f"Saved: {small_batch_plot_path}")
    else:
        print("No small-batch cartpole runs found.")
    if large_batch_runs:
        print(f"Saved: {large_batch_plot_path}")
    else:
        print("No large-batch cartpole_lb runs found.")


if __name__ == "__main__":
    main()
