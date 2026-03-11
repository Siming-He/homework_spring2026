from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

STEP_FIELD_CANDIDATES = ("Train_EnvstepsSoFar", "Train EnvstepsSoFar", "step")
RETURN_FIELD_CANDIDATES = ("Eval_AverageReturn", "Eval AverageReturn", "Train_AverageReturn")


def find_first_existing_key(candidates: tuple[str, ...], fieldnames: list[str] | None) -> str | None:
    if fieldnames is None:
        return None
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def load_curve(log_csv: Path) -> tuple[list[float], list[float]] | None:
    with log_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        step_key = find_first_existing_key(STEP_FIELD_CANDIDATES, reader.fieldnames)
        return_key = find_first_existing_key(RETURN_FIELD_CANDIDATES, reader.fieldnames)
        if step_key is None or return_key is None:
            return None

        points: list[tuple[float, float]] = []
        for row in reader:
            step_raw = row.get(step_key, "")
            ret_raw = row.get(return_key, "")
            if not step_raw or not ret_raw:
                continue
            try:
                points.append((float(step_raw), float(ret_raw)))
            except ValueError:
                continue

    if not points:
        return None

    points.sort(key=lambda point: point[0])
    x_values = [x for x, _ in points]
    y_values = [y for _, y in points]
    return x_values, y_values


def lambda_from_flags(run_dir: Path) -> float | None:
    flags_path = run_dir / "flags.json"
    if not flags_path.exists():
        return None
    try:
        with flags_path.open() as flags_file:
            flags = json.load(flags_file)
    except (json.JSONDecodeError, OSError):
        return None
    gae_lambda = flags.get("gae_lambda")
    if isinstance(gae_lambda, (float, int)):
        return float(gae_lambda)
    return None


def run_label(run_dir: Path) -> str:
    lambda_value = lambda_from_flags(run_dir)
    if lambda_value is None:
        return run_dir.name
    return f"lambda={lambda_value:g}"


def run_sort_key(run_dir: Path) -> tuple[int, float, str]:
    lambda_value = lambda_from_flags(run_dir)
    if lambda_value is None:
        return (1, 0.0, run_dir.name)
    return (0, lambda_value, run_dir.name)


def main() -> None:
    hw2_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Plot LunarLander-v2 learning curves for all runs in hw2/exp."
    )
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=hw2_dir / "exp",
        help="Directory containing experiment runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=hw2_dir / "exp" / "plots" / "lunarlander_learning_curves.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    run_dirs = sorted(
        [path.parent for path in args.exp_dir.glob("LunarLander-v2_*/log.csv")],
        key=run_sort_key,
    )
    if not run_dirs:
        raise ValueError(f"No LunarLander-v2 runs found in {args.exp_dir}")

    figure, axis = plt.subplots(figsize=(9, 6))
    plotted = False

    for run_dir in run_dirs:
        curve = load_curve(run_dir / "log.csv")
        if curve is None:
            print(f"Skipping {run_dir}: missing usable curve columns")
            continue
        x_values, y_values = curve
        axis.plot(x_values, y_values, label=run_label(run_dir))
        plotted = True

    if not plotted:
        plt.close(figure)
        raise ValueError("No valid LunarLander-v2 curves were plotted.")

    axis.set_title("LunarLander-v2 Learning Curves")
    axis.set_xlabel("Train EnvstepsSoFar")
    axis.set_ylabel("Eval Average Return")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
