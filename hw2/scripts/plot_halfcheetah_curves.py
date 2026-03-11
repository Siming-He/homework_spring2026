from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

STEP_FIELD_CANDIDATES = ("Train_EnvstepsSoFar", "Train EnvstepsSoFar", "step")
EVAL_RETURN_FIELD_CANDIDATES = ("Eval_AverageReturn", "Eval AverageReturn")
BASELINE_LOSS_FIELD_CANDIDATES = ("Baseline Loss", "Baseline_Loss")


def find_first_existing_key(candidates: tuple[str, ...], fieldnames: list[str] | None) -> str | None:
    if fieldnames is None:
        return None
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def load_curve(log_csv: Path, y_key_candidates: tuple[str, ...]) -> tuple[list[float], list[float]] | None:
    with log_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        step_key = find_first_existing_key(STEP_FIELD_CANDIDATES, reader.fieldnames)
        y_key = find_first_existing_key(y_key_candidates, reader.fieldnames)
        if step_key is None or y_key is None:
            return None

        pairs: list[tuple[float, float]] = []
        for row in reader:
            step_raw = row.get(step_key, "")
            y_raw = row.get(y_key, "")
            if not step_raw or not y_raw:
                continue
            try:
                pairs.append((float(step_raw), float(y_raw)))
            except ValueError:
                continue

    if not pairs:
        return None

    pairs.sort(key=lambda pair: pair[0])
    x_values = [x for x, _ in pairs]
    y_values = [y for _, y in pairs]
    return x_values, y_values


def run_label(run_dir: Path) -> str:
    return run_dir.name


def plot_metric(
    run_dirs: list[Path],
    y_key_candidates: tuple[str, ...],
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))

    plotted_any = False
    for run_dir in run_dirs:
        log_csv = run_dir / "log.csv"
        if not log_csv.exists():
            print(f"Skipping {run_dir}: missing log.csv")
            continue

        curve = load_curve(log_csv, y_key_candidates)
        if curve is None:
            print(f"Skipping {run_dir}: missing required metric columns {y_key_candidates}")
            continue

        x_values, y_values = curve
        axis.plot(x_values, y_values, label=run_label(run_dir))
        plotted_any = True

    if not plotted_any:
        plt.close(figure)
        raise ValueError(f"No valid curves found for {output_path.name}")

    axis.set_title(title)
    axis.set_xlabel("Train EnvstepsSoFar")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Saved: {output_path}")


def main() -> None:
    hw2_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Plot HalfCheetah baseline loss and eval return learning curves."
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        type=Path,
        default=[
            hw2_dir / "exp" / "HalfCheetah-v4_cheetah_baseline_sd1_20260225_215908",
            hw2_dir / "exp" / "HalfCheetah-v4_cheetah_sd1_20260225_215248",
        ],
        help="Run directories containing log.csv files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=hw2_dir / "exp" / "plots",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    run_dirs = [run_dir.resolve() for run_dir in args.run_dirs]
    out_dir = args.out_dir.resolve()

    plot_metric(
        run_dirs=run_dirs,
        y_key_candidates=BASELINE_LOSS_FIELD_CANDIDATES,
        y_label="Baseline Loss",
        title="HalfCheetah Baseline Loss vs Env Steps",
        output_path=out_dir / "halfcheetah_baseline_loss_learning_curve.png",
    )
    plot_metric(
        run_dirs=run_dirs,
        y_key_candidates=EVAL_RETURN_FIELD_CANDIDATES,
        y_label="Eval Average Return",
        title="HalfCheetah Eval Return vs Env Steps",
        output_path=out_dir / "halfcheetah_eval_return_learning_curve.png",
    )


if __name__ == "__main__":
    main()
