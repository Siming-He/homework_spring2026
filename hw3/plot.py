from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wandb.proto.wandb_internal_pb2 import Record
from wandb.sdk.internal.datastore import DataStore


DEFAULT_RUN_DIR = Path(
    "../../../../../tmp/tmpn4mxuicv/wandb/run-20260311_173617-iwb2l184"
)


def iter_history_rows(run_dir: Path):
    datastore = DataStore()
    datastore.open_for_scan(str(next(run_dir.glob("*.wandb"))))

    while True:
        data = datastore.scan_data()
        if data is None:
            break

        record = Record()
        record.ParseFromString(data)

        if not record.HasField("history"):
            continue

        yield {
            ".".join(item.nested_key): json.loads(item.value_json)
            for item in record.history.item
        }


def load_metric_history(run_dir: Path, metric: str):
    steps = []
    values = []

    for row in iter_history_rows(run_dir):
        if metric not in row:
            continue

        steps.append(row["step"])
        values.append(row[metric])

    return steps, values


def load_eval_history(run_dir: Path):
    return load_metric_history(run_dir, "Eval_AverageReturn")


def smooth(values, window: int):
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=Path("eval_return_plot.png"))
    parser.add_argument("--eval_key", type=str, default="Eval_AverageReturn")
    parser.add_argument("--train_key", type=str, default=None)
    parser.add_argument("--train_window", type=int, default=10)
    args = parser.parse_args()

    eval_steps, eval_returns = load_metric_history(args.run_dir, args.eval_key)

    plt.figure(figsize=(8, 5))
    if args.train_key is not None:
        train_steps, train_returns = load_metric_history(args.run_dir, args.train_key)
        train_returns = smooth(train_returns, args.train_window)
        train_steps = train_steps[args.train_window - 1 :]
        plt.plot(train_steps, train_returns, linewidth=1.5, label="Train Return")
    plt.plot(eval_steps, eval_returns, linewidth=2, label="Eval Return")
    plt.xlabel("Environment Steps")
    plt.ylabel("Return")
    if args.train_key is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
