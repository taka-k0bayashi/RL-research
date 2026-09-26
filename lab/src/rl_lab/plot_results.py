from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be at least 1")
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def load_scores(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as file:
        return np.array(
            [float(row["return"]) for row in csv.DictReader(file)], dtype=float
        )


def plot_learning_curves(run_dir: Path, window: int) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    found = False

    for model_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        metric_files = sorted(model_dir.glob("seed-*/metrics.csv"))
        if not metric_files:
            continue
        scores = [load_scores(path) for path in metric_files]
        common_length = min(map(len, scores))
        smoothed = np.stack(
            [moving_average(values[:common_length], window) for values in scores]
        )
        effective_window = min(window, common_length)
        episodes = np.arange(effective_window, common_length + 1)
        mean = smoothed.mean(axis=0)
        deviation = smoothed.std(axis=0)
        line = axis.plot(episodes, mean, label=f"{model_dir.name}: mean ± 1 SD")[0]
        axis.fill_between(
            episodes,
            np.clip(mean - deviation, 0, 500),
            np.clip(mean + deviation, 0, 500),
            color=line.get_color(),
            alpha=0.18,
        )
        found = True

    if not found:
        plt.close(figure)
        raise FileNotFoundError(f"No metrics.csv files under {run_dir}")

    axis.axhline(475, color="black", linestyle="--", linewidth=1, label="solved: 475")
    axis.set(title=f"CartPole learning curves ({window}-episode moving average)")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Episode return across seeds")
    axis.set_ylim(0, 510)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = run_dir / "learning_curves.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_evaluation(run_dir: Path) -> Path:
    with (run_dir / "comparison.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    names = [row["model"] for row in rows]
    means = [float(row["mean_evaluation_return"]) for row in rows]
    deviations = [float(row["std_evaluation_return"]) for row in rows]
    error_bars = np.array(
        [
            [min(mean, deviation) for mean, deviation in zip(means, deviations)],
            [min(500 - mean, deviation) for mean, deviation in zip(means, deviations)],
        ]
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(names, means, yerr=error_bars, capsize=5)
    axis.axhline(475, color="black", linestyle="--", linewidth=1, label="solved: 475")
    axis.set(title="CartPole evaluation by policy model (mean ± 1 SD)")
    axis.set_xlabel("Policy model")
    axis.set_ylabel("Mean evaluation return across seeds")
    axis.set_ylim(0, 525)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = run_dir / "evaluation_comparison.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a completed experiment run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args()

    for output in (
        plot_learning_curves(args.run_dir, args.window),
        plot_evaluation(args.run_dir),
    ):
        print(f"saved={output}")


if __name__ == "__main__":
    main()
