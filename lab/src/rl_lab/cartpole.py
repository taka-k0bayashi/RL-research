from __future__ import annotations

import argparse
import csv
import json
import statistics
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from rl_lab.algorithms import evaluate, train
from rl_lab.envs import make_environment
from rl_lab.models import make_model


@dataclass(frozen=True)
class Config:
    environment: str
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    episodes: int
    gamma: float
    learning_rate: float
    evaluate_episodes: int


def load_config(path: Path) -> Config:
    with path.open("rb") as file:
        raw = tomllib.load(file)
    training = raw["train"]
    return Config(
        environment=training["environment"],
        models=tuple(raw["models"]),
        seeds=tuple(training["seeds"]),
        episodes=training["episodes"],
        gamma=training["gamma"],
        learning_rate=training["learning_rate"],
        evaluate_episodes=raw["evaluate"]["episodes"],
    )


def save_run(
    directory: Path,
    model_name: str,
    seed: int,
    policy: nn.Module,
    train_scores: list[float],
    evaluation_scores: list[float],
) -> None:
    directory.mkdir(parents=True)
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("episode", "return"))
        writer.writerows(enumerate(train_scores, start=1))
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "parameters": sum(parameter.numel() for parameter in policy.parameters()),
                "seed": seed,
                "training_episodes": len(train_scores),
                "mean_last_50_training_return": statistics.mean(train_scores[-50:]),
                "mean_evaluation_return": statistics.mean(evaluation_scores),
                "evaluation_returns": evaluation_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(policy.state_dict(), directory / "model.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policies on CartPole")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    config = load_config(args.config)
    environment = make_environment(config.environment)
    try:
        observations = environment.observation_size
        actions = environment.action_size
    finally:
        environment.close()

    run_dir = args.runs_dir / datetime.now().strftime("cartpole-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    results: list[dict[str, int | float | str]] = []

    for model_name in config.models:
        for seed in config.seeds:
            torch.manual_seed(seed)
            policy = make_model(model_name, observations, actions)
            train_scores = train(
                config.environment,
                policy,
                seed,
                config.episodes,
                config.gamma,
                config.learning_rate,
                model_name,
            )
            evaluation_scores = evaluate(
                config.environment, policy, seed, config.evaluate_episodes
            )
            save_run(
                run_dir / model_name / f"seed-{seed}",
                model_name,
                seed,
                policy,
                train_scores,
                evaluation_scores,
            )
            results.append(
                {
                    "model": model_name,
                    "parameters": sum(
                        parameter.numel() for parameter in policy.parameters()
                    ),
                    "seed": seed,
                    "last_50_train_return": statistics.mean(train_scores[-50:]),
                    "evaluation_return": statistics.mean(evaluation_scores),
                }
            )

    comparison: list[dict[str, int | float | str]] = []
    for model_name in config.models:
        model_results = [row for row in results if row["model"] == model_name]
        evaluation_returns = [float(row["evaluation_return"]) for row in model_results]
        comparison.append(
            {
                "model": model_name,
                "parameters": model_results[0]["parameters"],
                "seeds": len(model_results),
                "mean_evaluation_return": statistics.mean(evaluation_returns),
                "std_evaluation_return": statistics.pstdev(evaluation_returns),
                "mean_last_50_train_return": statistics.mean(
                    float(row["last_50_train_return"]) for row in model_results
                ),
            }
        )

    with (run_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    for row in comparison:
        print(
            f"model={row['model']} parameters={row['parameters']} "
            f"evaluation={row['mean_evaluation_return']:.1f}"
            f"+/-{row['std_evaluation_return']:.1f}"
        )
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
