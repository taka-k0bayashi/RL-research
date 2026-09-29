from __future__ import annotations

import argparse
import csv
import json
import statistics
import tomllib
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from rl_lab.algorithms import evaluate, train_actor_critic, train_reinforce
from rl_lab.envs import make_environment
from rl_lab.models import make_model


@dataclass(frozen=True)
class Config:
    algorithm: str
    environment: str
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    episodes: int
    gamma: float
    gae_lambda: float
    learning_rate: float
    entropy_coefficient_start: float
    entropy_coefficient_end: float
    value_loss_coefficient: float
    max_gradient_norm: float
    episodes_per_update: int
    evaluate_episodes: int


def load_config(path: Path) -> Config:
    with path.open("rb") as file:
        raw = tomllib.load(file)
    training = raw["train"]
    algorithm = raw["algorithm"]
    if algorithm not in {"actor_critic", "reinforce"}:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return Config(
        algorithm=algorithm,
        environment=training["environment"],
        models=tuple(raw["models"]),
        seeds=tuple(training["seeds"]),
        episodes=training["episodes"],
        gamma=training["gamma"],
        gae_lambda=training.get("gae_lambda", 0.95),
        learning_rate=training["learning_rate"],
        entropy_coefficient_start=training.get("entropy_coefficient_start", 0.0),
        entropy_coefficient_end=training.get("entropy_coefficient_end", 0.0),
        value_loss_coefficient=training.get("value_loss_coefficient", 0.0),
        max_gradient_norm=training.get("max_gradient_norm", 0.0),
        episodes_per_update=training.get("episodes_per_update", 1),
        evaluate_episodes=raw["evaluate"]["episodes"],
    )


def save_run(
    directory: Path,
    algorithm: str,
    model_name: str,
    seed: int,
    policy: nn.Module,
    train_scores: list[float],
    evaluation_scores: list[float],
    interrupted: bool,
    checkpoint_state: dict[str, object] | None = None,
) -> None:
    directory.mkdir(parents=True)
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("episode", "return"))
        writer.writerows(enumerate(train_scores, start=1))
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "model": model_name,
                "parameters": sum(
                    parameter.numel()
                    for parameter in policy.parameters()
                    if parameter.requires_grad
                ),
                "seed": seed,
                "interrupted": interrupted,
                "training_episodes": len(train_scores),
                "mean_last_50_training_return": (
                    statistics.mean(train_scores[-50:]) if train_scores else None
                ),
                "mean_evaluation_return": (
                    statistics.mean(evaluation_scores) if evaluation_scores else None
                ),
                "evaluation_returns": evaluation_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {name: value.detach().cpu() for name, value in policy.state_dict().items()},
        directory / "model.pt",
    )
    if checkpoint_state is not None:
        torch.save(
            {
                "algorithm": algorithm,
                "model_name": model_name,
                "seed": seed,
                "episode": len(train_scores),
                "scores": train_scores,
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in policy.state_dict().items()
                },
                **checkpoint_state,
            },
            directory / "checkpoint.pt",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policies on an environment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--resume", type=Path, metavar="CHECKPOINT")
    args = parser.parse_args()

    config = load_config(args.config)
    resume_checkpoint: dict[str, object] | None = None
    if args.resume is not None:
        if args.models or args.learning_rate is not None:
            parser.error("--resume cannot be combined with model or learning-rate overrides")
        if not args.resume.is_file():
            parser.error(f"checkpoint not found: {args.resume}")
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=True
        )
        checkpoint_algorithm = str(
            resume_checkpoint.get("algorithm", "actor_critic")
        )
        if checkpoint_algorithm != config.algorithm:
            parser.error(
                f"checkpoint uses {checkpoint_algorithm}, config uses {config.algorithm}"
            )
        config = replace(
            config,
            models=(str(resume_checkpoint["model_name"]),),
            seeds=(int(resume_checkpoint["seed"]),),
        )
        if int(resume_checkpoint["episode"]) >= config.episodes:
            parser.error("checkpoint already reached the configured episode count")
    if args.models:
        config = replace(config, models=tuple(args.models))
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            parser.error("--learning-rate must be positive")
        config = replace(config, learning_rate=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"device={device} ({device_name})")
    environment = make_environment(config.environment)
    try:
        observation_shape = environment.observation_shape
        actions = environment.action_size
        continuous = getattr(environment, "continuous", False)
    finally:
        environment.close()

    run_dir = args.runs_dir / datetime.now().strftime(f"{config.environment}-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "device": str(device),
                "device_name": device_name,
                "torch_version": torch.__version__,
                **(
                    {"resumed_from": str(args.resume.resolve())}
                    if args.resume is not None
                    else {}
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results: list[dict[str, int | float | str]] = []

    for model_name in config.models:
        for seed in config.seeds:
            torch.manual_seed(seed)
            policy = make_model(
                model_name, observation_shape, actions, continuous
            ).to(device)
            if config.algorithm == "reinforce":
                policy.critic.requires_grad_(False)
            if resume_checkpoint is not None:
                policy.load_state_dict(resume_checkpoint["model_state_dict"])
            resume_arguments = dict(
                start_episode=(
                    int(resume_checkpoint["episode"])
                    if resume_checkpoint is not None
                    else 0
                ),
                initial_scores=(
                    list(resume_checkpoint["scores"])
                    if resume_checkpoint is not None
                    else None
                ),
                optimizer_state=(
                    resume_checkpoint["optimizer_state_dict"]
                    if resume_checkpoint is not None
                    else None
                ),
                torch_rng_state=(
                    resume_checkpoint["torch_rng_state"]
                    if resume_checkpoint is not None
                    else None
                ),
                cuda_rng_states=(
                    resume_checkpoint["cuda_rng_state_all"]
                    if resume_checkpoint is not None
                    else None
                ),
            )
            if config.algorithm == "reinforce":
                train_scores, interrupted, checkpoint_state = train_reinforce(
                    config.environment,
                    policy,
                    seed,
                    config.episodes,
                    config.gamma,
                    config.learning_rate,
                    model_name,
                    **resume_arguments,
                )
            else:
                train_scores, interrupted, checkpoint_state = train_actor_critic(
                    config.environment,
                    policy,
                    seed,
                    config.episodes,
                    config.gamma,
                    config.gae_lambda,
                    config.learning_rate,
                    config.entropy_coefficient_start,
                    config.entropy_coefficient_end,
                    config.value_loss_coefficient,
                    config.max_gradient_norm,
                    config.episodes_per_update,
                    model_name,
                    **resume_arguments,
                )
            if interrupted:
                output = run_dir / model_name / f"seed-{seed}"
                save_run(
                    output,
                    config.algorithm,
                    model_name,
                    seed,
                    policy,
                    train_scores,
                    [],
                    True,
                    checkpoint_state,
                )
                print(f"saved interrupted run={output}")
                return
            evaluation_scores = evaluate(
                config.environment,
                policy,
                seed,
                config.evaluate_episodes,
                config.episodes_per_update,
            )
            save_run(
                run_dir / model_name / f"seed-{seed}",
                config.algorithm,
                model_name,
                seed,
                policy,
                train_scores,
                evaluation_scores,
                False,
            )
            results.append(
                {
                    "model": model_name,
                    "parameters": sum(
                        parameter.numel()
                        for parameter in policy.parameters()
                        if parameter.requires_grad
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
