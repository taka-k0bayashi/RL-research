from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from rl_lab.algorithms import evaluate
from rl_lab.algorithms.distillation import train
from rl_lab.cartpole import save_run
from rl_lab.models import make_model


def load_policy_weights(model: nn.Module, path: Path) -> None:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    state = loaded.get("model_state_dict", loaded)
    try:
        model.load_state_dict(state)
        return
    except RuntimeError as original_error:
        actor_state = {
            name: value
            for name, value in model.state_dict().items()
            if not name.startswith("critic.")
        }
        if len(state) != len(actor_state) or any(
            source.shape != target.shape
            for source, target in zip(state.values(), actor_state.values())
        ):
            raise original_error
        model.load_state_dict(dict(zip(actor_state, state.values())), strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill a CartPole policy"
    )
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--teacher-model", default="mlp_32")
    parser.add_argument("--student-model", default="cnn_16x32_fc128")
    parser.add_argument(
        "--student-observation", choices=("pixels", "state"), default="pixels"
    )
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--parallel-environments", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if not args.teacher.is_file():
        parser.error(f"teacher weights not found: {args.teacher}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = make_model(args.teacher_model, (4,), 2)
    load_policy_weights(teacher, args.teacher)
    teacher.requires_grad_(False)
    teacher.to(device)
    teacher_scores = evaluate("cartpole", teacher, args.seed, args.evaluation_episodes)
    print(f"device={device} teacher_evaluation={statistics.mean(teacher_scores):.1f}")

    student_shape = (4, 84, 84) if args.student_observation == "pixels" else (4,)
    student = make_model(args.student_model, student_shape, 2).to(device)
    student.critic.requires_grad_(False)
    scores, losses, interrupted = train(
        teacher,
        student,
        args.seed,
        args.episodes,
        args.learning_rate,
        args.parallel_environments,
        args.student_observation,
    )
    evaluation_scores = (
        evaluate(
            "cartpole_pixels" if args.student_observation == "pixels" else "cartpole",
            student,
            args.seed,
            args.evaluation_episodes,
            args.parallel_environments,
        )
        if scores
        else []
    )

    run_dir = args.runs_dir / datetime.now().strftime(
        "cartpole-distillation-%Y%m%d-%H%M%S"
    )
    output = run_dir / args.student_model / f"seed-{args.seed}"
    save_run(
        output,
        "distillation",
        args.student_model,
        args.seed,
        student,
        scores,
        evaluation_scores,
        interrupted,
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "teacher": str(args.teacher),
                "runs_dir": str(args.runs_dir),
                "device": str(device),
                "teacher_evaluation_return": statistics.mean(teacher_scores),
                "mean_distillation_loss": statistics.mean(losses) if losses else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (run_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "model",
                "parameters",
                "seeds",
                "mean_evaluation_return",
                "std_evaluation_return",
                "mean_last_50_train_return",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": args.student_model,
                "parameters": sum(
                    parameter.numel()
                    for parameter in student.parameters()
                    if parameter.requires_grad
                ),
                "seeds": 1,
                "mean_evaluation_return": (
                    statistics.mean(evaluation_scores) if evaluation_scores else 0
                ),
                "std_evaluation_return": 0,
                "mean_last_50_train_return": (
                    statistics.mean(scores[-50:]) if scores else 0
                ),
            }
        )
    evaluation = statistics.mean(evaluation_scores) if evaluation_scores else 0
    print(f"student_evaluation={evaluation:.1f} saved={run_dir}")


if __name__ == "__main__":
    main()
