from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn

from rl_lab.envs.cartpole_pixels import CartPolePixelsEnvironment


def train(
    teacher: nn.Module,
    student: nn.Module,
    seed: int,
    episodes: int,
    learning_rate: float,
    parallel_environments: int,
    student_observation: str = "pixels",
) -> tuple[list[float], list[float], bool]:
    if episodes < 1 or parallel_environments < 1:
        raise ValueError("episodes and parallel_environments must be positive")
    if student_observation not in {"pixels", "state"}:
        raise ValueError("student_observation must be 'pixels' or 'state'")
    torch.manual_seed(seed)
    device = next(student.parameters()).device
    environments = [
        CartPolePixelsEnvironment()
        for _ in range(min(parallel_environments, episodes))
    ]
    optimizer = torch.optim.Adam(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    teacher.eval()
    student.train()
    scores: list[float] = []
    losses: list[float] = []
    interrupted = False
    executor = ThreadPoolExecutor(max_workers=len(environments))

    try:
        for batch_start in range(0, episodes, parallel_environments):
            batch_size = min(parallel_environments, episodes - batch_start)
            observations = list(
                executor.map(
                    lambda pair: pair[0].reset(pair[1]),
                    (
                        (environments[index], seed + batch_start + index)
                        for index in range(batch_size)
                    ),
                )
            )
            dones = [False] * batch_size
            batch_scores = [0.0] * batch_size
            teacher_probability = max(0.0, 1.0 - 2.0 * batch_start / episodes)

            while not all(dones):
                active = [index for index, done in enumerate(dones) if not done]
                pixels = torch.as_tensor(
                    np.stack([observations[index] for index in active]), device=device
                )
                states = torch.as_tensor(
                    np.stack(
                        [
                            environments[index].teacher_observation()
                            for index in active
                        ]
                    ),
                    device=device,
                )
                with torch.no_grad():
                    teacher_actions = teacher(states).argmax(dim=-1)
                student_logits = student(
                    states if student_observation == "state" else pixels
                )
                loss = nn.functional.cross_entropy(student_logits, teacher_actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                student_actions = student_logits.detach().argmax(dim=-1)
                use_teacher = torch.rand(len(active), device=device) < teacher_probability
                actions = torch.where(
                    use_teacher, teacher_actions, student_actions
                ).cpu().tolist()
                step_results = executor.map(
                    lambda pair: pair[0].step(pair[1]),
                    (
                        (environments[index], action)
                        for index, action in zip(active, actions)
                    ),
                )
                for index, result in zip(active, step_results):
                    observation, reward, done = result
                    observations[index] = observation
                    batch_scores[index] += reward
                    dones[index] = done

            scores.extend(batch_scores)
            if len(scores) % 50 < batch_size:
                print(
                    f"episode={len(scores)} mean_return={np.mean(scores[-50:]):.1f} "
                    f"loss={np.mean(losses[-100:]):.4f}"
                )
    except KeyboardInterrupt:
        interrupted = True
        print(f"interrupted after {len(scores)} episodes")
    finally:
        executor.shutdown(wait=True)
        for environment in environments:
            environment.close()

    return scores, losses, interrupted
