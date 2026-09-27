from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rl_lab.envs import make_environment


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns: list[float] = []
    total = 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        returns.append(total)
    return torch.tensor(list(reversed(returns)), dtype=torch.float32)


def train(
    environment_name: str,
    policy: nn.Module,
    seed: int,
    episodes: int,
    gamma: float,
    gae_lambda: float,
    learning_rate: float,
    entropy_coefficient_start: float,
    entropy_coefficient_end: float,
    value_loss_coefficient: float,
    max_gradient_norm: float,
    episodes_per_update: int,
    label: str,
    start_episode: int = 0,
    initial_scores: list[float] | None = None,
    optimizer_state: dict[str, object] | None = None,
    torch_rng_state: torch.Tensor | None = None,
    cuda_rng_states: list[torch.Tensor] | None = None,
) -> tuple[list[float], bool, dict[str, object]]:
    torch.manual_seed(seed)
    device = next(policy.parameters()).device
    scores = list(initial_scores or [])
    if len(scores) != start_episode:
        raise ValueError("start_episode must match the number of existing scores")
    if episodes_per_update < 1:
        raise ValueError("episodes_per_update must be positive")
    environments = [
        make_environment(environment_name)
        for _ in range(min(episodes_per_update, episodes - start_episode))
    ]
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)
    if cuda_rng_states is not None and device.type == "cuda":
        torch.cuda.set_rng_state_all(cuda_rng_states)
    completed_episodes = start_episode
    checkpoint_rng_state = torch.get_rng_state()
    checkpoint_cuda_rng_states = (
        torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    )
    interrupted = False

    executor = ThreadPoolExecutor(max_workers=len(environments))
    try:
        for batch_start in range(start_episode, episodes, episodes_per_update):
            batch_end = min(batch_start + episodes_per_update, episodes)
            batch_size = batch_end - batch_start
            log_probabilities: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            batch_advantages: list[torch.Tensor] = []
            batch_value_targets: list[torch.Tensor] = []
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
            episode_log_probabilities: list[list[torch.Tensor]] = [
                [] for _ in range(batch_size)
            ]
            episode_entropies: list[list[torch.Tensor]] = [
                [] for _ in range(batch_size)
            ]
            episode_values: list[list[torch.Tensor]] = [
                [] for _ in range(batch_size)
            ]
            episode_rewards: list[list[float]] = [[] for _ in range(batch_size)]

            while not all(dones):
                active = [index for index, done in enumerate(dones) if not done]
                logits, active_values = policy.actor_critic(
                    torch.as_tensor(
                        np.stack([observations[index] for index in active]),
                        device=device,
                    )
                )
                distribution = Categorical(logits=logits)
                actions = distribution.sample()
                active_log_probabilities = distribution.log_prob(actions)
                active_entropies = distribution.entropy()

                active_actions = actions.detach().cpu().tolist()
                step_results = executor.map(
                    lambda pair: pair[0].step(pair[1]),
                    (
                        (environments[index], action)
                        for index, action in zip(active, active_actions)
                    ),
                )
                for position, (index, result) in enumerate(zip(active, step_results)):
                    observation, reward, done = result
                    observations[index] = observation
                    dones[index] = done
                    episode_log_probabilities[index].append(
                        active_log_probabilities[position]
                    )
                    episode_entropies[index].append(active_entropies[position])
                    episode_values[index].append(active_values[position])
                    episode_rewards[index].append(reward)

            for index in range(batch_size):
                rewards = episode_rewards[index]
                log_probabilities.extend(episode_log_probabilities[index])
                entropies.extend(episode_entropies[index])
                values.extend(episode_values[index])
                episode_value_tensor = torch.stack(episode_values[index])
                next_values = torch.cat(
                    (episode_value_tensor[1:].detach(), torch.zeros(1, device=device))
                )
                deltas = (
                    torch.as_tensor(rewards, device=device)
                    + gamma * next_values
                    - episode_value_tensor.detach()
                )
                advantages = torch.empty_like(deltas)
                advantage = torch.zeros((), device=device)
                for step in range(len(rewards) - 1, -1, -1):
                    advantage = deltas[step] + gamma * gae_lambda * advantage
                    advantages[step] = advantage
                batch_advantages.append(advantages)
                batch_value_targets.append(advantages + episode_value_tensor.detach())
                scores.append(sum(rewards))

                episode = batch_start + index
                if (episode + 1) % 50 == 0:
                    print(
                        f"model={label} seed={seed} episode={episode + 1} "
                        f"mean_return={sum(scores[-50:]) / 50:.1f}"
                    )

            predicted_values = torch.stack(values)
            advantages = torch.cat(batch_advantages)
            advantages = (advantages - advantages.mean()) / (
                advantages.std(correction=0) + 1e-8
            )
            progress = batch_end / episodes
            entropy_coefficient = entropy_coefficient_start + progress * (
                entropy_coefficient_end - entropy_coefficient_start
            )
            actor_loss = -(
                torch.stack(log_probabilities) * advantages
            ).mean() - entropy_coefficient * torch.stack(entropies).mean()
            critic_loss = nn.functional.smooth_l1_loss(
                predicted_values, torch.cat(batch_value_targets)
            )
            loss = actor_loss + value_loss_coefficient * critic_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_gradient_norm)
            optimizer.step()
            completed_episodes = batch_end
            checkpoint_rng_state = torch.get_rng_state()
            checkpoint_cuda_rng_states = (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            )
    except KeyboardInterrupt:
        interrupted = True
        scores = scores[:completed_episodes]
        print(f"model={label} interrupted after {completed_episodes} episodes")
    finally:
        executor.shutdown(wait=True)
        for environment in environments:
            environment.close()

    return scores, interrupted, {
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": checkpoint_rng_state,
        "cuda_rng_state_all": checkpoint_cuda_rng_states,
    }


@torch.inference_mode()
def evaluate(
    environment_name: str,
    policy: nn.Module,
    seed: int,
    episodes: int,
    parallel_environments: int = 1,
) -> list[float]:
    if parallel_environments < 1:
        raise ValueError("parallel_environments must be positive")
    device = next(policy.parameters()).device
    environments = [
        make_environment(environment_name)
        for _ in range(min(parallel_environments, episodes))
    ]
    scores: list[float] = []
    executor = ThreadPoolExecutor(max_workers=len(environments))
    try:
        for batch_start in range(0, episodes, parallel_environments):
            batch_size = min(parallel_environments, episodes - batch_start)
            observations = list(
                executor.map(
                    lambda pair: pair[0].reset(pair[1]),
                    (
                        (
                            environments[index],
                            seed + 10_000 + batch_start + index,
                        )
                        for index in range(batch_size)
                    ),
                )
            )
            batch_scores = [0.0] * batch_size
            dones = [False] * batch_size
            while not all(dones):
                active = [index for index, done in enumerate(dones) if not done]
                actions = (
                    policy(
                        torch.as_tensor(
                            np.stack([observations[index] for index in active]),
                            device=device,
                        )
                    )
                    .argmax(dim=-1)
                    .cpu()
                    .tolist()
                )
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
    finally:
        executor.shutdown(wait=True)
        for environment in environments:
            environment.close()
    return scores
