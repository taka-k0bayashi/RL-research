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
    learning_rate: float,
    label: str,
) -> list[float]:
    torch.manual_seed(seed)
    environment = make_environment(environment_name)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    scores: list[float] = []

    try:
        for episode in range(episodes):
            observation = environment.reset(seed + episode)
            log_probabilities: list[torch.Tensor] = []
            rewards: list[float] = []
            done = False

            while not done:
                distribution = Categorical(logits=policy(torch.as_tensor(observation)))
                action = distribution.sample()
                observation, reward, done = environment.step(action.item())
                log_probabilities.append(distribution.log_prob(action))
                rewards.append(reward)

            returns = discounted_returns(rewards, gamma)
            returns = (returns - returns.mean()) / (
                returns.std(correction=0) + 1e-8
            )
            loss = -(torch.stack(log_probabilities) * returns).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scores.append(sum(rewards))

            if (episode + 1) % 50 == 0:
                print(
                    f"model={label} seed={seed} episode={episode + 1} "
                    f"mean_return={sum(scores[-50:]) / 50:.1f}"
                )
    finally:
        environment.close()

    return scores


@torch.inference_mode()
def evaluate(
    environment_name: str, policy: nn.Module, seed: int, episodes: int
) -> list[float]:
    environment = make_environment(environment_name)
    scores: list[float] = []
    try:
        for episode in range(episodes):
            observation = environment.reset(seed + 10_000 + episode)
            score = 0.0
            done = False
            while not done:
                action = policy(torch.as_tensor(observation)).argmax().item()
                observation, reward, done = environment.step(action)
                score += reward
            scores.append(score)
    finally:
        environment.close()
    return scores
