import torch
from torch import nn

from rl_lab.algorithms.actor_critic import action_distribution, discounted_returns
from rl_lab.envs import make_environment


def train(
    environment_name: str,
    policy: nn.Module,
    seed: int,
    episodes: int,
    gamma: float,
    learning_rate: float,
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
    environment = make_environment(environment_name)
    optimizer = torch.optim.Adam(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
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

    try:
        for episode in range(start_episode, episodes):
            observation = environment.reset(seed + episode)
            log_probabilities: list[torch.Tensor] = []
            rewards: list[float] = []
            done = False

            while not done:
                distribution = action_distribution(
                    policy, policy(torch.as_tensor(observation, device=device))
                )
                action = distribution.sample()
                observation, reward, done = environment.step(action.tolist())
                log_probabilities.append(distribution.log_prob(action))
                rewards.append(reward)

            returns = discounted_returns(rewards, gamma).to(device)
            returns = (returns - returns.mean()) / (
                returns.std(correction=0) + 1e-8
            )
            loss = -(torch.stack(log_probabilities) * returns).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scores.append(sum(rewards))
            completed_episodes = episode + 1
            checkpoint_rng_state = torch.get_rng_state()
            checkpoint_cuda_rng_states = (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            )

            if completed_episodes % 50 == 0:
                print(
                    f"model={label} seed={seed} episode={completed_episodes} "
                    f"mean_return={sum(scores[-50:]) / 50:.1f}"
                )
    except KeyboardInterrupt:
        interrupted = True
        scores = scores[:completed_episodes]
        print(f"model={label} interrupted after {completed_episodes} episodes")
    finally:
        environment.close()

    return scores, interrupted, {
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": checkpoint_rng_state,
        "cuda_rng_state_all": checkpoint_cuda_rng_states,
    }
