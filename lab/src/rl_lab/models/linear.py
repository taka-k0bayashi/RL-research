import torch
from torch import nn


class LinearPolicy(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...], actions: int) -> None:
        super().__init__()
        if len(observation_shape) != 1:
            raise ValueError("LinearPolicy requires vector observations")
        self.actor = nn.Linear(observation_shape[0], actions)
        self.critic = nn.Linear(observation_shape[0], 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)

    def actor_critic(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(observation), self.critic(observation).squeeze(-1)
