import torch
from torch import nn


class MLP32Policy(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...], actions: int) -> None:
        super().__init__()
        if len(observation_shape) != 1:
            raise ValueError("MLP32Policy requires vector observations")
        self.features = nn.Sequential(
            nn.Linear(observation_shape[0], 32),
            nn.ReLU(),
        )
        self.actor = nn.Linear(32, actions)
        self.critic = nn.Linear(32, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.features(observation))

    def actor_critic(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(observation)
        return self.actor(features), self.critic(features).squeeze(-1)
