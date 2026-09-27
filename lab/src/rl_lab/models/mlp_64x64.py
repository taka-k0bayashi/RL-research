import torch
from torch import nn


class MLP64x64Policy(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...], actions: int) -> None:
        super().__init__()
        if len(observation_shape) != 1:
            raise ValueError("MLP64x64Policy requires vector observations")
        self.features = nn.Sequential(
            nn.Linear(observation_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.features(observation))

    def actor_critic(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(observation)
        return self.actor(features), self.critic(features).squeeze(-1)
