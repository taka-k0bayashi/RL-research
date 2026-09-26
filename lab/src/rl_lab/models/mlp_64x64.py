import torch
from torch import nn


class MLP64x64Policy(nn.Module):
    def __init__(self, observations: int, actions: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, actions),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)
