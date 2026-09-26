import torch
from torch import nn


class LinearPolicy(nn.Module):
    def __init__(self, observations: int, actions: int) -> None:
        super().__init__()
        self.network = nn.Linear(observations, actions)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)
