import torch
from torch import nn


class TransformerD32H4FF64Policy(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...], actions: int) -> None:
        super().__init__()
        if len(observation_shape) != 1:
            raise ValueError(
                "TransformerD32H4FF64Policy requires vector observations"
            )
        features = observation_shape[0]
        self.embedding = nn.Linear(1, 32)
        self.positions = nn.Parameter(torch.empty(features, 32))
        self.encoder = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
        )
        self.actor = nn.Linear(32, actions)
        self.critic = nn.Linear(32, 1)
        nn.init.normal_(self.positions, std=0.02)

    def _features(self, observation: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(observation.unsqueeze(-1)) + self.positions
        return self.encoder(tokens).mean(dim=-2)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self._features(observation))

    def actor_critic(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(observation)
        return self.actor(features), self.critic(features).squeeze(-1)
