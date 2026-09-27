import torch
from torch import nn


class CNN16x32FC512x128Policy(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...], actions: int) -> None:
        super().__init__()
        if observation_shape != (4, 84, 84):
            raise ValueError("CNN16x32FC512x128Policy requires four 84 x 84 frames")
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor_critic(observation)[0]

    def actor_critic(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single = observation.ndim == 3
        features = self.features(observation.unsqueeze(0) if single else observation)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return (logits.squeeze(0), value.squeeze(0)) if single else (logits, value)
