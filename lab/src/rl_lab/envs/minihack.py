import gymnasium as gym
import minihack  # noqa: F401  # Registers MiniHack environments with Gymnasium.
import numpy as np
from numpy.typing import NDArray


class MiniHackRoom5x5Environment:
    def __init__(self) -> None:
        self._environment = gym.make(
            "MiniHack-Room-5x5-v0", observation_keys=("chars_crop",)
        )
        action_space = self._environment.action_space
        if not isinstance(action_space, gym.spaces.Discrete):
            self._environment.close()
            raise TypeError("MiniHack Room requires a discrete action space")
        self.observation_shape = (81,)
        self.action_size = int(action_space.n)

    @staticmethod
    def _vector(observation: dict[str, NDArray[np.uint8]]) -> NDArray[np.float32]:
        return (observation["chars_crop"].reshape(-1).astype(np.float32) - 32) / 96

    def reset(self, seed: int) -> NDArray[np.float32]:
        observation, _ = self._environment.reset(seed=seed)
        return self._vector(observation)

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool]:
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        return self._vector(observation), float(reward), terminated or truncated

    def close(self) -> None:
        self._environment.close()
