import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class CartPoleEnvironment:
    def __init__(self) -> None:
        self._environment = gym.make("CartPole-v1")
        observation_space = self._environment.observation_space
        action_space = self._environment.action_space
        if (
            not isinstance(observation_space, gym.spaces.Box)
            or observation_space.shape is None
            or len(observation_space.shape) != 1
        ):
            self._environment.close()
            raise TypeError("CartPole requires a one-dimensional Box observation space")
        if not isinstance(action_space, gym.spaces.Discrete):
            self._environment.close()
            raise TypeError("CartPole requires a discrete action space")
        self.observation_shape = observation_space.shape
        self.action_size = int(action_space.n)

    def reset(self, seed: int) -> NDArray[np.float32]:
        observation, _ = self._environment.reset(seed=seed)
        return np.asarray(observation, dtype=np.float32)

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool]:
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            terminated or truncated,
        )

    def close(self) -> None:
        self._environment.close()
