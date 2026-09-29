import gymnasium as gym
import gymnasium_robotics
import numpy as np
from numpy.typing import NDArray

gym.register_envs(gymnasium_robotics)


class PointMazeEnvironment:
    continuous = True

    def __init__(self) -> None:
        # Sparse reward, and the episode ends when the goal is reached.
        self._environment = gym.make("PointMaze_UMaze-v3", continuing_task=False)
        action_space = self._environment.action_space
        if not isinstance(action_space, gym.spaces.Box) or action_space.shape is None:
            self._environment.close()
            raise TypeError("PointMaze requires a continuous Box action space")
        self._low = action_space.low
        self._high = action_space.high
        self.observation_shape = (6,)
        self.action_size = action_space.shape[0]

    @staticmethod
    def _flatten(observation: dict[str, NDArray[np.float64]]) -> NDArray[np.float32]:
        return np.concatenate(
            (observation["observation"], observation["desired_goal"])
        ).astype(np.float32)

    def reset(self, seed: int) -> NDArray[np.float32]:
        observation, _ = self._environment.reset(seed=seed)
        return self._flatten(observation)

    def step(self, action: list[float]) -> tuple[NDArray[np.float32], float, bool]:
        observation, reward, terminated, truncated, _ = self._environment.step(
            np.clip(action, self._low, self._high)
        )
        return self._flatten(observation), float(reward), terminated or truncated

    def close(self) -> None:
        self._environment.close()
