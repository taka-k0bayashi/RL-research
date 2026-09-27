from collections import deque

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from PIL import Image


class CartPolePixelsEnvironment:
    def __init__(self) -> None:
        environment = gym.make("CartPole-v1", render_mode="rgb_array")
        self._environment = gym.wrappers.AddRenderObservation(
            environment, render_only=True
        )
        self._frames: deque[NDArray[np.float32]] = deque(maxlen=4)

        action_space = self._environment.action_space
        if not isinstance(action_space, gym.spaces.Discrete):
            self._environment.close()
            raise TypeError("CartPole requires a discrete action space")
        self.observation_shape = (4, 84, 84)
        self.action_size = int(action_space.n)

    @staticmethod
    def _preprocess(observation: object) -> NDArray[np.float32]:
        return 1.0 - np.asarray(
            Image.fromarray(np.asarray(observation)[140:340])
            .convert("L")
            .resize((84, 84), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

    def _observation(self) -> NDArray[np.float32]:
        frames = np.stack(self._frames)
        differences = frames[1:] - frames[:-1]
        return np.concatenate((frames[-1:], differences))

    def reset(self, seed: int) -> NDArray[np.float32]:
        observation, _ = self._environment.reset(seed=seed)
        frame = self._preprocess(observation)
        self._frames.extend(frame.copy() for _ in range(4))
        return self._observation()

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool]:
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        self._frames.append(self._preprocess(observation))
        return (
            self._observation(),
            float(reward),
            terminated or truncated,
        )

    def close(self) -> None:
        self._environment.close()
