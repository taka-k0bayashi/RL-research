from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Environment(Protocol):
    observation_shape: tuple[int, ...]
    action_size: int

    def reset(self, seed: int) -> NDArray[np.float32]: ...

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool]: ...

    def close(self) -> None: ...
