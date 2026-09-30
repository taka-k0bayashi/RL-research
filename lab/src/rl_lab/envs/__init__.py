from .base import Environment
from .cartpole import CartPoleEnvironment
from .cartpole_pixels import CartPolePixelsEnvironment
from .point_maze import PointMazeEnvironment


def make_environment(name: str) -> Environment:
    if name == "cartpole":
        return CartPoleEnvironment()
    if name == "cartpole_pixels":
        return CartPolePixelsEnvironment()
    if name == "point_maze":
        return PointMazeEnvironment()
    if name == "minihack_room_5x5":
        from .minihack import MiniHackRoom5x5Environment

        return MiniHackRoom5x5Environment()
    raise ValueError(f"Unknown environment: {name}")


__all__ = ["Environment", "make_environment"]
