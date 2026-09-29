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
    raise ValueError(f"Unknown environment: {name}")


__all__ = ["Environment", "make_environment"]
