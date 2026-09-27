from .base import Environment
from .cartpole import CartPoleEnvironment
from .cartpole_pixels import CartPolePixelsEnvironment


def make_environment(name: str) -> Environment:
    if name == "cartpole":
        return CartPoleEnvironment()
    if name == "cartpole_pixels":
        return CartPolePixelsEnvironment()
    raise ValueError(f"Unknown environment: {name}")


__all__ = ["Environment", "make_environment"]
