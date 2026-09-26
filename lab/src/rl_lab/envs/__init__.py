from .base import Environment
from .cartpole import CartPoleEnvironment


def make_environment(name: str) -> Environment:
    if name == "cartpole":
        return CartPoleEnvironment()
    raise ValueError(f"Unknown environment: {name}")


__all__ = ["Environment", "make_environment"]
