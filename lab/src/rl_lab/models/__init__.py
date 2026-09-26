from torch import nn

from .linear import LinearPolicy
from .mlp_32 import MLP32Policy
from .mlp_64x64 import MLP64x64Policy

MODELS: dict[str, type[nn.Module]] = {
    "linear": LinearPolicy,
    "mlp_32": MLP32Policy,
    "mlp_64x64": MLP64x64Policy,
}


def make_model(name: str, observations: int, actions: int) -> nn.Module:
    try:
        model = MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model: {name}") from None
    return model(observations, actions)


__all__ = ["make_model"]
