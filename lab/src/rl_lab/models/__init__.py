from torch import nn

from .cnn_16x32_fc128 import CNN16x32FC128Policy
from .cnn_16x32_fc512x128 import CNN16x32FC512x128Policy
from .linear import LinearPolicy
from .mlp_32 import MLP32Policy
from .mlp_64x64 import MLP64x64Policy
from .transformer_d32_h4_ff64 import TransformerD32H4FF64Policy

MODELS: dict[str, type[nn.Module]] = {
    "cnn_16x32_fc128": CNN16x32FC128Policy,
    "cnn_16x32_fc512x128": CNN16x32FC512x128Policy,
    "linear": LinearPolicy,
    "mlp_32": MLP32Policy,
    "mlp_64x64": MLP64x64Policy,
    "transformer_d32_h4_ff64": TransformerD32H4FF64Policy,
}


def make_model(
    name: str, observation_shape: tuple[int, ...], actions: int
) -> nn.Module:
    try:
        model = MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model: {name}") from None
    return model(observation_shape, actions)


__all__ = ["make_model"]
