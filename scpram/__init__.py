"""scPRAM perturbation prediction, with lazy optional plotting imports."""

from importlib import import_module

from .models import SCPRAM
from .transport import OTConfig, OTConvergenceError
from .version import __version__

__all__ = ["SCPRAM", "OTConfig", "OTConvergenceError", "__version__"]


def __getattr__(name):
    if name in {"models", "dataset", "utils", "data_process", "evaluate"}:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
