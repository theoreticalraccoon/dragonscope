"""DragonScope: a debugger for the memory of BDH models."""

from dragonscope.report import Case, ForgettingReport, Probe, forgetting_report
from dragonscope.toolkit import ForgetResult, MemoryCapture, Step, capture

__version__ = "0.2.0"


def load_model(path, device: str = "cpu"):
    """Load a model saved by ``dragonscope.train``, for inference."""
    # Imported here so ``python -m dragonscope.train`` and ``.data`` run cleanly.
    from dragonscope.train import load_model as load

    return load(path, device)


def dashboard(memory, **options):
    """Open the live dashboard for a session. Needs ``pip install dragonscope[dashboard]``."""
    from dragonscope.dashboard import dashboard as open_dashboard

    return open_dashboard(memory, **options)


__all__ = [
    "Case",
    "ForgetResult",
    "ForgettingReport",
    "MemoryCapture",
    "Probe",
    "Step",
    "capture",
    "dashboard",
    "forgetting_report",
    "load_model",
]
