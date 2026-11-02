"""DragonScope: a debugger for the memory of BDH models."""

from dragonscope.report import Case, ForgettingReport, Probe, forgetting_report
from dragonscope.toolkit import ForgetResult, MemoryCapture, Step, capture
from dragonscope.train import load_model

__version__ = "0.1.0"


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
