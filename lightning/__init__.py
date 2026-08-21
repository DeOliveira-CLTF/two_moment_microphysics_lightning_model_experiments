"""Diagnósticos reutilizáveis de atividade elétrica e relâmpagos."""

from .lpi import DEFAULT_W_THRESHOLD, LPIResult, compute_epsilon, compute_lpi_star
from .mccaul import McCaulResult, compute_mccaul

__all__ = [
    "DEFAULT_W_THRESHOLD",
    "LPIResult",
    "McCaulResult",
    "compute_epsilon",
    "compute_lpi_star",
    "compute_mccaul",
]
