"""Diagnosticos reutilizaveis de atividade eletrica e relampagos."""

from .lpi import DEFAULT_W_THRESHOLD, LPIResult, compute_epsilon, compute_lpi_star
from .mccaul import McCaulResult, compute_mccaul
from .diagnosticos_2d import diagnosticar_relampagos_2d, resumir_diagnosticos_2d

__all__ = [
    "DEFAULT_W_THRESHOLD",
    "LPIResult",
    "McCaulResult",
    "compute_epsilon",
    "compute_lpi_star",
    "compute_mccaul",
    "diagnosticar_relampagos_2d",
    "resumir_diagnosticos_2d",
]