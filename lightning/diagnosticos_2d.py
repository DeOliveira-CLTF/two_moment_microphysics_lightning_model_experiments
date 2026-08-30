# -*- coding: utf-8 -*-
"""Aplicacao de McCaul e LPI* a cada coluna do modelo dinamico 2D.

Os modulos ``mccaul.py`` e ``lpi.py`` continuam sendo os diagnosticos de uma
unica coluna vertical. Este arquivo apenas percorre as dimensoes tempo e x do
modelo 2D e organiza os resultados em matrizes [tempo, x].
"""

import numpy as np

from .mccaul import compute_mccaul
from .lpi import compute_lpi_star


_CAMPOS_NECESSARIOS = ("T", "w", "qc", "qr", "qi", "qs", "qg")


def diagnosticar_relampagos_2d(resultado, w_threshold_m_s=0.5):
    """Calcula McCaul F1/F2/F3 e LPI* para todas as colunas e tempos.

    Parameters
    ----------
    resultado : dict
        Saida de ``dinamica_2d.rodar_thompson_2d``.
    w_threshold_m_s : float
        Limiar de movimento ascendente usado no LPI*.

    Returns
    -------
    dict
        Matrizes ``f1``, ``f2``, ``f3``, ``lpi_star`` e campos auxiliares,
        todas com forma ``(nt, nx)``.
    """

    frames = resultado["frames"]
    faltantes = [nome for nome in _CAMPOS_NECESSARIOS if nome not in frames]
    if faltantes:
        raise KeyError(
            "resultado dinamico nao possui os campos necessarios: "
            + ", ".join(faltantes)
        )

    z = np.asarray(resultado["z_m"], dtype=float)
    rho = np.asarray(resultado["rho0_1d"], dtype=float)
    T = np.asarray(frames["T"], dtype=float)

    if T.ndim != 3:
        raise ValueError("frames['T'] deve ter forma (nt, nx, nz)")

    nt, nx, nz = T.shape
    if z.size != nz or rho.size != nz:
        raise ValueError("z_m/rho0_1d incompatíveis com a dimensao vertical")

    saida = {
        "t_s": np.asarray(frames["t"], dtype=float),
        "x_m": np.asarray(resultado["x_m"], dtype=float),
        "f1": np.full((nt, nx), np.nan),
        "f2": np.full((nt, nx), np.nan),
        "f3": np.full((nt, nx), np.nan),
        "lpi_star": np.full((nt, nx), np.nan),
        "mccaul_valido": np.zeros((nt, nx), dtype=bool),
        "lpi_valido": np.zeros((nt, nx), dtype=bool),
        "z_minus15_m": np.full((nt, nx), np.nan),
        "w_minus15_m_s": np.full((nt, nx), np.nan),
        "qg_minus15_kgkg": np.full((nt, nx), np.nan),
        "h_0c_m": np.full((nt, nx), np.nan),
        "h_minus20c_m": np.full((nt, nx), np.nan),
        "epsilon_max": np.full((nt, nx), np.nan),
        "epsilon_mean": np.full((nt, nx), np.nan),
    }

    for it in range(nt):
        for ix in range(nx):
            perfil_T = frames["T"][it, ix, :]
            perfil_w = frames["w"][it, ix, :]

            mc = compute_mccaul(
                z_m=z,
                temperature_k=perfil_T,
                rho_kg_m3=rho,
                w_m_s=perfil_w,
                qi_kgkg=frames["qi"][it, ix, :],
                qs_kgkg=frames["qs"][it, ix, :],
                qg_kgkg=frames["qg"][it, ix, :],
            )
            saida["f1"][it, ix] = mc.f1
            saida["f2"][it, ix] = mc.f2
            saida["f3"][it, ix] = mc.f3
            saida["mccaul_valido"][it, ix] = mc.valid_f1
            saida["z_minus15_m"][it, ix] = mc.z_minus15_m
            saida["w_minus15_m_s"][it, ix] = mc.w_minus15_m_s
            saida["qg_minus15_kgkg"][it, ix] = mc.qg_minus15_kgkg

            lp = compute_lpi_star(
                z_m=z,
                temperature_k=perfil_T,
                w_m_s=perfil_w,
                qc_kgkg=frames["qc"][it, ix, :],
                qr_kgkg=frames["qr"][it, ix, :],
                qi_kgkg=frames["qi"][it, ix, :],
                qs_kgkg=frames["qs"][it, ix, :],
                qg_kgkg=frames["qg"][it, ix, :],
                w_threshold_m_s=w_threshold_m_s,
            )
            saida["lpi_star"][it, ix] = lp.lpi_star
            saida["lpi_valido"][it, ix] = lp.valid
            saida["h_0c_m"][it, ix] = lp.h_0c_m
            saida["h_minus20c_m"][it, ix] = lp.h_minus20c_m
            saida["epsilon_max"][it, ix] = lp.max_epsilon
            saida["epsilon_mean"][it, ix] = lp.mean_epsilon

    return saida


def _nanmax_seguro(a):
    a = np.asarray(a, dtype=float)
    if np.all(np.isnan(a)):
        return np.nan
    return float(np.nanmax(a))


def _nanmean_seguro(a):
    a = np.asarray(a, dtype=float)
    if np.all(np.isnan(a)):
        return np.nan
    return float(np.nanmean(a))


def resumir_diagnosticos_2d(diagnosticos):
    """Retorna um resumo escalar simples sem substituir os campos 2D."""

    return {
        "f1_max": _nanmax_seguro(diagnosticos["f1"]),
        "f2_max": _nanmax_seguro(diagnosticos["f2"]),
        "f3_max": _nanmax_seguro(diagnosticos["f3"]),
        "lpi_star_max": _nanmax_seguro(diagnosticos["lpi_star"]),
        "f3_mean_valid": _nanmean_seguro(diagnosticos["f3"]),
        "lpi_star_mean_valid": _nanmean_seguro(diagnosticos["lpi_star"]),
        "frac_mccaul_valido": float(np.mean(diagnosticos["mccaul_valido"])),
        "frac_lpi_valido": float(np.mean(diagnosticos["lpi_valido"])),
    }
