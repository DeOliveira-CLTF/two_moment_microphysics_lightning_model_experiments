# -*- coding: utf-8 -*-
"""Testes basicos do nucleo dinamico 2D."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dinamica_2d import ConfiguracaoDinamica2D, criar_estado, rodar_thompson_2d, velocidades


def test_criar_estado_dinamica_2d_shapes():
    config = ConfiguracaoDinamica2D(nx=8, nz=12, tempo_total_s=3.0, salvar_a_cada_s=1.5)
    estado = criar_estado(config)

    assert estado.thp.shape == (8, 12)
    assert estado.qc.shape == (8, 12)
    assert estado.p_pa_1d.shape == (12,)
    assert estado.rho0_1d.shape == (12,)

    u, w = velocidades(estado.psi, config.dx, config.dz)
    assert u.shape == (8, 12)
    assert w.shape == (8, 12)


def test_rodar_thompson_2d_curto_sem_nan():
    config = ConfiguracaoDinamica2D(
        nx=8,
        nz=12,
        tempo_total_s=3.0,
        salvar_a_cada_s=1.5,
        microfisica="thompson",
        iteracoes_poisson=3,
    )
    resultado = rodar_thompson_2d(config)
    frames = resultado["frames"]

    assert frames["t"].shape[0] >= 2
    assert frames["w"].shape[1:] == (8, 12)
    assert np.isfinite(frames["w"]).all()
    assert np.isfinite(frames["qc_qi"]).all()
