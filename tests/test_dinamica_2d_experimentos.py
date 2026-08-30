# -*- coding: utf-8 -*-
"""Testes curtos para as extensoes experimentais do nucleo 2D."""

import numpy as np

from dinamica_2d import ConfiguracaoDinamica2D, criar_estado, diagnosticar_cfl


def test_bolha_permanece_no_centro_com_nx_reduzido():
    config = ConfiguracaoDinamica2D(nx=30, nz=45, bolha_k=3.0)
    estado = criar_estado(config)
    ix, _iz = np.unravel_index(np.argmax(estado.thp), estado.thp.shape)
    x_pico = estado.x[ix]
    x_centro = 0.5 * (estado.x[0] + estado.x[-1])
    assert abs(x_pico - x_centro) <= config.dx


def test_warm_adiciona_quatro_kelvin_em_temperatura_real():
    ctrl = criar_estado(ConfiguracaoDinamica2D(delta_t_ambiente_k=0.0))
    warm = criar_estado(ConfiguracaoDinamica2D(delta_t_ambiente_k=4.0))
    np.testing.assert_allclose(warm.T_env_1d - ctrl.T_env_1d, 4.0, atol=1.0e-12)


def test_cfl_detecta_violacao_advectiva():
    config = ConfiguracaoDinamica2D(nx=10, nz=10, dt=1.5, dx=100.0, dz=100.0)
    estado = criar_estado(config)
    u = np.full((config.nx, config.nz), 40.0)
    w = np.full((config.nx, config.nz), 40.0)
    cfl = diagnosticar_cfl(estado, config, u, w)
    assert cfl["adveccao"] > config.cfl_limite
    assert not cfl["estavel_adveccao"]
