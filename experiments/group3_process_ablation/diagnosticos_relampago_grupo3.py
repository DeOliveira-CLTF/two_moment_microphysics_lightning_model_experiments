# -*- coding: utf-8 -*-
"""
diagnosticos_relampago_grupo3.py
====================================

Aplica McCaul et al. (2009) e o Lightning Potential Index adaptado
(LPI*) aos campos 2D (tempo x x x z) salvos por `executar_grupo3.py`,
seguindo EXATAMENTE as formulas do Plano de Experimentos (secoes 3.2 e
3.3), reproduzidas abaixo apenas para referencia:

    F1 = 0.042 * [w * (1000 q_g)]  em -15 graus C
    F2 = 0.20 * integral rho (q_g + q_s + q_i) dz
    F3 = 0.95 F1 + 0.05 F2

    LPI* = [1/(H_-20 - H_0)] * integral w^2 g(w) epsilon dz , entre H_0 e H_-20
    q_L = q_c + q_r
    q_F = q_g [ sqrt(q_i q_g)/(q_i+q_g) + sqrt(q_s q_g)/(q_s+q_g) ]
    epsilon = 2 sqrt(q_L q_F) / (q_L + q_F)
    g(w) = 1 para w > 0.5 m/s; 0 caso contrario

Nota de integridade cientifica
--------------------------------
O plano indica que o repositorio ja contem `lightning/mccaul.py` e
`lightning/lpi.py` (testados) e uma ponte `lightning/diagnosticos_2d.py`
para aplicar essas formulas em tempo x x. Esses arquivos NAO foram
fornecidos para a montagem deste driver, entao este modulo os
REIMPLEMENTA de forma independente, apenas a partir das formulas
publicadas no plano, para que o Grupo 3 funcione de forma autocontida.

Antes de gerar a Tabela 2 final, confirme se os valores aqui batem com
`lightning/diagnosticos_2d.py` do repositorio (mesmo commit da secao
5.5) -- se um dos dois modulos diferir, use o do repositorio como
fonte de verdade e trate este arquivo como um fallback/checagem
cruzada, nao como substituto silencioso.

f1 e f2 (filtros horizontais da formulacao completa do LPI) NAO sao
aplicados aqui (f1=f2=1), conforme a mesma limitacao documentada no
plano para `lightning/lpi.py` -- por isso o resultado continua sendo
chamado LPI*.
"""

import numpy as np

# numpy >= 2.0 renomeou trapz -> trapezoid; mantem compatibilidade com
# ambas as versoes (o restante do repositorio usa numpy diretamente).
_trapz = getattr(np, "trapezoid", None) or np.trapz

T0_K = 273.15


def _altura_isoterma(z, T_col, T_alvo_C):
    """Altura (m) onde a coluna cruza T_alvo_C, por interpolacao linear,
    varrendo de baixo para cima e parando na PRIMEIRA passagem -- mesma
    convencao usada em `lightning/mccaul.py` segundo o plano (secao
    3.2: 'se a isoterma estiver fora do dominio, retorna NaN, sem
    extrapolacao')."""
    T_alvo = T_alvo_C + T0_K
    diff = T_col - T_alvo
    sinal = np.sign(diff)
    cruzamentos = np.where(np.diff(sinal) != 0)[0]
    if len(cruzamentos) == 0:
        return np.nan
    k = cruzamentos[0]
    T1, T2 = T_col[k], T_col[k + 1]
    z1, z2 = z[k], z[k + 1]
    if T2 == T1:
        return z1
    frac = (T_alvo - T1) / (T2 - T1)
    return z1 + frac * (z2 - z1)


def _interp_na_altura(z, campo_col, altura):
    if np.isnan(altura):
        return np.nan
    return float(np.interp(altura, z, campo_col))


def mccaul_F3_coluna(z, T_col, w_col, qg_col, qs_col, qi_col, rho_col):
    """F1, F2, F3 (McCaul et al. 2009) para UMA coluna, em um instante."""
    h_m15 = _altura_isoterma(z, T_col, -15.0)
    if np.isnan(h_m15):
        return np.nan, np.nan, np.nan

    w_m15 = _interp_na_altura(z, w_col, h_m15)
    qg_m15 = _interp_na_altura(z, qg_col, h_m15)
    F1 = 0.042 * (w_m15 * (1000.0 * qg_m15))

    integrando = rho_col * (qg_col + qs_col + qi_col)
    F2 = 0.20 * _trapz(integrando, z)

    F3 = 0.95 * F1 + 0.05 * F2
    return F1, F2, F3


def lpi_estrela_coluna(z, T_col, w_col, qc_col, qr_col, qi_col, qs_col, qg_col):
    """LPI* para UMA coluna, em um instante (nucleo vertical apenas;
    f1=f2=1, ver docstring do modulo)."""
    h0 = _altura_isoterma(z, T_col, 0.0)
    hm20 = _altura_isoterma(z, T_col, -20.0)
    if np.isnan(h0) or np.isnan(hm20) or hm20 <= h0:
        return np.nan

    mascara = (z >= h0) & (z <= hm20)
    if mascara.sum() < 2:
        return np.nan

    z_sub = z[mascara]
    w_sub = w_col[mascara]
    qL = qc_col[mascara] + qr_col[mascara]
    qi_s, qs_s, qg_s = qi_col[mascara], qs_col[mascara], qg_col[mascara]

    with np.errstate(divide="ignore", invalid="ignore"):
        termo_i = np.sqrt(qi_s * qg_s) / (qi_s + qg_s)
        termo_s = np.sqrt(qs_s * qg_s) / (qs_s + qg_s)
        termo_i = np.nan_to_num(termo_i, nan=0.0)
        termo_s = np.nan_to_num(termo_s, nan=0.0)
        qF = qg_s * (termo_i + termo_s)

        epsilon = 2.0 * np.sqrt(qL * qF) / (qL + qF)
        epsilon = np.nan_to_num(epsilon, nan=0.0)

    g_w = (w_sub > 0.5).astype(float)
    integrando = (w_sub ** 2) * g_w * epsilon
    lpi = _trapz(integrando, z_sub) / (hm20 - h0)
    return float(lpi)


def diagnosticos_2d_para_caso(frames, z, rho0_1d):
    """Aplica McCaul (F3) e LPI* a TODOS os instantes salvos e a TODAS
    as colunas x de um caso, replicando a funcao de
    `lightning/diagnosticos_2d.py` descrita no plano ('aplicacao de
    McCaul/LPI* em tempo x x').

    Parametros
    ----------
    frames : dict retornado por rodar_thompson_2d()["frames"], contendo
             arrays de shape (nt, nx, nz) para T, w, qc, qr, qi, qs, qg.
    z : array (nz,)
    rho0_1d : array (nz,), densidade de referencia do ambiente

    Retorna
    -------
    F3_txx, LPI_txx : arrays (nt, nx)
    """
    T = frames["T"]
    w = frames["w"]
    qc, qr = frames["qc"], frames["qr"]
    qi, qs, qg = frames["qi"], frames["qs"], frames["qg"]

    nt, nx, _ = T.shape
    F3_txx = np.full((nt, nx), np.nan)
    LPI_txx = np.full((nt, nx), np.nan)

    for it in range(nt):
        for ix in range(nx):
            _, _, F3 = mccaul_F3_coluna(
                z, T[it, ix], w[it, ix], qg[it, ix], qs[it, ix], qi[it, ix],
                rho0_1d,
            )
            F3_txx[it, ix] = F3

            LPI_txx[it, ix] = lpi_estrela_coluna(
                z, T[it, ix], w[it, ix], qc[it, ix], qr[it, ix],
                qi[it, ix], qs[it, ix], qg[it, ix],
            )

    return F3_txx, LPI_txx
