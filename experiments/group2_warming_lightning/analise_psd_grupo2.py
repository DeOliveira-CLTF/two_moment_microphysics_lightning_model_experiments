#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analise offline das distribuicoes de tamanho do Grupo 2
========================================================

Este script NAO reroda o modelo. Ele usa os arquivos resultados_*.npz ja
gerados pelo Grupo 2 e calcula, a partir dos dois momentos prognosticados
(q e N):

- concentracao numerica de goticulas e graupel;
- parametro lambda da distribuicao gama;
- diametro medio ponderado pela massa Dm;
- estatisticas no dominio e na camada de fase mista (0 a -20 C);
- diagnosticos interpolados exatamente na isoterma de -15 C;
- massa e numero integrados de graupel;
- massa media por particula de graupel;
- diagnosticos no ponto de qg maximo;
- checagens de qualidade dos campos N;
- CSVs e figuras para comparacao entre os seis casos do Grupo 2.

A formulacao de Dm e a mesma usada no Grupo 1:

    lambda^3 =
        pi * rho_x * Gamma(mu+4) * N
        --------------------------------
        6 * rho_ar * Gamma(mu+1) * q

    Dm = (mu + 4) / lambda

Sugestao de local para salvar este arquivo:
    experiments/group2_warming_lightning/analise_psd_grupo2.py

Execucao a partir da raiz do repositorio:
    python experiments/group2_warming_lightning/analise_psd_grupo2.py

Para indicar manualmente uma pasta contendo os .npz:
    python experiments/group2_warming_lightning/analise_psd_grupo2.py \
        --entrada "C:/caminho/para/os/arquivos"

Autor: script de pos-processamento para o projeto
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# 1. CASOS
# ============================================================================

CASOS = (
    "CTRL",
    "DYN_PLUS",
    "WARM_QV",
    "WARM_QV_DYN_PLUS",
    "WARM_RH",
    "WARM_RH_DYN_PLUS",
)

ROTULOS = {
    "CTRL": "CTRL",
    "DYN_PLUS": "DYN PLUS",
    "WARM_QV": "WARM QV",
    "WARM_QV_DYN_PLUS": "WARM QV + DYN PLUS",
    "WARM_RH": "WARM RH",
    "WARM_RH_DYN_PLUS": "WARM RH + DYN PLUS",
}

CORES = {
    "CTRL": "black",
    "DYN_PLUS": "dimgray",
    "WARM_QV": "darkorange",
    "WARM_QV_DYN_PLUS": "firebrick",
    "WARM_RH": "royalblue",
    "WARM_RH_DYN_PLUS": "purple",
}

ESTILOS = {
    "CTRL": "-",
    "DYN_PLUS": "--",
    "WARM_QV": "-",
    "WARM_QV_DYN_PLUS": "--",
    "WARM_RH": "-",
    "WARM_RH_DYN_PLUS": "--",
}

T0_K = 273.15
T_MINUS20_K = 253.15
T_MINUS15_C = -15.0

# Mesmos tempos usados frequentemente na discussao do Grupo 2.
TEMPOS_DESTAQUE_MIN = (5.0, 10.0, 15.0, 20.0)


# ============================================================================
# 2. LOCALIZACAO DA RAIZ DO REPOSITORIO
# ============================================================================

def localizar_raiz_repo() -> Path:
    """Procura a raiz do repositorio a partir do script e do cwd."""

    candidatos = []

    arquivo = Path(__file__).resolve()
    candidatos.extend([arquivo.parent, *arquivo.parents])

    cwd = Path.cwd().resolve()
    candidatos.extend([cwd, *cwd.parents])

    vistos = set()

    for candidato in candidatos:
        candidato = candidato.resolve()

        if candidato in vistos:
            continue

        vistos.add(candidato)

        if (
            (candidato / "microfisica").is_dir()
            and (candidato / "dinamica_2d").is_dir()
            and (candidato / "experiments").is_dir()
        ):
            return candidato

    raise RuntimeError(
        "Nao foi possivel localizar a raiz do repositorio. "
        "Salve este script dentro do repositorio "
        "two_moment_microphysics_lightning_model_experiments."
    )


ROOT = localizar_raiz_repo()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Importamos diretamente as constantes do codigo atual para manter o
# pos-processamento consistente com a simulacao.
from microfisica.constantes import (  # noqa: E402
    MU_CLOUD,
    MU_SNOW,
    NMIN,
    QMIN,
    rho_g,
    rho_w,
)

MU_GRAUPEL = MU_SNOW


# ============================================================================
# 3. UTILITARIOS
# ============================================================================

def inferir_caso(caminho: Path) -> str | None:
    """
    Infere o caso a partir de nomes como:
        resultados_CTRL.npz
        resultados_CTRL(1).npz
    """

    nome = caminho.stem

    if not nome.startswith("resultados_"):
        return None

    nome = nome[len("resultados_"):]

    # Remove sufixos de download, por exemplo "(1)".
    nome = re.sub(r"\(\d+\)$", "", nome)

    if nome in CASOS:
        return nome

    return None


def localizar_arquivos(entrada: Path) -> dict[str, Path]:
    """Localiza recursivamente um arquivo NPZ para cada caso."""

    entrada = entrada.resolve()

    if not entrada.exists():
        raise FileNotFoundError(f"Pasta de entrada nao existe: {entrada}")

    candidatos = list(entrada.rglob("resultados_*.npz"))

    encontrados: dict[str, list[Path]] = {caso: [] for caso in CASOS}

    for caminho in candidatos:
        caso = inferir_caso(caminho)

        if caso is not None:
            encontrados[caso].append(caminho)

    selecionados = {}

    for caso in CASOS:
        arquivos = encontrados[caso]

        if not arquivos:
            raise FileNotFoundError(
                f"Nao encontrei arquivo resultados_{caso}.npz "
                f"em {entrada} nem em subpastas."
            )

        # Prioriza o nome exato e, depois, o arquivo mais recente.
        arquivos = sorted(
            arquivos,
            key=lambda p: (
                p.name == f"resultados_{caso}.npz",
                p.stat().st_mtime,
            ),
            reverse=True,
        )

        selecionados[caso] = arquivos[0]

        if len(arquivos) > 1:
            print(
                f"[AVISO] {caso}: encontrei {len(arquivos)} arquivos. "
                f"Usando: {arquivos[0]}"
            )

    return selecionados


def escrever_csv(caminho: Path, linhas: list[dict]) -> None:
    """Escreve uma lista de dicionarios em CSV."""

    if not linhas:
        return

    colunas = []

    for linha in linhas:
        for chave in linha:
            if chave not in colunas:
                colunas.append(chave)

    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas)


def mediana_segura(campo, mascara=None) -> float:
    campo = np.asarray(campo, dtype=float)

    if mascara is not None:
        mascara = np.asarray(mascara, dtype=bool)
        valores = campo[mascara]
    else:
        valores = campo.ravel()

    valores = valores[np.isfinite(valores)]

    if valores.size == 0:
        return np.nan

    return float(np.median(valores))


def media_ponderada_segura(campo, pesos, mascara=None) -> float:
    campo = np.asarray(campo, dtype=float)
    pesos = np.asarray(pesos, dtype=float)

    valido = np.isfinite(campo) & np.isfinite(pesos) & (pesos > 0.0)

    if mascara is not None:
        valido &= np.asarray(mascara, dtype=bool)

    if not np.any(valido):
        return np.nan

    soma_pesos = np.sum(pesos[valido])

    if soma_pesos <= 0.0:
        return np.nan

    return float(
        np.sum(campo[valido] * pesos[valido])
        / soma_pesos
    )


def fracao_segura(numerador: int, denominador: int) -> float:
    if denominador <= 0:
        return np.nan

    return float(numerador / denominador)


# ============================================================================
# 4. DISTRIBUICAO GAMA E DIAMETRO MEDIO DE MASSA
# ============================================================================

def diametro_medio_massa_array(
    q: np.ndarray,
    N: np.ndarray,
    rho_ar_1d: np.ndarray,
    rho_x: float,
    mu: float,
    q_min: float,
    n_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula Dm para um campo 3D (tempo, x, z).

    Retorna
    -------
    Dm_m : ndarray
        Diametro medio ponderado pela massa [m].
    valido : ndarray bool
        Mascara onde q e N sao fisicamente utilizaveis.
    """

    q = np.asarray(q, dtype=float)
    N = np.asarray(N, dtype=float)
    rho_ar_1d = np.asarray(rho_ar_1d, dtype=float)

    if q.shape != N.shape:
        raise ValueError("q e N precisam ter a mesma forma.")

    if q.ndim != 3:
        raise ValueError(
            f"Esperado campo 3D (tempo, x, z), recebido {q.shape}."
        )

    if q.shape[-1] != rho_ar_1d.size:
        raise ValueError(
            "Ultima dimensao de q/N nao coincide com rho0_1d."
        )

    rho_ar = np.broadcast_to(
        rho_ar_1d[None, None, :],
        q.shape,
    )

    valido = (
        np.isfinite(q)
        & np.isfinite(N)
        & np.isfinite(rho_ar)
        & (rho_ar > 0.0)
        & (q > q_min)
        & (N > n_min)
    )

    Dm = np.full(q.shape, np.nan, dtype=float)

    if not np.any(valido):
        return Dm, valido

    numerador = (
        np.pi
        * rho_x
        * math.gamma(mu + 4.0)
        * N[valido]
    )

    denominador = (
        6.0
        * rho_ar[valido]
        * math.gamma(mu + 1.0)
        * q[valido]
    )

    lam = (numerador / denominador) ** (1.0 / 3.0)

    Dm[valido] = (mu + 4.0) / lam

    return Dm, valido


def diametro_medio_massa_escalar(
    q: float,
    N: float,
    rho_ar: float,
    rho_x: float,
    mu: float,
    q_min: float,
    n_min: float,
) -> float:
    """Versao escalar de Dm."""

    valores = (q, N, rho_ar)

    if not all(np.isfinite(v) for v in valores):
        return np.nan

    if q <= q_min or N <= n_min or rho_ar <= 0.0:
        return np.nan

    lam = (
        (
            np.pi
            * rho_x
            * math.gamma(mu + 4.0)
            * N
        )
        /
        (
            6.0
            * rho_ar
            * math.gamma(mu + 1.0)
            * q
        )
    ) ** (1.0 / 3.0)

    return float((mu + 4.0) / lam)


# ============================================================================
# 5. INTERPOLACAO NA ISOTERMA
# ============================================================================

def interpolar_isoterma(
    z_m: np.ndarray,
    temperatura_c: np.ndarray,
    campos: dict[str, np.ndarray],
    alvo_c: float,
):
    """
    Replica a logica usada pelos diagnosticos de lightning do repositorio:

    - sem extrapolacao;
    - interpolacao linear;
    - se houver mais de um cruzamento, usa o de menor altitude.
    """

    z_m = np.asarray(z_m, dtype=float)
    temperatura_c = np.asarray(temperatura_c, dtype=float)

    exato = np.flatnonzero(
        np.isclose(
            temperatura_c,
            alvo_c,
            rtol=0.0,
            atol=1.0e-10,
        )
    )

    if exato.size:
        k = int(exato[0])

        return float(z_m[k]), {
            nome: float(np.asarray(campo)[k])
            for nome, campo in campos.items()
        }

    delta = temperatura_c - alvo_c

    for superior in range(1, z_m.size):
        inferior = superior - 1

        if delta[inferior] * delta[superior] < 0.0:
            frac = (
                -delta[inferior]
                / (delta[superior] - delta[inferior])
            )

            z_alvo = (
                z_m[inferior]
                + frac * (z_m[superior] - z_m[inferior])
            )

            valores = {}

            for nome, campo in campos.items():
                campo = np.asarray(campo, dtype=float)

                valores[nome] = float(
                    campo[inferior]
                    + frac * (
                        campo[superior]
                        - campo[inferior]
                    )
                )

            return float(z_alvo), valores

    return None


# ============================================================================
# 6. ANALISE DE UM CASO
# ============================================================================

def analisar_caso(
    caso: str,
    caminho: Path,
    qc_min: float,
    qg_min: float,
    n_min: float,
) -> tuple[list[dict], dict]:

    print()
    print("=" * 78)
    print(f"ANALISANDO {caso}")
    print(caminho)
    print("=" * 78)

    obrigatorios = {
        "t_s",
        "x_m",
        "z_m",
        "rho0_1d",
        "T",
        "qc",
        "Nc",
        "qg",
        "Ng",
        "w",
    }

    with np.load(caminho, allow_pickle=False) as npz:
        faltantes = obrigatorios - set(npz.files)

        if faltantes:
            raise KeyError(
                f"{caso}: faltam campos no NPZ: {sorted(faltantes)}"
            )

        dados = {
            nome: np.asarray(npz[nome], dtype=float).copy()
            for nome in obrigatorios
        }

        qg_m15_salvo = (
            np.asarray(
                npz["lightning_qg_minus15_kgkg"],
                dtype=float,
            ).copy()
            if "lightning_qg_minus15_kgkg" in npz.files
            else None
        )

        w_m15_salvo = (
            np.asarray(
                npz["lightning_w_minus15_m_s"],
                dtype=float,
            ).copy()
            if "lightning_w_minus15_m_s" in npz.files
            else None
        )

    t_s = dados["t_s"]
    x_m = dados["x_m"]
    z_m = dados["z_m"]
    rho = dados["rho0_1d"]

    T = dados["T"]
    qc = dados["qc"]
    Nc = dados["Nc"]
    qg = dados["qg"]
    Ng = dados["Ng"]
    w = dados["w"]

    nt = t_s.size
    nx = x_m.size
    nz = z_m.size

    forma_esperada = (nt, nx, nz)

    for nome in ("T", "qc", "Nc", "qg", "Ng", "w"):
        if dados[nome].shape != forma_esperada:
            raise ValueError(
                f"{caso}: {nome} tem forma {dados[nome].shape}; "
                f"esperado {forma_esperada}."
            )

    dx = float(np.mean(np.diff(x_m)))
    dz = float(np.mean(np.diff(z_m)))
    area_celula = dx * dz

    rho2d = rho[None, :]

    # ----------------------------------------------------------------------
    # Dm em toda a grade
    # ----------------------------------------------------------------------

    Dmc, valido_c = diametro_medio_massa_array(
        q=qc,
        N=Nc,
        rho_ar_1d=rho,
        rho_x=rho_w,
        mu=MU_CLOUD,
        q_min=qc_min,
        n_min=n_min,
    )

    Dmg, valido_g = diametro_medio_massa_array(
        q=qg,
        N=Ng,
        rho_ar_1d=rho,
        rho_x=rho_g,
        mu=MU_GRAUPEL,
        q_min=qg_min,
        n_min=n_min,
    )

    # ----------------------------------------------------------------------
    # Diagnosticos temporais
    # ----------------------------------------------------------------------

    linhas = []

    qg15_recalculado = np.full((nt, nx), np.nan)
    w15_recalculado = np.full((nt, nx), np.nan)

    for it in range(nt):

        tempo_min = float(t_s[it] / 60.0)

        fm = (
            (T[it] <= T0_K)
            & (T[it] >= T_MINUS20_K)
        )

        # --------------------------
        # Agua de nuvem
        # --------------------------

        mascara_qc = qc[it] > qc_min
        mascara_c = valido_c[it]

        mascara_c_fm = mascara_c & fm

        pesos_qc = rho2d * np.maximum(qc[it], 0.0)

        nc_med = mediana_segura(Nc[it], mascara_c)

        nc_qcw = media_ponderada_segura(
            Nc[it],
            pesos_qc,
            mascara_c,
        )

        dmc_med_um = (
            mediana_segura(Dmc[it], mascara_c) * 1.0e6
        )

        dmc_qcw_um = (
            media_ponderada_segura(
                Dmc[it],
                pesos_qc,
                mascara_c,
            )
            * 1.0e6
        )

        nc_fm_med = mediana_segura(
            Nc[it],
            mascara_c_fm,
        )

        nc_fm_qcw = media_ponderada_segura(
            Nc[it],
            pesos_qc,
            mascara_c_fm,
        )

        dmc_fm_med_um = (
            mediana_segura(
                Dmc[it],
                mascara_c_fm,
            )
            * 1.0e6
        )

        dmc_fm_qcw_um = (
            media_ponderada_segura(
                Dmc[it],
                pesos_qc,
                mascara_c_fm,
            )
            * 1.0e6
        )

        # Integracao em area 2D.
        # Unidade de massa: kg por metro na dimensao transversal omitida.
        # Unidade de numero: particulas por metro.
        massa_qc_kg_m = float(
            np.sum(
                np.where(
                    mascara_c,
                    rho2d * np.maximum(qc[it], 0.0),
                    0.0,
                )
            )
            * area_celula
        )

        numero_c_m1 = float(
            np.sum(
                np.where(
                    mascara_c,
                    rho2d * np.maximum(Nc[it], 0.0),
                    0.0,
                )
            )
            * area_celula
        )

        # Qualidade de Nc: celulas com massa de nuvem relevante,
        # mas N <= 0.
        n_qc = int(np.count_nonzero(mascara_qc))
        n_nc_invalido = int(
            np.count_nonzero(
                mascara_qc
                & (
                    ~np.isfinite(Nc[it])
                    | (Nc[it] <= 0.0)
                )
            )
        )

        frac_nc_invalido = fracao_segura(
            n_nc_invalido,
            n_qc,
        )

        # --------------------------
        # Graupel
        # --------------------------

        mascara_qg = qg[it] > qg_min
        mascara_g = valido_g[it]
        mascara_g_fm = mascara_g & fm

        pesos_qg = rho2d * np.maximum(qg[it], 0.0)

        ng_med = mediana_segura(
            Ng[it],
            mascara_g,
        )

        ng_qgw = media_ponderada_segura(
            Ng[it],
            pesos_qg,
            mascara_g,
        )

        dmg_med_mm = (
            mediana_segura(
                Dmg[it],
                mascara_g,
            )
            * 1.0e3
        )

        dmg_qgw_mm = (
            media_ponderada_segura(
                Dmg[it],
                pesos_qg,
                mascara_g,
            )
            * 1.0e3
        )

        ng_fm_med = mediana_segura(
            Ng[it],
            mascara_g_fm,
        )

        dmg_fm_qgw_mm = (
            media_ponderada_segura(
                Dmg[it],
                pesos_qg,
                mascara_g_fm,
            )
            * 1.0e3
        )

        massa_qg_kg_m = float(
            np.sum(
                np.where(
                    mascara_g,
                    rho2d * np.maximum(qg[it], 0.0),
                    0.0,
                )
            )
            * area_celula
        )

        numero_g_m1 = float(
            np.sum(
                np.where(
                    mascara_g,
                    rho2d * np.maximum(Ng[it], 0.0),
                    0.0,
                )
            )
            * area_celula
        )

        if numero_g_m1 > 0.0:
            massa_media_g_kg = (
                massa_qg_kg_m / numero_g_m1
            )
            massa_media_g_mg = massa_media_g_kg * 1.0e6
        else:
            massa_media_g_mg = np.nan

        n_qg = int(np.count_nonzero(mascara_qg))

        n_ng_invalido = int(
            np.count_nonzero(
                mascara_qg
                & (
                    ~np.isfinite(Ng[it])
                    | (Ng[it] <= 0.0)
                )
            )
        )

        frac_ng_invalido = fracao_segura(
            n_ng_invalido,
            n_qg,
        )

        # --------------------------
        # Ponto do maximo de qg
        # --------------------------

        qg_frame = np.asarray(qg[it], dtype=float)

        if np.any(np.isfinite(qg_frame)):
            ix_qg, iz_qg = np.unravel_index(
                np.nanargmax(qg_frame),
                qg_frame.shape,
            )

            qg_max = float(qg_frame[ix_qg, iz_qg])
            ng_no_qgmax = float(Ng[it, ix_qg, iz_qg])
            dmg_no_qgmax = float(
                Dmg[it, ix_qg, iz_qg]
            )
            t_no_qgmax = float(T[it, ix_qg, iz_qg])
            w_no_qgmax = float(w[it, ix_qg, iz_qg])
            z_no_qgmax = float(z_m[iz_qg])
            x_no_qgmax = float(x_m[ix_qg])

            if (
                qg_max <= qg_min
                or ng_no_qgmax <= n_min
                or not np.isfinite(dmg_no_qgmax)
            ):
                ng_no_qgmax = np.nan
                dmg_no_qgmax_mm = np.nan
            else:
                dmg_no_qgmax_mm = (
                    dmg_no_qgmax * 1.0e3
                )
        else:
            qg_max = np.nan
            ng_no_qgmax = np.nan
            dmg_no_qgmax_mm = np.nan
            t_no_qgmax = np.nan
            w_no_qgmax = np.nan
            z_no_qgmax = np.nan
            x_no_qgmax = np.nan

        # --------------------------
        # Isoterma de -15 C
        # --------------------------

        qc15 = np.full(nx, np.nan)
        Nc15 = np.full(nx, np.nan)
        qg15 = np.full(nx, np.nan)
        Ng15 = np.full(nx, np.nan)
        w15 = np.full(nx, np.nan)
        rho15 = np.full(nx, np.nan)
        z15 = np.full(nx, np.nan)

        Dmc15 = np.full(nx, np.nan)
        Dmg15 = np.full(nx, np.nan)

        for ix in range(nx):

            resultado_iso = interpolar_isoterma(
                z_m=z_m,
                temperatura_c=T[it, ix, :] - 273.15,
                campos={
                    "qc": qc[it, ix, :],
                    "Nc": Nc[it, ix, :],
                    "qg": qg[it, ix, :],
                    "Ng": Ng[it, ix, :],
                    "w": w[it, ix, :],
                    "rho": rho,
                },
                alvo_c=T_MINUS15_C,
            )

            if resultado_iso is None:
                continue

            z_alvo, valores = resultado_iso

            z15[ix] = z_alvo
            qc15[ix] = valores["qc"]
            Nc15[ix] = valores["Nc"]
            qg15[ix] = valores["qg"]
            Ng15[ix] = valores["Ng"]
            w15[ix] = valores["w"]
            rho15[ix] = valores["rho"]

            Dmc15[ix] = diametro_medio_massa_escalar(
                q=qc15[ix],
                N=Nc15[ix],
                rho_ar=rho15[ix],
                rho_x=rho_w,
                mu=MU_CLOUD,
                q_min=qc_min,
                n_min=n_min,
            )

            Dmg15[ix] = diametro_medio_massa_escalar(
                q=qg15[ix],
                N=Ng15[ix],
                rho_ar=rho15[ix],
                rho_x=rho_g,
                mu=MU_GRAUPEL,
                q_min=qg_min,
                n_min=n_min,
            )

        qg15_recalculado[it] = qg15
        w15_recalculado[it] = w15

        mascara_c15 = (
            np.isfinite(Dmc15)
            & (qc15 > qc_min)
            & (Nc15 > n_min)
        )

        mascara_g15 = (
            np.isfinite(Dmg15)
            & (qg15 > qg_min)
            & (Ng15 > n_min)
        )

        pesos_c15 = np.maximum(rho15 * qc15, 0.0)
        pesos_g15 = np.maximum(rho15 * qg15, 0.0)

        nc15_med = mediana_segura(
            Nc15,
            mascara_c15,
        )

        dmc15_qcw_um = (
            media_ponderada_segura(
                Dmc15,
                pesos_c15,
                mascara_c15,
            )
            * 1.0e6
        )

        dmg15_qgw_mm = (
            media_ponderada_segura(
                Dmg15,
                pesos_g15,
                mascara_g15,
            )
            * 1.0e3
        )

        # Maximo bruto de qg em -15 C.
        qg15_finito = np.isfinite(qg15)

        if np.any(qg15_finito):
            ix15_max = int(
                np.nanargmax(qg15)
            )

            qg15_max = float(qg15[ix15_max])
            ng_qg15max = float(Ng15[ix15_max])
            w_qg15max = float(w15[ix15_max])

            if (
                qg15_max > qg_min
                and ng_qg15max > n_min
                and np.isfinite(Dmg15[ix15_max])
            ):
                dmg_qg15max_mm = (
                    float(Dmg15[ix15_max]) * 1.0e3
                )
            else:
                ng_qg15max = np.nan
                dmg_qg15max_mm = np.nan
        else:
            qg15_max = np.nan
            ng_qg15max = np.nan
            w_qg15max = np.nan
            dmg_qg15max_mm = np.nan

        # --------------------------
        # Linha temporal
        # --------------------------

        linhas.append(
            {
                "caso": caso,
                "tempo_min": tempo_min,

                "qc_max_g_kg": (
                    float(np.nanmax(qc[it])) * 1.0e3
                ),

                "Nc_mediana_nuvem_kg1": nc_med,
                "Nc_media_ponderada_qc_kg1": nc_qcw,
                "Dmc_mediana_nuvem_um": dmc_med_um,
                "Dmc_media_ponderada_qc_um": dmc_qcw_um,

                "Nc_mediana_fase_mista_kg1": nc_fm_med,
                "Nc_media_ponderada_qc_fase_mista_kg1": nc_fm_qcw,
                "Dmc_mediana_fase_mista_um": dmc_fm_med_um,
                "Dmc_media_ponderada_qc_fase_mista_um": dmc_fm_qcw_um,

                "massa_qc_integrada_kg_m": massa_qc_kg_m,
                "Nc_integrado_m1": numero_c_m1,
                "fracao_Nc_invalido_em_qc": frac_nc_invalido,

                "qg_max_g_kg": qg_max * 1.0e3,
                "Ng_no_qgmax_kg1": ng_no_qgmax,
                "Dmg_no_qgmax_mm": dmg_no_qgmax_mm,
                "T_no_qgmax_C": t_no_qgmax - 273.15,
                "w_no_qgmax_m_s": w_no_qgmax,
                "x_no_qgmax_m": x_no_qgmax,
                "z_no_qgmax_m": z_no_qgmax,

                "Ng_mediana_graupel_kg1": ng_med,
                "Ng_media_ponderada_qg_kg1": ng_qgw,
                "Dmg_mediana_graupel_mm": dmg_med_mm,
                "Dmg_media_ponderada_qg_mm": dmg_qgw_mm,

                "Ng_mediana_fase_mista_kg1": ng_fm_med,
                "Dmg_media_ponderada_qg_fase_mista_mm": dmg_fm_qgw_mm,

                "massa_qg_integrada_kg_m": massa_qg_kg_m,
                "Ng_integrado_m1": numero_g_m1,
                "massa_media_graupel_mg_particula": massa_media_g_mg,
                "fracao_Ng_invalido_em_qg": frac_ng_invalido,

                "Nc_mediana_m15_kg1": nc15_med,
                "Dmc_media_ponderada_qc_m15_um": dmc15_qcw_um,

                "qg_m15_max_g_kg": qg15_max * 1.0e3,
                "Ng_no_qg_m15_max_kg1": ng_qg15max,
                "Dmg_no_qg_m15_max_mm": dmg_qg15max_mm,
                "Dmg_media_ponderada_qg_m15_mm": dmg15_qgw_mm,
                "w_no_qg_m15_max_m_s": w_qg15max,
            }
        )

    # ----------------------------------------------------------------------
    # Verificacao da interpolacao contra os campos lightning ja salvos
    # ----------------------------------------------------------------------

    erro_qg15 = np.nan
    erro_w15 = np.nan

    if qg_m15_salvo is not None:
        valido = (
            np.isfinite(qg_m15_salvo)
            & np.isfinite(qg15_recalculado)
        )

        if np.any(valido):
            erro_qg15 = float(
                np.max(
                    np.abs(
                        qg_m15_salvo[valido]
                        - qg15_recalculado[valido]
                    )
                )
            )

    if w_m15_salvo is not None:
        valido = (
            np.isfinite(w_m15_salvo)
            & np.isfinite(w15_recalculado)
        )

        if np.any(valido):
            erro_w15 = float(
                np.max(
                    np.abs(
                        w_m15_salvo[valido]
                        - w15_recalculado[valido]
                    )
                )
            )

    # ----------------------------------------------------------------------
    # Resumo do caso
    # ----------------------------------------------------------------------

    linha_pico_qg = max(
        linhas,
        key=lambda l: (
            -np.inf
            if not np.isfinite(l["qg_max_g_kg"])
            else l["qg_max_g_kg"]
        ),
    )

    linha_pico_qg15 = max(
        linhas,
        key=lambda l: (
            -np.inf
            if not np.isfinite(l["qg_m15_max_g_kg"])
            else l["qg_m15_max_g_kg"]
        ),
    )

    resumo = {
        "caso": caso,
        "arquivo": str(caminho),

        "Nc_min_bruto_kg1": float(np.nanmin(Nc)),
        "Nc_max_bruto_kg1": float(np.nanmax(Nc)),
        "Ng_min_bruto_kg1": float(np.nanmin(Ng)),
        "Ng_max_bruto_kg1": float(np.nanmax(Ng)),

        "qg_max_global_g_kg": linha_pico_qg["qg_max_g_kg"],
        "tempo_qg_max_min": linha_pico_qg["tempo_min"],
        "Ng_no_qg_max_global_kg1": linha_pico_qg["Ng_no_qgmax_kg1"],
        "Dmg_no_qg_max_global_mm": linha_pico_qg["Dmg_no_qgmax_mm"],
        "massa_qg_integrada_no_tempo_qgmax_kg_m": (
            linha_pico_qg["massa_qg_integrada_kg_m"]
        ),
        "Ng_integrado_no_tempo_qgmax_m1": (
            linha_pico_qg["Ng_integrado_m1"]
        ),
        "massa_media_graupel_no_tempo_qgmax_mg_particula": (
            linha_pico_qg["massa_media_graupel_mg_particula"]
        ),

        "qg_m15_max_global_g_kg": linha_pico_qg15["qg_m15_max_g_kg"],
        "tempo_qg_m15_max_min": linha_pico_qg15["tempo_min"],
        "Ng_no_qg_m15_max_global_kg1": (
            linha_pico_qg15["Ng_no_qg_m15_max_kg1"]
        ),
        "Dmg_no_qg_m15_max_global_mm": (
            linha_pico_qg15["Dmg_no_qg_m15_max_mm"]
        ),
        "w_no_qg_m15_max_global_m_s": (
            linha_pico_qg15["w_no_qg_m15_max_m_s"]
        ),

        "erro_max_interpolacao_qg_m15_kgkg": erro_qg15,
        "erro_max_interpolacao_w_m15_m_s": erro_w15,

        "max_fracao_Nc_invalido_em_qc": float(
            np.nanmax(
                [
                    l["fracao_Nc_invalido_em_qc"]
                    for l in linhas
                ]
            )
        ),
        "max_fracao_Ng_invalido_em_qg": float(
            np.nanmax(
                [
                    l["fracao_Ng_invalido_em_qg"]
                    for l in linhas
                ]
            )
        ),
    }

    # Adiciona diagnosticos em tempos de destaque.
    for tempo_alvo in TEMPOS_DESTAQUE_MIN:

        linha = min(
            linhas,
            key=lambda l: abs(
                l["tempo_min"] - tempo_alvo
            ),
        )

        sufixo = str(int(tempo_alvo))

        resumo[f"tempo_real_{sufixo}min"] = (
            linha["tempo_min"]
        )

        resumo[f"Nc_qcw_{sufixo}min_kg1"] = (
            linha["Nc_media_ponderada_qc_kg1"]
        )

        resumo[f"Dmc_qcw_{sufixo}min_um"] = (
            linha["Dmc_media_ponderada_qc_um"]
        )

        resumo[f"Nc_fm_qcw_{sufixo}min_kg1"] = (
            linha[
                "Nc_media_ponderada_qc_fase_mista_kg1"
            ]
        )

        resumo[f"Dmc_fm_qcw_{sufixo}min_um"] = (
            linha[
                "Dmc_media_ponderada_qc_fase_mista_um"
            ]
        )

        resumo[f"qg_max_{sufixo}min_g_kg"] = (
            linha["qg_max_g_kg"]
        )

        resumo[f"Ng_qgmax_{sufixo}min_kg1"] = (
            linha["Ng_no_qgmax_kg1"]
        )

        resumo[f"Dmg_qgmax_{sufixo}min_mm"] = (
            linha["Dmg_no_qgmax_mm"]
        )

    print(
        f"qg max global = "
        f"{resumo['qg_max_global_g_kg']:.6g} g kg-1 "
        f"em {resumo['tempo_qg_max_min']:.1f} min"
    )

    print(
        f"Ng no qg max  = "
        f"{resumo['Ng_no_qg_max_global_kg1']:.6g} kg-1"
    )

    print(
        f"Dmg no qg max = "
        f"{resumo['Dmg_no_qg_max_global_mm']:.6g} mm"
    )

    if np.isfinite(erro_qg15):
        print(
            f"Erro max vs lightning qg(-15 C) salvo = "
            f"{erro_qg15:.3e} kg kg-1"
        )

    return linhas, resumo


# ============================================================================
# 7. FIGURAS
# ============================================================================

def organizar_series(linhas: list[dict]) -> dict[str, list[dict]]:
    series = {caso: [] for caso in CASOS}

    for linha in linhas:
        series[linha["caso"]].append(linha)

    for caso in CASOS:
        series[caso] = sorted(
            series[caso],
            key=lambda l: l["tempo_min"],
        )

    return series


def extrair(linhas, chave):
    return np.asarray(
        [linha[chave] for linha in linhas],
        dtype=float,
    )


def fazer_figura_principal(
    linhas: list[dict],
    caminho: Path,
) -> None:

    series = organizar_series(linhas)

    fig, eixos = plt.subplots(
        2,
        2,
        figsize=(11.69, 8.27),
        dpi=180,
    )

    ax1, ax2, ax3, ax4 = eixos.ravel()

    for caso in CASOS:

        linhas_caso = series[caso]

        t = extrair(
            linhas_caso,
            "tempo_min",
        )

        estilo = {
            "label": ROTULOS[caso],
            "color": CORES[caso],
            "linestyle": ESTILOS[caso],
            "linewidth": 2.0,
        }

        ax1.plot(
            t,
            extrair(
                linhas_caso,
                "Dmc_media_ponderada_qc_um",
            ),
            **estilo,
        )

        ax2.plot(
            t,
            extrair(
                linhas_caso,
                "Nc_media_ponderada_qc_kg1",
            ) / 1.0e8,
            **estilo,
        )

        ax3.plot(
            t,
            extrair(
                linhas_caso,
                "qg_max_g_kg",
            ),
            **estilo,
        )

        ax4.plot(
            t,
            extrair(
                linhas_caso,
                "Dmg_media_ponderada_qg_fase_mista_mm",
            ),
            **estilo,
        )

    ax1.set_title("(a) Goticulas de nuvem")
    ax1.set_ylabel(r"$D_{m,c}$ ponderado por $q_c$ ($\mu$m)")

    ax2.set_title("(b) Concentracao numerica de goticulas")
    ax2.set_ylabel(r"$N_c$ ponderado por $q_c$ ($10^8$ kg$^{-1}$)")

    ax3.set_title("(c) Maximo de graupel")
    ax3.set_ylabel(r"$q_{g,\max}$ (g kg$^{-1}$)")

    ax4.set_title("(d) Tamanho do graupel na fase mista")
    ax4.set_ylabel(r"$D_{m,g}$ ponderado por $q_g$ (mm)")

    for ax in eixos.ravel():
        ax.set_xlabel("Tempo (min)")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=10)

    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.suptitle(
        "Grupo 2 - Evolucao das concentracoes numericas e tamanhos de particulas",
        y=0.995,
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0.03, 0.03, 0.97, 0.91)
    )

    fig.savefig(
        caminho,
        bbox_inches="tight",
    )

    plt.close(fig)


def fazer_figura_fase_mista(
    linhas: list[dict],
    caminho: Path,
) -> None:

    series = organizar_series(linhas)

    fig, eixos = plt.subplots(
        2,
        2,
        figsize=(11.69, 8.27),
        dpi=180,
    )

    ax1, ax2, ax3, ax4 = eixos.ravel()

    for caso in CASOS:

        linhas_caso = series[caso]

        t = extrair(
            linhas_caso,
            "tempo_min",
        )

        estilo = {
            "label": ROTULOS[caso],
            "color": CORES[caso],
            "linestyle": ESTILOS[caso],
            "linewidth": 2.0,
        }

        ax1.plot(
            t,
            extrair(
                linhas_caso,
                "Dmc_media_ponderada_qc_fase_mista_um",
            ),
            **estilo,
        )

        ax2.plot(
            t,
            extrair(
                linhas_caso,
                "Dmc_media_ponderada_qc_m15_um",
            ),
            **estilo,
        )

        ax3.plot(
            t,
            extrair(
                linhas_caso,
                "qg_m15_max_g_kg",
            ),
            **estilo,
        )

        ax4.plot(
            t,
            extrair(
                linhas_caso,
                "massa_media_graupel_mg_particula",
            ),
            **estilo,
        )

    ax1.set_title("(a) Goticulas na fase mista")
    ax1.set_ylabel(r"$D_{m,c}$ ponderado por $q_c$ ($\mu$m)")

    ax2.set_title(r"(b) Goticulas na isoterma de $-15^\circ$C")
    ax2.set_ylabel(r"$D_{m,c}$ ponderado por $q_c$ ($\mu$m)")

    ax3.set_title(r"(c) Graupel na isoterma de $-15^\circ$C")
    ax3.set_ylabel(r"$q_{g,\max,-15}$ (g kg$^{-1}$)")

    ax4.set_title("(d) Massa media por particula de graupel")
    ax4.set_ylabel("Massa media (mg particula$^{-1}$)")

    for ax in eixos.ravel():
        ax.set_xlabel("Tempo (min)")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=10)

    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.suptitle(
        "Grupo 2 - Microfisica da fase mista e isoterma de -15 C",
        y=0.995,
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0.03, 0.03, 0.97, 0.91)
    )

    fig.savefig(
        caminho,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# 8. TABELA ENXUTA PARA DISCUSSAO CIENTIFICA
# ============================================================================

def gerar_tabela_discussao(
    resumos: list[dict],
) -> list[dict]:

    tabela = []

    for resumo in resumos:

        tabela.append(
            {
                "caso": resumo["caso"],

                "qg_max_g_kg": (
                    resumo["qg_max_global_g_kg"]
                ),

                "tempo_qg_max_min": (
                    resumo["tempo_qg_max_min"]
                ),

                "Ng_no_qgmax_kg1": (
                    resumo["Ng_no_qg_max_global_kg1"]
                ),

                "Dmg_no_qgmax_mm": (
                    resumo["Dmg_no_qg_max_global_mm"]
                ),

                "massa_media_graupel_mg_particula": (
                    resumo[
                        "massa_media_graupel_no_tempo_qgmax_mg_particula"
                    ]
                ),

                "Nc_qcw_10min_kg1": (
                    resumo["Nc_qcw_10min_kg1"]
                ),

                "Dmc_qcw_10min_um": (
                    resumo["Dmc_qcw_10min_um"]
                ),

                "Nc_fm_qcw_20min_kg1": (
                    resumo["Nc_fm_qcw_20min_kg1"]
                ),

                "Dmc_fm_qcw_20min_um": (
                    resumo["Dmc_fm_qcw_20min_um"]
                ),

                "qg_m15_max_g_kg": (
                    resumo["qg_m15_max_global_g_kg"]
                ),

                "Ng_no_qg_m15max_kg1": (
                    resumo[
                        "Ng_no_qg_m15_max_global_kg1"
                    ]
                ),

                "Dmg_no_qg_m15max_mm": (
                    resumo[
                        "Dmg_no_qg_m15_max_global_mm"
                    ]
                ),
            }
        )

    return tabela


# ============================================================================
# 9. IMPRESSAO FINAL
# ============================================================================

def imprimir_resumo(resumos: list[dict]) -> None:

    print()
    print("=" * 120)
    print("RESUMO PRINCIPAL")
    print("=" * 120)

    cabecalho = (
        f"{'CASO':24s}"
        f"{'qgmax(g/kg)':>14s}"
        f"{'t(min)':>9s}"
        f"{'Ng@qgmax':>15s}"
        f"{'Dmg(mm)':>12s}"
        f"{'Dmc10(um)':>13s}"
    )

    print(cabecalho)
    print("-" * len(cabecalho))

    for r in resumos:

        def fmt(valor, formato):
            if not np.isfinite(valor):
                return "nan"
            return format(valor, formato)

        print(
            f"{ROTULOS[r['caso']]:24s}"
            f"{fmt(r['qg_max_global_g_kg'], '.4g'):>14s}"
            f"{fmt(r['tempo_qg_max_min'], '.1f'):>9s}"
            f"{fmt(r['Ng_no_qg_max_global_kg1'], '.3e'):>15s}"
            f"{fmt(r['Dmg_no_qg_max_global_mm'], '.3f'):>12s}"
            f"{fmt(r['Dmc_qcw_10min_um'], '.2f'):>13s}"
        )


# ============================================================================
# 10. MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Pos-processamento offline de Nc, Ng e Dm "
            "para os experimentos do Grupo 2."
        )
    )

    parser.add_argument(
        "--entrada",
        type=Path,
        default=ROOT / "outputs" / "group2",
        help=(
            "Pasta que contem os resultados do Grupo 2. "
            "A busca pelos NPZ e recursiva."
        ),
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help=(
            "Pasta de saida. Padrao: <entrada>/analise_psd."
        ),
    )

    parser.add_argument(
        "--qc-min",
        type=float,
        default=1.0e-5,
        help=(
            "Limiar de qc [kg/kg] para considerar uma celula "
            "de nuvem no diagnostico de tamanho. "
            "Padrao = 1e-5, igual ao usado na analise do Grupo 1."
        ),
    )

    parser.add_argument(
        "--qg-min",
        type=float,
        default=1.0e-8,
        help=(
            "Limiar de qg [kg/kg] para diagnosticos de tamanho "
            "do graupel. Padrao = 1e-8."
        ),
    )

    parser.add_argument(
        "--n-min",
        type=float,
        default=float(NMIN),
        help=(
            "Limiar minimo de N [kg^-1]. "
            "Padrao = NMIN do esquema."
        ),
    )

    args = parser.parse_args()

    entrada = args.entrada.resolve()

    saida = (
        args.saida.resolve()
        if args.saida is not None
        else entrada / "analise_psd"
    )

    saida.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Raiz do repositorio: {ROOT}")
    print(f"Entrada:             {entrada}")
    print(f"Saida:               {saida}")
    print(f"qc_min:              {args.qc_min:.3e} kg/kg")
    print(f"qg_min:              {args.qg_min:.3e} kg/kg")
    print(f"N_min:               {args.n_min:.3e} kg^-1")

    arquivos = localizar_arquivos(
        entrada
    )

    todas_linhas = []
    resumos = []

    for caso in CASOS:

        linhas, resumo = analisar_caso(
            caso=caso,
            caminho=arquivos[caso],
            qc_min=args.qc_min,
            qg_min=args.qg_min,
            n_min=args.n_min,
        )

        todas_linhas.extend(
            linhas
        )

        resumos.append(
            resumo
        )

    # CSV completo, uma linha por caso e tempo.
    escrever_csv(
        saida / "series_temporais_psd_grupo2.csv",
        todas_linhas,
    )

    # Resumo por experimento.
    escrever_csv(
        saida / "resumo_psd_grupo2.csv",
        resumos,
    )

    # Tabela enxuta para leitura e discussao.
    tabela_discussao = gerar_tabela_discussao(
        resumos
    )

    escrever_csv(
        saida / "tabela_discussao_psd_grupo2.csv",
        tabela_discussao,
    )

    fazer_figura_principal(
        todas_linhas,
        saida / "psd_grupo2_temporal.png",
    )

    fazer_figura_fase_mista(
        todas_linhas,
        saida / "psd_grupo2_fase_mista.png",
    )

    imprimir_resumo(
        resumos
    )

    print()
    print("=" * 78)
    print("ARQUIVOS GERADOS")
    print("=" * 78)

    for nome in (
        "series_temporais_psd_grupo2.csv",
        "resumo_psd_grupo2.csv",
        "tabela_discussao_psd_grupo2.csv",
        "psd_grupo2_temporal.png",
        "psd_grupo2_fase_mista.png",
    ):
        print(saida / nome)

    print()
    print("Concluido sem rerodar as simulacoes.")


if __name__ == "__main__":
    main()
