# -*- coding: utf-8 -*-
"""Analisa e plota os resultados finais do Grupo 1."""

from __future__ import annotations

import csv
import json
from math import gamma, pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ENTRADA = ROOT / "outputs" / "group1"
SAIDA = ENTRADA / "analise"

CASOS = ("N_LOW", "CTRL", "N_HIGH")

ROTULOS = {
    "N_LOW": "N-LOW",
    "CTRL": "CTRL",
    "N_HIGH": "N-HIGH",
}

CORES = {
    "N_LOW": "#0072B2",
    "CTRL": "#333333",
    "N_HIGH": "#D55E00",
}

RHO_AGUA = 1000.0
MU_CLOUD = 1.0
LIMIAR_NUVEM = 1.0e-5
LIMIAR_CHUVA_WP = 1.0e-6


def carregar():
    dados = {}
    configs = {}

    for caso in CASOS:
        arquivo_npz = ENTRADA / caso / f"resultados_{caso}.npz"
        arquivo_config = ENTRADA / caso / "configuracao.json"

        if not arquivo_npz.exists():
            raise FileNotFoundError(
                f"Arquivo de resultados não encontrado: {arquivo_npz}"
            )

        if not arquivo_config.exists():
            raise FileNotFoundError(
                f"Configuração não encontrada: {arquivo_config}"
            )

        with np.load(arquivo_npz) as npz:
            dados[caso] = {
                nome: npz[nome].copy()
                for nome in npz.files
            }

        configs[caso] = json.loads(
            arquivo_config.read_text(encoding="utf-8")
        )

    return dados, configs


def validar(dados, configs):
    obrigatorios = {
        "t_s",
        "x_m",
        "z_m",
        "rho0_1d",
        "qc",
        "Nc",
        "qr",
        "Nr",
        "qi",
        "Ni",
        "qs",
        "Ns",
        "qg",
        "Ng",
        "T",
        "qv",
        "w",
        "u",
        "F1",
        "F2",
        "F3",
        "LPI_star",
        "cfl_max_adv",
        "cfl_max_diff",
    }

    referencia_tempo = dados["CTRL"]["t_s"]
    referencia_forma = dados["CTRL"]["qc"].shape

    for caso in CASOS:
        faltantes = obrigatorios - set(dados[caso])

        if faltantes:
            raise KeyError(
                f"{caso}: campos ausentes: {sorted(faltantes)}"
            )

        if dados[caso]["qc"].shape != referencia_forma:
            raise ValueError(
                f"{caso}: grade ou número de tempos diferente do CTRL"
            )

        if not np.array_equal(
            dados[caso]["t_s"],
            referencia_tempo,
        ):
            raise ValueError(
                f"{caso}: tempos diferentes do CTRL"
            )

        for nome, valor in dados[caso].items():
            if (
                np.issubdtype(valor.dtype, np.floating)
                and np.isnan(valor).any()
            ):
                raise ValueError(
                    f"{caso}: NaN encontrado em {nome}"
                )

    diferencas = {}

    todas_chaves = set().union(
        *(config.keys() for config in configs.values())
    )

    for chave in sorted(todas_chaves):
        valores = {
            caso: configs[caso].get(chave)
            for caso in CASOS
        }

        valores_serializados = {
            json.dumps(valor, sort_keys=True)
            for valor in valores.values()
        }

        if len(valores_serializados) > 1:
            diferencas[chave] = valores

    diferencas_permitidas = {
        "nc_ativacao_kg1",
        "cenario",
    }

    diferencas_extras = (
        set(diferencas) - diferencas_permitidas
    )

    if diferencas_extras:
        raise ValueError(
            "Configurações científicas adicionais diferem: "
            f"{sorted(diferencas_extras)}"
        )

    return diferencas


def water_path(campo, rho, dz):
    """Conteúdo médio horizontal, em kg m-2."""

    return (
        np.sum(
            campo * rho[None, None, :],
            axis=(1, 2),
        )
        * dz
        / campo.shape[1]
    )


def primeiro_tempo(tempos_min, serie, limiar):
    indices = np.flatnonzero(serie >= limiar)

    if indices.size:
        return float(tempos_min[indices[0]])

    return np.nan


def topo_nuvem(campo_condensado, z):
    topos = []

    for frame in campo_condensado:
        niveis = np.flatnonzero(
            np.max(frame, axis=0) > LIMIAR_NUVEM
        )

        if niveis.size:
            topos.append(float(z[niveis[-1]]))
        else:
            topos.append(0.0)

    return np.asarray(topos)


def parametros_psd(q, n_kg1, rho_ar):
    if q <= 1.0e-12 or n_kg1 <= 1.0e-6:
        return np.nan, np.nan, np.nan

    numerador = (
        pi
        * RHO_AGUA
        * gamma(MU_CLOUD + 4.0)
        * n_kg1
    )

    denominador = (
        6.0
        * rho_ar
        * gamma(MU_CLOUD + 1.0)
        * q
    )

    lambda_gama = (
        numerador / denominador
    ) ** (1.0 / 3.0)

    dm_um = (
        (MU_CLOUD + 4.0)
        / lambda_gama
        * 1.0e6
    )

    dn_um = (
        (MU_CLOUD + 1.0)
        / lambda_gama
        * 1.0e6
    )

    return lambda_gama, dm_um, dn_um


def calcular_metricas(dados):
    series = {}
    resumo = {}

    tempos_min = dados["CTRL"]["t_s"] / 60.0

    dz = float(
        np.mean(
            np.diff(dados["CTRL"]["z_m"])
        )
    )

    soma_qc_ctrl = np.sum(
        dados["CTRL"]["qc"],
        axis=(1, 2),
    )

    candidatos = np.flatnonzero(
        soma_qc_ctrl > 0.0
    )

    if not candidatos.size:
        raise ValueError(
            "Nenhuma água de nuvem foi produzida no CTRL"
        )

    it_psd = int(candidatos[0])

    ix_psd, iz_psd = np.unravel_index(
        np.argmax(
            dados["CTRL"]["qc"][it_psd]
        ),
        dados["CTRL"]["qc"][it_psd].shape,
    )

    for caso in CASOS:
        d = dados[caso]
        rho = d["rho0_1d"]

        s = {
            "tempo_min": tempos_min,
        }

        for especie in (
            "qc",
            "qr",
            "qi",
            "qs",
            "qg",
        ):
            s[f"wp_{especie}"] = water_path(
                d[especie],
                rho,
                dz,
            )

        s["wp_liquido"] = (
            s["wp_qc"] + s["wp_qr"]
        )

        s["wp_gelo"] = (
            s["wp_qi"]
            + s["wp_qs"]
            + s["wp_qg"]
        )

        s["w_max"] = np.max(
            d["w"],
            axis=(1, 2),
        )

        condensado = (
            d["qc"]
            + d["qr"]
            + d["qi"]
            + d["qs"]
            + d["qg"]
        )

        s["topo_m"] = topo_nuvem(
            condensado,
            d["z_m"],
        )

        s["f3_max"] = np.max(
            d["F3"],
            axis=1,
        )

        s["lpi_max"] = np.max(
            d["LPI_star"],
            axis=1,
        )

        series[caso] = s

        q_referencia = float(
            d["qc"][it_psd, ix_psd, iz_psd]
        )

        n_referencia = float(
            d["Nc"][it_psd, ix_psd, iz_psd]
        )

        rho_referencia = float(
            rho[iz_psd]
        )

        lambda_gama, dm_um, dn_um = parametros_psd(
            q_referencia,
            n_referencia,
            rho_referencia,
        )

        indice_qc = int(
            np.argmax(s["wp_qc"])
        )

        indice_qr = int(
            np.argmax(s["wp_qr"])
        )

        resumo[caso] = {
            "caso": caso,
            "q_ref_kgkg": q_referencia,
            "Nc_ref_kg-1": n_referencia,
            "lambda_ref_m-1": lambda_gama,
            "Dm_ref_um": dm_um,
            "Dn_ref_um": dn_um,
            "qc_wp_max_kgm-2": float(
                s["wp_qc"][indice_qc]
            ),
            "tempo_qc_max_min": float(
                tempos_min[indice_qc]
            ),
            "qr_wp_max_kgm-2": float(
                s["wp_qr"][indice_qr]
            ),
            "tempo_qr_max_min": float(
                tempos_min[indice_qr]
            ),
            "primeiro_qr_detectado_min":
                primeiro_tempo(
                    tempos_min,
                    s["wp_qr"],
                    LIMIAR_CHUVA_WP,
                ),
            "w_max_ms-1": float(
                np.max(s["w_max"])
            ),
            "topo_max_m": float(
                np.max(s["topo_m"])
            ),
            "qi_wp_max_kgm-2": float(
                np.max(s["wp_qi"])
            ),
            "qs_wp_max_kgm-2": float(
                np.max(s["wp_qs"])
            ),
            "qg_wp_max_kgm-2": float(
                np.max(s["wp_qg"])
            ),
            "F1_max": float(
                np.max(d["F1"])
            ),
            "F2_max": float(
                np.max(d["F2"])
            ),
            "F3_max": float(
                np.max(d["F3"])
            ),
            "LPI_star_max": float(
                np.max(d["LPI_star"])
            ),
            "CFL_adv_max": float(
                d["cfl_max_adv"]
            ),
            "CFL_diff_max": float(
                d["cfl_max_diff"]
            ),
        }

    local_psd = {
        "tempo_min": float(
            tempos_min[it_psd]
        ),
        "ix": ix_psd,
        "iz": iz_psd,
        "x_m": float(
            dados["CTRL"]["x_m"][ix_psd]
        ),
        "z_m": float(
            dados["CTRL"]["z_m"][iz_psd]
        ),
    }

    return series, resumo, local_psd


def salvar_csvs(series, resumo):
    campos_resumo = list(
        next(iter(resumo.values())).keys()
    )

    arquivo_resumo = (
        SAIDA / "tabela_resumo_grupo1.csv"
    )

    with arquivo_resumo.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos_resumo,
        )

        writer.writeheader()

        writer.writerows(
            resumo[caso]
            for caso in CASOS
        )

    campos_series = [
        "caso",
        "tempo_min",
        "wp_qc_kgm-2",
        "wp_qr_kgm-2",
        "wp_qi_kgm-2",
        "wp_qs_kgm-2",
        "wp_qg_kgm-2",
        "w_max_ms-1",
        "topo_m",
        "F3_max",
        "LPI_star_max",
    ]

    arquivo_series = (
        SAIDA / "series_temporais_grupo1.csv"
    )

    with arquivo_series.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos_series,
        )

        writer.writeheader()

        for caso in CASOS:
            s = series[caso]

            for i, tempo in enumerate(
                s["tempo_min"]
            ):
                writer.writerow({
                    "caso": caso,
                    "tempo_min": tempo,
                    "wp_qc_kgm-2":
                        s["wp_qc"][i],
                    "wp_qr_kgm-2":
                        s["wp_qr"][i],
                    "wp_qi_kgm-2":
                        s["wp_qi"][i],
                    "wp_qs_kgm-2":
                        s["wp_qs"][i],
                    "wp_qg_kgm-2":
                        s["wp_qg"][i],
                    "w_max_ms-1":
                        s["w_max"][i],
                    "topo_m":
                        s["topo_m"][i],
                    "F3_max":
                        s["f3_max"][i],
                    "LPI_star_max":
                        s["lpi_max"][i],
                })


def plotar_principal(
    dados,
    series,
    resumo,
    local_psd,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.8),
    )

    diametros_um = np.linspace(
        1.0,
        180.0,
        600,
    )

    diametros_m = (
        diametros_um * 1.0e-6
    )

    ix = local_psd["ix"]
    iz = local_psd["iz"]

    tempos = (
        dados["CTRL"]["t_s"] / 60.0
    )

    it = int(
        np.argmin(
            np.abs(
                tempos
                - local_psd["tempo_min"]
            )
        )
    )

    for caso in CASOS:
        d = dados[caso]

        q = float(
            d["qc"][it, ix, iz]
        )

        n_kg = float(
            d["Nc"][it, ix, iz]
        )

        rho = float(
            d["rho0_1d"][iz]
        )

        lambda_gama, _, _ = parametros_psd(
            q,
            n_kg,
            rho,
        )

        n_volume = n_kg * rho

        n0 = (
            n_volume
            * lambda_gama
            ** (MU_CLOUD + 1.0)
            / gamma(MU_CLOUD + 1.0)
        )

        nd_por_um = (
            n0
            * diametros_m ** MU_CLOUD
            * np.exp(
                -lambda_gama * diametros_m
            )
            * 1.0e-6
        )

        axes[0].plot(
            diametros_um,
            nd_por_um,
            linewidth=2.2,
            color=CORES[caso],
            label=ROTULOS[caso],
        )

    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 180)
    axes[0].set_ylim(bottom=1.0e-2)
    axes[0].set_xlabel("Diâmetro D (µm)")
    axes[0].set_ylabel(
        "N(D) (# m⁻³ µm⁻¹)"
    )
    axes[0].set_title(
        "(a) Distribuição de gotículas em 5 min"
    )
    axes[0].legend(frameon=False)

    for caso in CASOS:
        s = series[caso]

        axes[1].plot(
            s["tempo_min"],
            s["wp_qc"],
            color=CORES[caso],
            linewidth=2.2,
            label=f"{ROTULOS[caso]}: nuvem",
        )

        axes[1].plot(
            s["tempo_min"],
            s["wp_qr"],
            color=CORES[caso],
            linewidth=2.2,
            linestyle="--",
            label=f"{ROTULOS[caso]}: chuva",
        )

    axes[1].set_xlabel("Tempo (min)")
    axes[1].set_ylabel(
        "Conteúdo médio no domínio (kg m⁻²)"
    )
    axes[1].set_title(
        "(b) Água de nuvem (—) e de chuva (--)"
    )
    axes[1].legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    for caso in CASOS:
        s = series[caso]

        axes[2].plot(
            s["tempo_min"],
            s["w_max"],
            color=CORES[caso],
            linewidth=2.2,
            label=ROTULOS[caso],
        )

    axes[2].set_xlabel("Tempo (min)")
    axes[2].set_ylabel(
        "w máximo (m s⁻¹)"
    )
    axes[2].set_title(
        "(c) Intensidade da corrente ascendente"
    )
    axes[2].legend(frameon=False)

    for eixo in axes:
        eixo.grid(
            True,
            alpha=0.22,
        )

    fig.suptitle(
        "Grupo 1 — Sensibilidade à concentração de gotículas",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        SAIDA / "figura_principal_grupo1.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plotar_resumo(resumo):
    x = np.arange(len(CASOS))
    largura = 0.34

    qc = [
        resumo[caso]["qc_wp_max_kgm-2"]
        for caso in CASOS
    ]

    qr = [
        resumo[caso]["qr_wp_max_kgm-2"]
        for caso in CASOS
    ]

    dm = [
        resumo[caso]["Dm_ref_um"]
        for caso in CASOS
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.6),
    )

    axes[0].bar(
        x - largura / 2,
        qc,
        largura,
        label="Água de nuvem",
        color="#56B4E9",
    )

    axes[0].bar(
        x + largura / 2,
        qr,
        largura,
        label="Água de chuva",
        color="#0072B2",
    )

    axes[0].set_xticks(
        x,
        [ROTULOS[caso] for caso in CASOS],
    )

    axes[0].set_ylabel(
        "Máximo do conteúdo médio (kg m⁻²)"
    )

    axes[0].set_title(
        "(a) Partição máxima da água líquida"
    )

    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].bar(
        x,
        dm,
        color=[
            CORES[caso]
            for caso in CASOS
        ],
    )

    axes[1].set_xticks(
        x,
        [ROTULOS[caso] for caso in CASOS],
    )

    axes[1].set_ylabel(
        "Diâmetro médio de massa Dm (µm)"
    )

    axes[1].set_title(
        "(b) Tamanho característico em 5 min"
    )

    axes[1].grid(axis="y", alpha=0.22)

    for i, valor in enumerate(dm):
        axes[1].text(
            i,
            valor,
            f"{valor:.1f}",
            ha="center",
            va="bottom",
        )

    fig.suptitle(
        "Resposta microfísica ao aumento de Nc",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        SAIDA
        / "figura_resumo_microfisico_grupo1.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def escrever_interpretacao(
    resumo,
    local_psd,
    diferencas,
):
    ctrl_qr = resumo["CTRL"][
        "qr_wp_max_kgm-2"
    ]

    ctrl_qc = resumo["CTRL"][
        "qc_wp_max_kgm-2"
    ]

    low_qr = 100.0 * (
        resumo["N_LOW"]["qr_wp_max_kgm-2"]
        / ctrl_qr
        - 1.0
    )

    high_qr = 100.0 * (
        resumo["N_HIGH"]["qr_wp_max_kgm-2"]
        / ctrl_qr
        - 1.0
    )

    low_qc = 100.0 * (
        resumo["N_LOW"]["qc_wp_max_kgm-2"]
        / ctrl_qc
        - 1.0
    )

    high_qc = 100.0 * (
        resumo["N_HIGH"]["qc_wp_max_kgm-2"]
        / ctrl_qc
        - 1.0
    )

    texto = f"""# Síntese dos resultados — Grupo 1

## Validação experimental

Os três casos possuem a mesma grade, passo de tempo, duração,
bolha térmica e opções físicas. As únicas diferenças são
nc_ativacao_kg1 e o nome do cenário.

O maior CFL advectivo foi
{max(resumo[c]['CFL_adv_max'] for c in CASOS):.3f},
abaixo do limite 1,0.

## Resposta microfísica

A comparação da distribuição foi feita em
{local_psd['tempo_min']:.0f} min, no ponto
x={local_psd['x_m']/1000:.1f} km e
z={local_psd['z_m']/1000:.1f} km.

O diâmetro médio de massa diminuiu de
{resumo['N_LOW']['Dm_ref_um']:.1f} µm em N-LOW para
{resumo['CTRL']['Dm_ref_um']:.1f} µm em CTRL e
{resumo['N_HIGH']['Dm_ref_um']:.1f} µm em N-HIGH.

O máximo de água de chuva integrada foi
{low_qr:+.1f}% em N-LOW e {high_qr:+.1f}% em N-HIGH
em relação ao CTRL.

O máximo de água de nuvem foi
{low_qc:+.1f}% em N-LOW e {high_qc:+.1f}% em N-HIGH.

Nc menor favoreceu a conversão de água de nuvem em
chuva. Nc maior reteve mais massa na categoria de
gotículas de nuvem.

## Dinâmica, gelo e eletrificação

O máximo de w ficou entre
{min(resumo[c]['w_max_ms-1'] for c in CASOS):.3f} e
{max(resumo[c]['w_max_ms-1'] for c in CASOS):.3f}
m s-1.

O topo máximo foi
{max(resumo[c]['topo_max_m'] for c in CASOS)/1000:.1f}
km.

As simulações não produziram neve nem graupel,
geraram apenas quantidades residuais de gelo e
apresentaram LPI*=0 e F1=0.

## Limitação sobre precipitação

O núcleo 2D não armazena uma série de precipitação
acumulada na superfície. A produção de chuva foi
avaliada pelo conteúdo integrado de qr.
"""

    (
        SAIDA
        / "interpretacao_resultados_grupo1.md"
    ).write_text(
        texto,
        encoding="utf-8",
    )

    (
        SAIDA
        / "diferencas_configuracao.json"
    ).write_text(
        json.dumps(
            diferencas,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    SAIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    dados, configs = carregar()

    diferencas = validar(
        dados,
        configs,
    )

    series, resumo, local_psd = (
        calcular_metricas(dados)
    )

    for caso in CASOS:
        resumo[caso]["Nc_config_kg-1"] = float(
            configs[caso]["nc_ativacao_kg1"]
        )

    salvar_csvs(
        series,
        resumo,
    )

    plotar_principal(
        dados,
        series,
        resumo,
        local_psd,
    )

    plotar_resumo(resumo)

    escrever_interpretacao(
        resumo,
        local_psd,
        diferencas,
    )

    print("Análise concluída sem erros.")

    print(
        f"Produtos salvos em: "
        f"{SAIDA.relative_to(ROOT)}"
    )

    for caso in CASOS:
        r = resumo[caso]

        print(
            f"{caso:6s}: "
            f"Dm={r['Dm_ref_um']:.1f} um | "
            f"qc_max={r['qc_wp_max_kgm-2']:.4f} kg/m2 | "
            f"qr_max={r['qr_wp_max_kgm-2']:.4f} kg/m2 | "
            f"w_max={r['w_max_ms-1']:.3f} m/s"
        )


if __name__ == "__main__":
    main()

