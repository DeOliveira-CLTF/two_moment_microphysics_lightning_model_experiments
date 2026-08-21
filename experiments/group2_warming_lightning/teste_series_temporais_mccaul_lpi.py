# -*- coding: utf-8 -*-
"""Teste temporal dos diagnósticos McCaul e LPI* no caso do Passo 3.

Este script integra apenas a microfísica já existente e diagnostica, em cada
instante salvo, F1/F2/F3 de McCaul e o LPI* da coluna. A velocidade vertical é
um perfil externo prescrito e não alimenta a microfísica.

Os valores são diagnósticos relativos: não representam uma taxa absoluta em
flashes por minuto sem calibração observacional.

Executar a partir da raiz do repositório:

    python experiments/group2_warming_lightning/teste_series_temporais_mccaul_lpi.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "group2"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lightning import compute_lpi_star, compute_mccaul
from microfisica.coluna_step3 import ColunaFaseMista


TEMPO_TOTAL_S = 1800.0
DT_S = 2.0
INTERVALO_SAIDA_S = 60.0
Z_PICO_W_M = 4000.0
SIGMA_W_M = 2000.0
W_PICO_M_S = 8.0


def executar_teste():
    """Integre o caso-base e retorne as séries diagnósticas calculadas."""
    coluna = ColunaFaseMista(
        nz=80,
        dz=100.0,
        T_base=293.0,
        p_base=95000.0,
    )
    k_base = int(1500.0 / coluna.dz)
    k_topo = int(6000.0 / coluna.dz)
    coluna.inserir_nuvem(
        k_base,
        k_topo,
        qc_valor=1.0e-3,
        Nc_valor=2.0e8,
    )

    # Configuração herdada do Passo 3 para exercitar graupel.
    k_fria = int(4500.0 / coluna.dz)
    coluna.qr[k_fria] = 5.0e-4
    coluna.Nr[k_fria] = 5.0e5

    # Updraft externo: é usado somente pelos diagnósticos de lightning.
    w_prescrito = W_PICO_M_S * np.exp(
        -0.5 * ((coluna.z - Z_PICO_W_M) / SIGMA_W_M) ** 2
    )
    historico = coluna.integrar(
        TEMPO_TOTAL_S,
        dt=DT_S,
        salvar_a_cada=INTERVALO_SAIDA_S,
    )

    f1 = []
    f2 = []
    f3 = []
    lpi_star = []

    for indice in range(len(historico["t"])):
        campos = {
            nome: np.asarray(historico[nome][indice], dtype=np.float64)
            for nome in ("T", "qc", "qr", "qi", "qs", "qg")
        }
        resultado_mccaul = compute_mccaul(
            z_m=coluna.z,
            temperature_k=campos["T"],
            rho_kg_m3=coluna.rho,
            w_m_s=w_prescrito,
            qi_kgkg=campos["qi"],
            qs_kgkg=campos["qs"],
            qg_kgkg=campos["qg"],
        )
        resultado_lpi = compute_lpi_star(
            z_m=coluna.z,
            temperature_k=campos["T"],
            w_m_s=w_prescrito,
            qc_kgkg=campos["qc"],
            qr_kgkg=campos["qr"],
            qi_kgkg=campos["qi"],
            qs_kgkg=campos["qs"],
            qg_kgkg=campos["qg"],
        )
        f1.append(resultado_mccaul.f1)
        f2.append(resultado_mccaul.f2)
        f3.append(resultado_mccaul.f3)
        lpi_star.append(resultado_lpi.lpi_star)

    return {
        "tempo_min": np.asarray(historico["t"], dtype=np.float64) / 60.0,
        "f1": np.asarray(f1),
        "f2": np.asarray(f2),
        "f3": np.asarray(f3),
        "lpi_star": np.asarray(lpi_star),
    }


def plotar_series(series):
    """Gere uma figura de McCaul e outra de LPI* em outputs/group2/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tempo_min = series["tempo_min"]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True)
    axes[0].plot(tempo_min, series["f1"], color="tab:blue", lw=2.0)
    axes[0].set_ylabel("F1 (diagnóstico relativo)")
    axes[0].set_title("McCaul: fluxo ascendente de graupel em -15 °C")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        tempo_min,
        series["f2"],
        color="tab:orange",
        lw=1.8,
        label="F2: gelo integrado",
    )
    axes[1].plot(
        tempo_min,
        series["f3"],
        color="tab:green",
        lw=2.0,
        label="F3: combinação",
    )
    axes[1].set_xlabel("Tempo (min)")
    axes[1].set_ylabel("Diagnóstico relativo")
    axes[1].set_title("McCaul: conteúdo sólido e diagnóstico combinado")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Série temporal dos diagnósticos de McCaul et al. (2009)")
    fig.tight_layout()
    caminho_mccaul = OUTPUT_DIR / "fig_teste_mccaul_serie_temporal.png"
    fig.savefig(caminho_mccaul, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(tempo_min, series["lpi_star"], color="tab:purple", lw=2.2)
    ax.set_xlabel("Tempo (min)")
    ax.set_ylabel("LPI* (m² s⁻²)")
    ax.set_title("Série temporal do Lightning Potential Index da coluna (LPI*)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    caminho_lpi = OUTPUT_DIR / "fig_teste_lpi_star_serie_temporal.png"
    fig.savefig(caminho_lpi, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return caminho_mccaul, caminho_lpi


def main():
    series = executar_teste()
    nomes = ("f1", "f2", "f3", "lpi_star")
    if not all(np.all(np.isfinite(series[nome])) for nome in nomes):
        raise RuntimeError("A série contém diagnóstico inválido ou não finito")

    caminhos = plotar_series(series)
    print("Teste temporal concluído:")
    print(f" - F3 final:   {series['f3'][-1]:.6e}")
    print(f" - LPI* final: {series['lpi_star'][-1]:.6e} m2 s-2")
    for caminho in caminhos:
        print(f" - {caminho.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
