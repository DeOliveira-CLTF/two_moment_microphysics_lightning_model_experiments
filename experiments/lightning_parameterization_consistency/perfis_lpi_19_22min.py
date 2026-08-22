# -*- coding: utf-8 -*-
"""Perfis que explicam a evolução do LPI* entre 19 e 22 min.

Executar da raiz:
    python experiments/lightning_parameterization_consistency/perfis_lpi_19_22min.py
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lightning import compute_epsilon, compute_lpi_star
from experiments.lightning_parameterization_consistency.teste_series_temporais_mccaul_lpi import (
    DT_S,
    INTERVALO_SAIDA_S,
    TEMPO_TOTAL_S,
    W_PICO_M_S,
    _configurar_caso,
    _extrair_camada,
    _qf_microfisico,
)

TEMPOS_MIN = (19.0, 20.0, 21.0, 22.0)
OUTPUT_PATH = REPO_ROOT / "outputs" / "lightning_parameterization_consistency" / "fig_perfis_lpi_19_22min.png"


def extrair_perfil(z_m, campos, w_m_s, lpi):
    """Perfis dentro de H0--H-20, com fronteiras interpoladas da integração."""
    z, camada = _extrair_camada(
        z_m, lpi.h_0c_m, lpi.h_minus20c_m,
        {"w": w_m_s, **{nome: campos[nome] for nome in ("qc", "qr", "qi", "qs", "qg")}},
    )
    ql = camada["qc"] + camada["qr"]
    qf = _qf_microfisico(camada["qi"], camada["qs"], camada["qg"])
    epsilon = compute_epsilon(camada["qc"], camada["qr"], camada["qi"], camada["qs"], camada["qg"])
    integrando = camada["w"] ** 2 * (camada["w"] > 0.5) * epsilon

    # Apenas níveis nativos, para não contar as fronteiras interpoladas duas vezes.
    internos = (z_m > lpi.h_0c_m) & (z_m < lpi.h_minus20c_m)
    epsilon_nativo = compute_epsilon(
        campos["qc"][internos], campos["qr"][internos], campos["qi"][internos],
        campos["qs"][internos], campos["qg"][internos],
    )
    return {
        "z_m": z,
        "qL": ql,
        "qF": qf,
        "epsilon": epsilon,
        "ilpi": integrando,
        "h0_m": lpi.h_0c_m,
        "h20_m": lpi.h_minus20c_m,
        "lpi_star": lpi.lpi_star,
        "n_epsilon_positivo": int(np.count_nonzero(epsilon_nativo > 0.0)),
    }


def executar():
    coluna, w, _ = _configurar_caso()
    historico = coluna.integrar(TEMPO_TOTAL_S, dt=DT_S, salvar_a_cada=INTERVALO_SAIDA_S)
    perfis = {}
    for indice, tempo_s in enumerate(historico["t"]):
        tempo_min = float(tempo_s / 60.0)
        if not any(np.isclose(tempo_min, alvo) for alvo in TEMPOS_MIN):
            continue
        campos = {nome: np.asarray(historico[nome][indice], dtype=np.float64)
                  for nome in ("T", "qc", "qr", "qi", "qs", "qg")}
        lpi = compute_lpi_star(coluna.z, campos["T"], w, campos["qc"], campos["qr"],
                               campos["qi"], campos["qs"], campos["qg"])
        if not lpi.valid:
            raise RuntimeError(lpi.status)
        perfis[tempo_min] = extrair_perfil(coluna.z, campos, w, lpi)
    if tuple(perfis) != TEMPOS_MIN:
        raise RuntimeError(f"Instantes ausentes: esperado {TEMPOS_MIN}, obtido {tuple(perfis)}")
    return perfis


def plotar(perfis):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 7), sharey=True)
    paineis = (
        ("qL", "qL (g kg⁻¹)", 1000.0, "Água líquida"),
        ("qF", "qF (g kg⁻¹)", 1000.0, "Fase congelada efetiva"),
        ("epsilon", "epsilon (-)", 1.0, "Coexistência"),
        ("ilpi", "w² g(w) epsilon (m² s⁻²)", 1.0, "Integrando real do LPI*"),
    )
    for ax, (campo, xlabel, escala, titulo) in zip(axes, paineis):
        for tempo in TEMPOS_MIN:
            perfil = perfis[tempo]
            ax.plot(perfil[campo] * escala, perfil["z_m"] / 1000.0, lw=2, label=f"{tempo:.0f} min")
        ax.set(xlabel=xlabel, title=titulo)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Altura (km)")
    axes[-1].legend(title="Instante")
    fig.suptitle("Perfis na camada de carregamento (0 a −20 °C)")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    perfis = executar()
    print("Perfis do LPI* (níveis nativos com epsilon > 0):")
    for tempo in TEMPOS_MIN:
        p = perfis[tempo]
        print(
            f"t={tempo:.0f} min | H0={p['h0_m']/1000:.3f} km | "
            f"H-20={p['h20_m']/1000:.3f} km | "
            f"H-20-H0={(p['h20_m']-p['h0_m'])/1000:.3f} km | "
            f"n(epsilon>0)={p['n_epsilon_positivo']} | LPI*={p['lpi_star']:.6e} m2 s-2"
        )
    plotar(perfis)
    print(f"Figura: {OUTPUT_PATH.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
