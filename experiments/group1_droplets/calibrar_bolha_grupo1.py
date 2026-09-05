# -*- coding: utf-8 -*-
"""Calibra a amplitude da bolha termica antes dos experimentos do Grupo 1.

Cada bolha e executada somente com o Nc do CTRL (2.0e8 kg-1). As saidas sao
independentes dos experimentos finais de Nc e ficam em:

    outputs/group1/calibracao/B08, B10, B12 e B14

Uso, a partir da raiz do repositorio:

    python experiments/group1_droplets/calibrar_bolha_grupo1.py --modo teste
    python experiments/group1_droplets/calibrar_bolha_grupo1.py --modo calibracao
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d
from lightning import diagnosticar_relampagos_2d, resumir_diagnosticos_2d


BOLHAS_PADRAO = (8.0, 10.0, 12.0, 14.0)
NC_CTRL = 2.0e8
LIMIAR_HIDRO = 1.0e-6  # kg kg-1
T0 = 273.15
T_MENOS20 = 253.15
CORES = ("#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7")


def rotulo_bolha(valor: float) -> str:
    if float(valor).is_integer():
        return f"B{int(valor):02d}"
    return "B" + str(valor).replace(".", "p")


def obter_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "git_indisponivel"


def escrever_json(caminho: Path, conteudo) -> None:
    caminho.write_text(
        json.dumps(conteudo, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def salvar_npz(resultado, diagnosticos, caminho: Path) -> None:
    frames = resultado["frames"]
    dados = {
        "t_s": frames["t"],
        "x_m": resultado["x_m"],
        "z_m": resultado["z_m"],
        "p_pa_1d": resultado["p_pa_1d"],
        "rho0_1d": resultado["rho0_1d"],
        "theta_env_1d": resultado["theta_env_1d"],
        "T_env_1d": resultado["T_env_1d"],
        "qv_env_1d": resultado["qv_env_1d"],
        "rh_env_1d": resultado["rh_env_1d"],
        "cfl_max_adv": np.asarray(resultado["cfl_max_adv"]),
        "cfl_max_diff": np.asarray(resultado["cfl_max_diff"]),
    }
    for nome, valores in frames.items():
        if nome != "t":
            dados[nome] = valores
    nomes_eletricos = {
        "f1": "F1", "f2": "F2", "f3": "F3", "lpi_star": "LPI_star"
    }
    for nome, valores in diagnosticos.items():
        if nome not in {"t_s", "x_m"}:
            dados[nomes_eletricos.get(nome, f"lightning_{nome}")] = valores
    np.savez_compressed(caminho, **dados)


def water_path(campo, rho, dz):
    return np.sum(campo * rho[None, None, :], axis=(1, 2)) * dz / campo.shape[1]


def diagnosticar_fase_mista(resultado):
    f = resultado["frames"]
    rho = resultado["rho0_1d"]
    z = resultado["z_m"]
    dz = float(np.mean(np.diff(z)))
    liquido = f["qc"] + f["qr"]
    gelo = f["qi"] + f["qs"] + f["qg"]
    faixa_termica = (f["T"] <= T0) & (f["T"] >= T_MENOS20)
    mascara = faixa_termica & (liquido > LIMIAR_HIDRO) & (gelo > LIMIAR_HIDRO)
    conteudo = np.where(mascara, liquido + gelo, 0.0)
    wp_mista = water_path(conteudo, rho, dz)
    fracao_pontos = np.mean(mascara, axis=(1, 2))
    return wp_mista, fracao_pontos


def diagnosticar_topo(resultado):
    f = resultado["frames"]
    z = resultado["z_m"]
    condensado = f["qc"] + f["qr"] + f["qi"] + f["qs"] + f["qg"]
    topos = []
    for frame in condensado:
        niveis = np.flatnonzero(np.max(frame, axis=0) > 1.0e-5)
        topos.append(float(z[niveis[-1]]) if niveis.size else 0.0)
    return np.asarray(topos)


def resumir_caso(resultado, diagnosticos, bolha_k, nome):
    f = resultado["frames"]
    rho = resultado["rho0_1d"]
    dz = float(np.mean(np.diff(resultado["z_m"])))
    wp_mista, fracao_mista = diagnosticar_fase_mista(resultado)
    topo = diagnosticar_topo(resultado)
    eletrico = resumir_diagnosticos_2d(diagnosticos)

    resumo = {
        "caso": nome,
        "bolha_k": bolha_k,
        "Nc_ctrl_kg-1": NC_CTRL,
        "tempo_total_min": float(f["t"][-1] / 60.0),
        "w_max_ms-1": float(np.max(f["w"])),
        "topo_max_km": float(np.max(topo) / 1000.0),
        "qc_max_gkg": float(np.max(f["qc"]) * 1000.0),
        "qr_max_gkg": float(np.max(f["qr"]) * 1000.0),
        "qi_max_gkg": float(np.max(f["qi"]) * 1000.0),
        "qs_max_gkg": float(np.max(f["qs"]) * 1000.0),
        "qg_max_gkg": float(np.max(f["qg"]) * 1000.0),
        "wp_qc_max_kgm-2": float(np.max(water_path(f["qc"], rho, dz))),
        "wp_qr_max_kgm-2": float(np.max(water_path(f["qr"], rho, dz))),
        "wp_gelo_max_kgm-2": float(
            np.max(water_path(f["qi"] + f["qs"] + f["qg"], rho, dz))
        ),
        "wp_fase_mista_max_kgm-2": float(np.max(wp_mista)),
        "fracao_fase_mista_max": float(np.max(fracao_mista)),
        "tempo_max_fase_mista_min": float(f["t"][int(np.argmax(wp_mista))] / 60.0),
        "F1_max": eletrico["f1_max"],
        "F2_max": eletrico["f2_max"],
        "F3_max": eletrico["f3_max"],
        "LPI_star_max": eletrico["lpi_star_max"],
        "CFL_adv_max": float(resultado["cfl_max_adv"]),
        "CFL_diff_max": float(resultado["cfl_max_diff"]),
    }
    return resumo, wp_mista, topo


def plotar_caso(resultado, resumo, wp_mista, topo, saida: Path, nome: str):
    f = resultado["frames"]
    x = resultado["x_m"] / 1000.0
    z = resultado["z_m"] / 1000.0
    tempos = f["t"] / 60.0
    indice = int(np.argmax(wp_mista)) if np.max(wp_mista) > 0 else int(
        np.argmax(np.max(f["qc"] + f["qi"], axis=(1, 2)))
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    qc = axes[0, 0].contourf(x, z, (f["qc"][indice] * 1000.0).T, 20, cmap="Blues")
    axes[0, 0].contour(x, z, f["T"][indice].T - 273.15,
                       levels=[-20, 0], colors=["purple", "black"], linewidths=1.1)
    fig.colorbar(qc, ax=axes[0, 0], label="qc (g kg⁻¹)")
    axes[0, 0].set_title("Água de nuvem; contornos -20 e 0 °C")

    gelo_total = f["qi"][indice] + f["qs"][indice] + f["qg"][indice]
    gelo = axes[0, 1].contourf(x, z, (gelo_total * 1000.0).T, 20, cmap="Purples")
    fig.colorbar(gelo, ax=axes[0, 1], label="qi+qs+qg (g kg⁻¹)")
    axes[0, 1].set_title("Hidrometeoros de gelo")

    w = axes[1, 0].contourf(x, z, f["w"][indice].T, 20, cmap="RdBu_r")
    fig.colorbar(w, ax=axes[1, 0], label="w (m s⁻¹)")
    axes[1, 0].set_title("Velocidade vertical")

    eixo = axes[1, 1]
    eixo.plot(tempos, topo, color="#333333", lw=2.2, label="Topo da nuvem")
    eixo.set_ylabel("Topo (m)")
    eixo2 = eixo.twinx()
    eixo2.plot(tempos, wp_mista, color="#D55E00", lw=2.2,
               label="Conteúdo de fase mista")
    eixo2.set_ylabel("Fase mista (kg m⁻²)", color="#D55E00")
    eixo.set_title("Evolução do topo e da fase mista")
    eixo.grid(True, alpha=0.2)

    axes[0, 0].set_ylabel("Altura (km)")
    axes[1, 0].set_ylabel("Altura (km)")
    axes[1, 0].set_xlabel("x (km)")
    axes[1, 1].set_xlabel("Tempo (min)")
    fig.suptitle(
        f"Calibração {nome}: bolha={resumo['bolha_k']:g} K | "
        f"quadro={tempos[indice]:.0f} min | CFL={resumo['CFL_adv_max']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(saida / f"figura_calibracao_{nome}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plotar_comparacao(resumos, pasta: Path):
    nomes = [r["caso"] for r in resumos]
    bolhas = [r["bolha_k"] for r in resumos]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    itens = (
        ("topo_max_km", "Topo máximo (km)"),
        ("w_max_ms-1", "w máximo (m s⁻¹)"),
        ("wp_fase_mista_max_kgm-2", "Fase mista máxima (kg m⁻²)"),
        ("LPI_star_max", "LPI* máximo"),
    )
    for ax, (chave, titulo) in zip(axes.flat, itens):
        valores = [r[chave] for r in resumos]
        ax.plot(bolhas, valores, marker="o", lw=2.2, color="#0072B2")
        ax.set_xlabel("Amplitude da bolha (K)")
        ax.set_ylabel(titulo)
        ax.set_xticks(bolhas, nomes)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Comparação das calibrações da bolha térmica", fontsize=14)
    fig.tight_layout()
    fig.savefig(pasta / "figura_comparacao_calibracao_bolha.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def salvar_tabela(resumos, pasta: Path):
    campos = list(resumos[0].keys())
    with (pasta / "tabela_comparacao_calibracao_bolha.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resumos)


def executar_bolha(bolha_k, modo, sobrescrever):
    nome = rotulo_bolha(bolha_k)
    if modo == "teste":
        pasta = ROOT / "outputs" / "group1" / "calibracao" / "testes" / nome
        tempo_total_s = 120.0
        salvar_a_cada_s = 60.0
    else:
        pasta = ROOT / "outputs" / "group1" / "calibracao" / nome
        tempo_total_s = 40.0 * 60.0
        salvar_a_cada_s = 60.0
    arquivo_npz = pasta / f"resultados_calibracao_{nome}.npz"
    if arquivo_npz.exists() and not sobrescrever:
        raise FileExistsError(
            f"A saida {arquivo_npz} ja existe. Use --sobrescrever apenas se necessario."
        )
    pasta.mkdir(parents=True, exist_ok=True)

    config = ConfiguracaoDinamica2D(
        nx=90, nz=110, dx=100.0, dz=100.0, dt=1.5,
        tempo_total_s=tempo_total_s, salvar_a_cada_s=salvar_a_cada_s,
        bolha_k=bolha_k, delta_t_ambiente_k=0.0, preservar_rh=True,
        microfisica="thompson", evap_chuva=True, nc_ativacao_kg1=NC_CTRL,
        radiacao=False, ciclo_diurno=False, cenario=f"calibracao_{nome}",
        cfl_aviso=0.80, cfl_limite=1.00, abortar_se_cfl_violar=True,
    )
    print("\n" + "=" * 72)
    print(f"CALIBRACAO {nome} | bolha={bolha_k:g} K | modo={modo}")
    print("=" * 72)
    resultado = rodar_thompson_2d(config, verbose=True)
    diagnosticos = diagnosticar_relampagos_2d(resultado)
    resumo, wp_mista, topo = resumir_caso(resultado, diagnosticos, bolha_k, nome)
    resumo["inicio_registro_utc"] = datetime.now(timezone.utc).isoformat()
    resumo["commit"] = obter_commit()

    salvar_npz(resultado, diagnosticos, arquivo_npz)
    escrever_json(pasta / f"configuracao_calibracao_{nome}.json", asdict(config))
    escrever_json(pasta / f"resumo_calibracao_{nome}.json", resumo)
    (pasta / f"comando_calibracao_{nome}.txt").write_text(
        " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (pasta / f"commit_calibracao_{nome}.txt").write_text(
        resumo["commit"] + "\n", encoding="utf-8"
    )
    plotar_caso(resultado, resumo, wp_mista, topo, pasta, nome)
    print(
        f"{nome} concluido | topo={resumo['topo_max_km']:.2f} km | "
        f"fase_mista={resumo['wp_fase_mista_max_kgm-2']:.3e} kg/m2 | "
        f"qg_max={resumo['qg_max_gkg']:.3e} g/kg | "
        f"LPI*={resumo['LPI_star_max']:.3e} | CFL={resumo['CFL_adv_max']:.3f}"
    )
    return resumo


def main():
    parser = argparse.ArgumentParser(description="Calibracao da bolha termica")
    parser.add_argument("--modo", choices=("teste", "calibracao"), required=True)
    parser.add_argument("--bolhas", nargs="+", type=float, default=list(BOLHAS_PADRAO))
    parser.add_argument("--sobrescrever", action="store_true")
    args = parser.parse_args()

    resumos = [
        executar_bolha(valor, args.modo, args.sobrescrever)
        for valor in args.bolhas
    ]
    if args.modo == "teste":
        pasta_comparacao = ROOT / "outputs" / "group1" / "calibracao" / "testes"
    else:
        pasta_comparacao = ROOT / "outputs" / "group1" / "calibracao"
    salvar_tabela(resumos, pasta_comparacao)
    plotar_comparacao(resumos, pasta_comparacao)
    print("\nTodas as calibracoes selecionadas terminaram sem erro.")
    print(f"Comparacao salva em: {pasta_comparacao.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
