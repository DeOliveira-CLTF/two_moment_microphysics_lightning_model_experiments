# -*- coding: utf-8 -*-
"""Driver 2D acoplado ao esquema Thompson de dois momentos.

Este exemplo substitui o script monolitico `nuvem_2d_thompson.py` do professor
por uma chamada ao pacote `dinamica_2d`. Os grupos podem copiar apenas a logica
de configuracao/saida para seus experimentos, mantendo o nucleo dinamico fora
da pasta `experiments/`.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d


def salvar_figuras(resultado, saida: Path, cenario: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = resultado["frames"]
    x = resultado["x_m"]
    z = resultado["z_m"]
    tempos = frames["t"]
    n_frames = min(6, len(tempos))
    idxs = np.linspace(0, len(tempos) - 1, n_frames).astype(int)

    fig, axes = plt.subplots(3, n_frames, figsize=(3.0 * n_frames, 8.0), sharex=True, sharey=True)
    if n_frames == 1:
        axes = axes.reshape(3, 1)

    for j, idx in enumerate(idxs):
        t_min = tempos[idx] / 60.0
        axes[0, j].contourf(x / 1000.0, z / 1000.0, (frames["qc_qi"][idx] * 1000.0).T, levels=20, cmap="Blues")
        axes[0, j].set_title(f"t={t_min:.0f} min\nqc+qi [g/kg]", fontsize=8)

        axes[1, j].contourf(x / 1000.0, z / 1000.0, frames["w"][idx].T, levels=20, cmap="RdBu_r")
        axes[1, j].set_title("w [m/s]", fontsize=8)

        axes[2, j].contourf(x / 1000.0, z / 1000.0, (frames["qr_qs_qg"][idx] * 1000.0).T, levels=20, cmap="Greens")
        axes[2, j].set_title("chuva+neve+graupel [g/kg]", fontsize=8)
        axes[2, j].set_xlabel("x [km]", fontsize=8)
        if j == 0:
            for row in range(3):
                axes[row, j].set_ylabel("altura [km]", fontsize=8)

    fig.suptitle(f"Modelo 2D + Thompson completo - {cenario}", fontsize=10)
    plt.tight_layout()
    fig.savefig(saida / f"nuvem_2d_thompson_evolucao_{cenario}.png", dpi=120)
    plt.close(fig)

    thp_min_sfc = [campo[:, 0:3].min() for campo in frames["thp"]]
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tempos / 60.0, thp_min_sfc, marker="o")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("tempo [min]")
    ax.set_ylabel("theta' minimo perto do chao [K]")
    ax.set_title(f"Cold pool - {cenario}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig(saida / f"nuvem_2d_thompson_coldpool_{cenario}.png", dpi=120)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Modelo 2D acoplado ao esquema Thompson completo")
    parser.add_argument("--bolha", type=float, default=3.0, help="amplitude da bolha termica inicial [K]")
    parser.add_argument("--cenario", type=str, default="bolha", help="rotulo para os arquivos de saida")
    parser.add_argument("--microfisica", choices=["nenhuma", "thompson"], default="thompson")
    parser.add_argument("--evap-chuva", choices=["on", "off"], default="on")
    parser.add_argument("--radiacao", choices=["on", "off"], default="off")
    parser.add_argument("--ciclo-diurno", choices=["on", "off"], default="off")
    parser.add_argument("--tempo", type=float, default=10.0, help="tempo total de simulacao [min]")
    parser.add_argument("--nx", type=int, default=40, help="pontos de grade na horizontal")
    parser.add_argument("--nz", type=int, default=60, help="pontos de grade na vertical")
    parser.add_argument("--saida", type=Path, default=Path("outputs") / "dynamic_2d")
    parser.add_argument("--sem-figuras", action="store_true", help="salva apenas o arquivo .npz")
    args = parser.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)
    config = ConfiguracaoDinamica2D(
        nx=args.nx,
        nz=args.nz,
        tempo_total_s=args.tempo * 60.0,
        bolha_k=args.bolha,
        microfisica=args.microfisica,
        evap_chuva=args.evap_chuva == "on",
        radiacao=args.radiacao == "on",
        ciclo_diurno=args.ciclo_diurno == "on",
        cenario=args.cenario,
    )

    resultado = rodar_thompson_2d(config, verbose=True)
    frames = resultado["frames"]
    np.savez_compressed(
        args.saida / f"resultados_nuvem_2d_thompson_{args.cenario}.npz",
        t_s=frames["t"],
        x_m=resultado["x_m"],
        z_m=resultado["z_m"],
        qc_qi=frames["qc_qi"],
        qr_qs_qg=frames["qr_qs_qg"],
        qg=frames["qg"],
        w=frames["w"],
        u=frames["u"],
        thp=frames["thp"],
        qvp=frames["qvp"],
    )

    if not args.sem_figuras:
        salvar_figuras(resultado, args.saida, args.cenario)

    print(f"Saidas salvas em: {args.saida}")


if __name__ == "__main__":
    main()
