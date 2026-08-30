# -*- coding: utf-8 -*-
"""Driver geral do modelo 2D + microfisica de dois momentos.

Use este script para testes manuais. Os diretorios experiments/group*/ estao preparados para que
cada grupo implemente seu proprio driver usando a API comum.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d
from lightning import diagnosticar_relampagos_2d


def salvar_npz(resultado, caminho: Path, diagnosticos=None):
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
        "cfl_max_adv": np.array(resultado["cfl_max_adv"]),
        "cfl_max_diff": np.array(resultado["cfl_max_diff"]),
    }
    for nome, valores in frames.items():
        if nome == "t":
            continue
        dados[nome] = valores
    if diagnosticos is not None:
        for nome, valores in diagnosticos.items():
            if nome in {"t_s", "x_m"}:
                continue
            dados[f"lightning_{nome}"] = valores
    np.savez_compressed(caminho, **dados)


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

    fig, axes = plt.subplots(
        3,
        n_frames,
        figsize=(3.0 * n_frames, 8.0),
        sharex=True,
        sharey=True,
    )
    if n_frames == 1:
        axes = axes.reshape(3, 1)

    for j, idx in enumerate(idxs):
        t_min = tempos[idx] / 60.0
        axes[0, j].contourf(
            x / 1000.0,
            z / 1000.0,
            (frames["qc_qi"][idx] * 1000.0).T,
            levels=20,
        )
        axes[0, j].set_title(f"t={t_min:.0f} min\nqc+qi [g/kg]", fontsize=8)

        axes[1, j].contourf(
            x / 1000.0,
            z / 1000.0,
            frames["w"][idx].T,
            levels=20,
        )
        axes[1, j].set_title("w [m/s]", fontsize=8)

        axes[2, j].contourf(
            x / 1000.0,
            z / 1000.0,
            (frames["qg"][idx] * 1000.0).T,
            levels=20,
        )
        axes[2, j].set_title("graupel [g/kg]", fontsize=8)
        axes[2, j].set_xlabel("x [km]", fontsize=8)
        if j == 0:
            for row in range(3):
                axes[row, j].set_ylabel("altura [km]", fontsize=8)

    fig.suptitle(f"Modelo 2D + microfisica de dois momentos - {cenario}")
    plt.tight_layout()
    fig.savefig(saida / f"evolucao_{cenario}.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Modelo 2D + microfisica de dois momentos")
    parser.add_argument("--bolha", type=float, default=8.0, help="amplitude inicial [K]")
    parser.add_argument("--delta-t", type=float, default=0.0, help="aquecimento do ambiente [K]")
    parser.add_argument(
        "--warm-umidade",
        choices=["rh", "qv"],
        default="rh",
        help="rh=preserva RH; qv=preserva qv do ambiente CTRL",
    )
    parser.add_argument("--nc", type=float, default=2.0e8, help="Nc de ativacao [kg-1]")
    parser.add_argument("--cenario", type=str, default="teste")
    parser.add_argument("--tempo", type=float, default=40.0, help="tempo [min]")
    parser.add_argument("--salvar-cada", type=float, default=300.0, help="intervalo de saida [s]")
    parser.add_argument("--nx", type=int, default=90)
    parser.add_argument("--nz", type=int, default=110)
    parser.add_argument("--dx", type=float, default=100.0)
    parser.add_argument("--dz", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1.5)
    parser.add_argument("--evap-chuva", choices=["on", "off"], default="on")
    parser.add_argument("--radiacao", choices=["on", "off"], default="off")
    parser.add_argument("--ciclo-diurno", choices=["on", "off"], default="off")
    parser.add_argument("--raios", action="store_true", help="calcula McCaul e LPI* 2D")
    parser.add_argument("--sem-figuras", action="store_true")
    parser.add_argument("--saida", type=Path, default=Path("outputs") / "dynamic_2d")
    args = parser.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)

    config = ConfiguracaoDinamica2D(
        nx=args.nx,
        nz=args.nz,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
        tempo_total_s=args.tempo * 60.0,
        salvar_a_cada_s=args.salvar_cada,
        bolha_k=args.bolha,
        delta_t_ambiente_k=args.delta_t,
        preservar_rh=args.warm_umidade == "rh",
        nc_ativacao_kg1=args.nc,
        evap_chuva=args.evap_chuva == "on",
        radiacao=args.radiacao == "on",
        ciclo_diurno=args.ciclo_diurno == "on",
        cenario=args.cenario,
    )

    resultado = rodar_thompson_2d(config, verbose=True)
    diagnosticos = diagnosticar_relampagos_2d(resultado) if args.raios else None

    caminho = args.saida / f"resultados_{args.cenario}.npz"
    salvar_npz(resultado, caminho, diagnosticos)

    if not args.sem_figuras:
        salvar_figuras(resultado, args.saida, args.cenario)

    print(f"Saidas salvas em: {args.saida}")
    print(f"CFL maximo advectivo/sedimentacao: {resultado['cfl_max_adv']:.3f}")
    print(f"CFL maximo difusivo: {resultado['cfl_max_diff']:.5f}")


if __name__ == "__main__":
    main()
