# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""Executa a matriz experimental do Grupo 1 (sensibilidade a Nc).

Exemplos (a partir da raiz do repositorio):

    python experiments/group1_droplets/rodar_experimentos_grupo1.py --modo teste
    python experiments/group1_droplets/rodar_experimentos_grupo1.py --modo final
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d
from lightning import diagnosticar_relampagos_2d, resumir_diagnosticos_2d


CASOS = {
    "N_LOW": 5.0e7,
    "CTRL": 2.0e8,
    "N_HIGH": 5.0e8,
}

CONFIGURACAO_COMUM = {
    "nx": 90,
    "nz": 110,
    "dx": 100.0,
    "dz": 100.0,
    "dt": 1.5,
    "salvar_a_cada_s": 300.0,
    "bolha_k": 10.0,
    "delta_t_ambiente_k": 0.0,
    "preservar_rh": True,
    "microfisica": "thompson",
    "evap_chuva": True,
    "radiacao": False,
    "ciclo_diurno": False,
    "cenario": "grupo1_sensibilidade_nc",
    "cfl_aviso": 0.80,
    "cfl_limite": 1.00,
    "abortar_se_cfl_violar": True,
}


def obter_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "git_indisponivel"


def salvar_resultado_npz(resultado, diagnosticos, caminho: Path) -> None:
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

    # Os nomes pedidos no README do grupo sao preservados diretamente.
    mapa_lightning = {
        "f1": "F1",
        "f2": "F2",
        "f3": "F3",
        "lpi_star": "LPI_star",
    }
    for nome, valores in diagnosticos.items():
        if nome not in {"t_s", "x_m"}:
            dados[mapa_lightning.get(nome, f"lightning_{nome}")] = valores

    np.savez_compressed(caminho, **dados)


def escrever_json(caminho: Path, conteudo) -> None:
    caminho.write_text(
        json.dumps(conteudo, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def executar_caso(caso: str, nc: float, modo: str, sobrescrever: bool) -> None:
    if modo == "final":
        tempo_total_s = 40.0 * 60.0
        salvar_a_cada_s = CONFIGURACAO_COMUM["salvar_a_cada_s"]
        diretorio = ROOT / "outputs" / "group1" / "experimentos_B10" / caso
    else:
        tempo_total_s = 2.0 * 60.0
        salvar_a_cada_s = 60.0
        diretorio = ROOT / "outputs" / "group1" / "experimentos_B10" / "testes" / caso

    arquivo_npz = diretorio / f"resultados_{caso}.npz"
    if arquivo_npz.exists() and not sobrescrever:
        raise FileExistsError(
            f"A saida ja existe: {arquivo_npz}\n"
            "Use --sobrescrever somente se tiver certeza de que deseja substitui-la."
        )
    diretorio.mkdir(parents=True, exist_ok=True)

    parametros = dict(CONFIGURACAO_COMUM)
    parametros["tempo_total_s"] = tempo_total_s
    parametros["salvar_a_cada_s"] = salvar_a_cada_s
    parametros["nc_ativacao_kg1"] = nc
    parametros["cenario"] = caso
    config = ConfiguracaoDinamica2D(**parametros)

    commit = obter_commit()
    comando = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    inicio_utc = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 72)
    print(f"CASO {caso} | Nc={nc:.3e} kg-1 | modo={modo}")
    print("=" * 72)
    resultado = rodar_thompson_2d(config, verbose=True)
    diagnosticos = diagnosticar_relampagos_2d(resultado)
    resumo_eletrico = resumir_diagnosticos_2d(diagnosticos)

    salvar_resultado_npz(resultado, diagnosticos, arquivo_npz)
    (diretorio / "comando.txt").write_text(comando + "\n", encoding="utf-8")
    (diretorio / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    escrever_json(diretorio / "configuracao.json", asdict(config))

    resumo = {
        "caso": caso,
        "modo": modo,
        "nc_ativacao_kg1": nc,
        "inicio_utc": inicio_utc,
        "fim_utc": datetime.now(timezone.utc).isoformat(),
        "cfl_max_adv": float(resultado["cfl_max_adv"]),
        "cfl_max_diff": float(resultado["cfl_max_diff"]),
        "commit": commit,
        "arquivo_resultado": str(arquivo_npz.relative_to(ROOT)),
        **resumo_eletrico,
    }
    escrever_json(diretorio / "resumo_execucao.json", resumo)

    print(f"Concluido: {caso}")
    print(f"Resultado: {arquivo_npz.relative_to(ROOT)}")
    print(f"CFL maximo advectivo/sedimentacao: {resultado['cfl_max_adv']:.3f}")
    print(f"CFL maximo difusivo: {resultado['cfl_max_diff']:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimentos do Grupo 1: sensibilidade a concentracao de goticulas"
    )
    parser.add_argument(
        "--modo",
        choices=("teste", "final"),
        required=True,
        help="teste=2 min e saida a cada 60 s; final=40 min e saida a cada 300 s",
    )
    parser.add_argument(
        "--caso",
        choices=("TODOS", *CASOS.keys()),
        default="TODOS",
        help="executa todos os casos ou apenas um caso selecionado",
    )
    parser.add_argument(
        "--sobrescrever",
        action="store_true",
        help="permite substituir uma saida existente",
    )
    args = parser.parse_args()

    selecionados = CASOS if args.caso == "TODOS" else {args.caso: CASOS[args.caso]}
    for caso, nc in selecionados.items():
        executar_caso(caso, nc, args.modo, args.sobrescrever)

    print("\nTodos os casos selecionados terminaram sem erro.")


if __name__ == "__main__":
    main()
