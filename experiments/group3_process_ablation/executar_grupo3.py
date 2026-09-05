# -*- coding: utf-8 -*-
"""
executar_grupo3.py
=====================

Driver do GRUPO EXPERIMENTAL 3 (atribuição de processos da fase mista),
seguindo estritamente o plano de experimentos (seção 4 e 5.2):

    - Importa o núcleo dinâmico comum (dinamica_2d) e a microfísica
      completa (microfisica.coluna_generica), SEM copiar ou reescrever
      nenhuma parametrização aqui.
    - A ÚNICA coisa que varia entre os casos é o campo `processos`
      (OpcoesMicrofisica) de `ConfiguracaoDinamica2D` -- ver
      `configuracao_casos_grupo3.py`.
    - Todos os demais parâmetros (grade, Δt, bolha, Nc, evaporação de
      chuva, radiação, ciclo diurno) são mantidos IDÊNTICOS ao CTRL
      comum descrito na seção 1.1 do plano.

Uso (a partir da raiz do repositório):

    python experiments/group3_process_ablation/executar_grupo3.py
    python experiments/group3_process_ablation/executar_grupo3.py --tempo 40 --dt 0.5 --bolha 16

Cada execução salva, em outputs/group3/<CASO>/:
    resultados_<CASO>.npz   -- séries/campos completos + diagnósticos de raios
    configuracao_<CASO>.json -- parâmetros científicos declarados (seção 5.5)
    comando.txt              -- linha de comando exata usada
    commit.txt               -- hash do commit git no momento da execução
    status.txt               -- OK / FALHOU (+ motivo) e CFL máximo atingido
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
import concurrent.futures
import multiprocessing

# Ajusta o PYTHONPATH para importar módulos da raiz
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d
from lightning import diagnosticar_relampagos_2d
from configuracao_casos_grupo3 import CASOS_GRUPO3, ORDEM_CASOS, validar_casos

OUTPUT_ROOT = ROOT / "outputs" / "group3"


# ----------------------------------------------------------------------
# Metadados de reprodutibilidade (seção 5.5 do plano)
# ----------------------------------------------------------------------
def obter_commit_git():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return commit.decode().strip()
    except Exception:
        return "DESCONHECIDO (git indisponível ou fora de um repositório)"


def salvar_npz_caso(resultado, caminho: Path):
    """
    Salva os resultados no formato .npz, incluindo os diagnósticos de
    relâmpagos se presentes no dicionário 'frames'.
    """
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
    np.savez_compressed(caminho, **dados)


def rodar_um_caso(caso, args, comando_completo, commit, python_version):
    """
    Função auxiliar que executa UM caso e retorna o status.
    Esta função é chamada em paralelo por ProcessPoolExecutor.
    """
    spec = CASOS_GRUPO3[caso]
    saida_caso = OUTPUT_ROOT / caso
    saida_caso.mkdir(parents=True, exist_ok=True)

    config = ConfiguracaoDinamica2D(
        nx=args.nx,
        nz=args.nz,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
        tempo_total_s=args.tempo * 60.0,
        salvar_a_cada_s=args.salvar_cada,
        bolha_k=args.bolha,
        delta_t_ambiente_k=0.0,          # Grupo 3: sem aquecimento
        preservar_rh=True,
        nc_ativacao_kg1=args.nc,
        evap_chuva=(args.evap_chuva == "on"),
        radiacao=(args.radiacao == "on"),
        ciclo_diurno=(args.ciclo_diurno == "on"),
        processos=spec["opcoes"],
        cenario=caso,
    )

    # Prepara as linhas de status
    status_linhas = [
        f"caso={caso}",
        f"commit={commit}",
        f"python={python_version}",
        f"comando={comando_completo}"
    ]

    try:
        # 1. Executa a simulação
        resultado = rodar_thompson_2d(config, verbose=args.verbose)

        # 2. Calcula diagnósticos de relâmpagos (McCaul e LPI*)
        diag = diagnosticar_relampagos_2d(resultado)
        if diag is not None:
            frames = resultado["frames"]
            for chave, valor in diag.items():
                if chave not in frames:  # evita sobrescrever campos existentes
                    frames[chave] = valor
            resultado["frames"] = frames

        # 3. Salva o arquivo .npz
        salvar_npz_caso(resultado, saida_caso / f"resultados_{caso}.npz")

        # 4. Atualiza status
        status_linhas.append("status=OK")
        status_linhas.append(f"cfl_max_adv={resultado['cfl_max_adv']:.4f}")
        status_linhas.append(f"cfl_max_diff={resultado['cfl_max_diff']:.6f}")
        print(f"OK | {caso} | CFL_adv_max={resultado['cfl_max_adv']:.3f}")
        sucesso = True
        cfl_adv = resultado['cfl_max_adv']

    except RuntimeError as exc:
        status_linhas.append("status=FALHOU")
        status_linhas.append(f"motivo={exc}")
        print(f"FALHOU | {caso} | {exc}")
        sucesso = False
        cfl_adv = None

    # 5. Salva metadados (sempre, mesmo em caso de falha)
    (saida_caso / "comando.txt").write_text(comando_completo + "\n")
    (saida_caso / "commit.txt").write_text(commit + "\n")
    (saida_caso / "status.txt").write_text("\n".join(status_linhas) + "\n")

    # 6. Salva configuração em JSON (com todas as opções microfísicas)
    config_dict = asdict(config)
    config_dict["processos"] = asdict(config.processos)
    config_dict["processo_removido"] = spec["processo_removido"]
    config_dict["pergunta_fisica"] = spec["pergunta_fisica"]
    with open(saida_caso / f"configuracao_{caso}.json", "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    return caso, sucesso, cfl_adv


def main():
    parser = argparse.ArgumentParser(
        description="Grupo 3 - ablação de processos da fase mista"
    )
    parser.add_argument(
        "--tempo", type=float, default=40.0,
        help=(
            "duração da simulação [min]. ATENÇÃO (seção 1.1 do plano): "
            "este valor deve ser CONGELADO antes da bateria científica "
            "e usado idêntico em todos os casos; ajuste o default "
            "aqui apenas uma vez, na calibração."
        ),
    )
    parser.add_argument("--salvar-cada", type=float, default=300.0,
                        help="intervalo de saída [s]")
    parser.add_argument("--nx", type=int, default=90)
    parser.add_argument("--nz", type=int, default=110)
    parser.add_argument("--dx", type=float, default=100.0)
    parser.add_argument("--dz", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1.5,
                        help="passo de tempo [s] - reduzir se CFL violar")
    parser.add_argument("--bolha", type=float, default=8.0,
                        help="B0 [K], referência operacional (seção 1.1/3.1)")
    parser.add_argument("--nc", type=float, default=2.0e8,
                        help="Nc de ativação [kg-1], CTRL comum")
    parser.add_argument("--evap-chuva", choices=["on", "off"], default="on")
    parser.add_argument("--radiacao", choices=["on", "off"], default="off")
    parser.add_argument("--ciclo-diurno", choices=["on", "off"], default="off")
    parser.add_argument(
        "--casos", nargs="+", default=ORDEM_CASOS, choices=ORDEM_CASOS,
        help="subconjunto de casos a rodar (default: todos, na ordem do plano)",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Valida que cada ablação desliga o(s) processo(s) correto(s)
    validar_casos()

    comando_completo = " ".join(sys.argv)
    commit = obter_commit_git()
    python_version = sys.version.split()[0]

    # Número de workers = número de núcleos disponíveis
    n_workers = min(len(args.casos), multiprocessing.cpu_count())
    print(f"Rodando {len(args.casos)} casos com {n_workers} processadores em paralelo...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submete todos os casos
        futuros = {
            executor.submit(rodar_um_caso, caso, args, comando_completo, commit, python_version): caso
            for caso in args.casos
        }

        # Aguarda todos terminarem e coleta resultados
        for futuro in concurrent.futures.as_completed(futuros):
            caso, sucesso, cfl = futuro.result()
            if sucesso:
                print(f"✅ {caso} concluído (CFL={cfl:.3f})")
            else:
                print(f"❌ {caso} falhou")

    print(f"\nTodas as saídas em: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()