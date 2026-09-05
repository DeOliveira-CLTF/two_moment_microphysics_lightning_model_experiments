# -*- coding: utf-8 -*-
"""
gerar_figuras_grupo3.py
========================

Gera as figuras de evolução espacial (qc+qi, w, graupel) para todos os
casos do Grupo 3 a partir dos arquivos .npz já salvos por executar_grupo3.py.

Uso:
    python experiments/group3_process_ablation/gerar_figuras_grupo3.py
"""

import sys
from pathlib import Path

# Ajusta PYTHONPATH para encontrar módulos da raiz
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from configuracao_casos_grupo3 import ORDEM_CASOS
# Importa diretamente do exemplo (ou você pode copiar a função, mas assim é mais limpo)
from examples.nuvem_2d_thompson import salvar_figuras

OUTPUT_ROOT = REPO_ROOT / "outputs" / "group3"


def carregar_resultado_do_npz(caminho_npz):
    """
    Converte o arquivo .npz salvo por executar_grupo3.py em um dicionário
    com a mesma estrutura esperada por salvar_figuras().

    A principal adaptação: renomeia 't_s' → 't' e garante que frames
    contenha apenas arrays 2D (tempo, x, z).
    """
    dados = np.load(caminho_npz, allow_pickle=True)

    # Monta o dicionário 'frames' com todos os campos 2D
    frames = {}
    for chave in dados.files:
        # Ignora campos 1D e metadados
        if chave not in ["x_m", "z_m", "p_pa_1d", "rho0_1d", "theta_env_1d",
                         "T_env_1d", "qv_env_1d", "rh_env_1d", "cfl_max_adv",
                         "cfl_max_diff"]:
            frames[chave] = dados[chave]

    # Renomeia a chave de tempo (se existir) para 't'
    if 't_s' in frames:
        frames['t'] = frames.pop('t_s')
    elif 't' not in frames:
        # Caso raro: tenta usar qualquer chave que pareça tempo
        for k in frames:
            if k.startswith('t'):
                frames['t'] = frames.pop(k)
                break

    # Verifica se 't' está presente
    if 't' not in frames:
        raise KeyError("Nenhum campo de tempo ('t' ou 't_s') encontrado no .npz")

    resultado = {
        "x_m": dados["x_m"],
        "z_m": dados["z_m"],
        "frames": frames,
        "config": None,          # não usado por salvar_figuras
        "estado_final": None,    # não usado
        "cfl_max_adv": float(dados["cfl_max_adv"]),
        "cfl_max_diff": float(dados["cfl_max_diff"]),
    }
    return resultado


def main():
    for caso in ORDEM_CASOS:
        caminho_npz = OUTPUT_ROOT / caso / f"resultados_{caso}.npz"
        if not caminho_npz.exists():
            print(f"[aviso] {caminho_npz} não encontrado. Pule.")
            continue

        print(f"Gerando figuras para {caso}...")
        try:
            resultado = carregar_resultado_do_npz(caminho_npz)
            # salvar_figuras espera: resultado, path_saida, cenario
            salvar_figuras(resultado, OUTPUT_ROOT / caso, f"group3_{caso}")
            print(f"  -> Figuras salvas em {OUTPUT_ROOT / caso}")
        except Exception as e:
            print(f"  -> ERRO ao gerar figuras para {caso}: {e}")

    print("\nTodas as figuras geradas.")


if __name__ == "__main__":
    main()