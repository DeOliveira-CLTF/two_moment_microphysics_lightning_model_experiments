# -*- coding: utf-8 -*-
"""
gerar_figuras_grupo3_colorbar_fixa_final.py
============================================

Gera figuras de evolução 2D (qc+qi, w, qg) para o Grupo 3,
com colorbars e mapas de cores via pyart (HomeyerRainbow, balance),
com escala FIXA para todos os casos.

A escala pode ser definida manualmente via argumentos de linha de comando,
ou calculada automaticamente com base no percentil (padrão 90) dos valores.

As figuras são geradas em torno do instante de pico da velocidade vertical
do caso CTRL, com intervalo de 10 minutos entre cada imagem,
no máximo 5 colunas.

Uso:
    # Automático (percentil 90)
    python experiments/group3_process_ablation/gerar_figuras_grupo3_colorbar_fixa_final.py
    
    # Manual
    python ... --w-lim 15 --qcqi-lim 20 --qg-lim 5
    
    # Automático com percentil diferente
    python ... --percentil 95
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pyart  # Para os colormaps

from configuracao_casos_grupo3 import ORDEM_CASOS

OUTPUT_ROOT = REPO_ROOT / "outputs" / "group3"

# ----------------------------------------------------------------------
# Escolha dos colormaps (via pyart)
# ----------------------------------------------------------------------
# Para campos não divergentes (qc+qi, qg): HomeyerRainbow
CMAP_POS = 'HomeyerRainbow'
# Para w (divergente): balance (escala simétrica azul-branco-vermelho)
CMAP_DIV = 'balance'
# Número de níveis para contornos
N_LEVELS = 31


def carregar_dados(caso):
    """Carrega os dados de um caso específico."""
    caminho = OUTPUT_ROOT / caso / f"resultados_{caso}.npz"
    if not caminho.exists():
        return None
    dados = np.load(caminho, allow_pickle=True)
    if 't_s' in dados:
        t = dados['t_s']
    elif 't' in dados:
        t = dados['t']
    else:
        t = None
    return {
        'x': dados['x_m'],
        'z': dados['z_m'],
        't': t,
        'qc': dados.get('qc'),
        'qi': dados.get('qi'),
        'w': dados.get('w'),
        'qg': dados.get('qg'),
    }


def arredondar_para_multiplo(valor, multiplo=5):
    """Arredonda para o próximo múltiplo de 'multiplo' (para cima)."""
    return int(np.ceil(valor / multiplo) * multiplo)


def obter_percentis_globais(casos, percentil=90):
    """
    Calcula o percentil escolhido dos valores de w, qc+qi e qg
    entre todos os casos. Retorna os limites arredondados para múltiplos de 5.
    """
    w_all = []
    qcqi_all = []
    qg_all = []

    for caso in casos:
        dados = carregar_dados(caso)
        if dados is None:
            continue
        if dados['w'] is not None:
            w_all.append(dados['w'].flatten())
        if dados['qc'] is not None and dados['qi'] is not None:
            qcqi_all.append((dados['qc'] + dados['qi']).flatten())
        if dados['qg'] is not None:
            qg_all.append(dados['qg'].flatten())

    w_all = np.concatenate(w_all) if w_all else np.array([0.0])
    qcqi_all = np.concatenate(qcqi_all) if qcqi_all else np.array([0.0])
    qg_all = np.concatenate(qg_all) if qg_all else np.array([0.0])

    w_perc = np.percentile(np.abs(w_all), percentil)
    qcqi_perc = np.percentile(qcqi_all, percentil)
    qg_perc = np.percentile(qg_all, percentil)

    w_lim = arredondar_para_multiplo(w_perc, 5)
    qcqi_lim = arredondar_para_multiplo(qcqi_perc * 1000, 5)  # g/kg
    qg_lim = arredondar_para_multiplo(qg_perc * 1000, 5)

    w_lim = max(w_lim, 5)
    qcqi_lim = max(qcqi_lim, 5)
    qg_lim = max(qg_lim, 5)

    return w_lim, qcqi_lim, qg_lim


def gerar_figura_caso(caso, w_lim, qcqi_lim, qg_lim, idxs):
    """
    Gera a figura para um caso, utilizando os índices de tempo fornecidos.
    """
    dados = carregar_dados(caso)
    if dados is None:
        print(f"[aviso] Caso {caso} não encontrado.")
        return

    nt = dados['t'].shape[0] if dados['t'] is not None else 1
    idxs_validos = [i for i in idxs if 0 <= i < nt]
    if len(idxs_validos) == 0:
        print(f"[aviso] Nenhum índice válido para {caso}.")
        return

    x = dados['x'] / 1000.0
    z = dados['z'] / 1000.0
    times = dados['t'][idxs_validos] / 60.0

    qcqi = (dados['qc'] + dados['qi']) * 1000 if (dados['qc'] is not None and dados['qi'] is not None) else None
    qg = dados['qg'] * 1000 if dados['qg'] is not None else None
    w = dados['w'] if dados['w'] is not None else None

    levels_qcqi = np.linspace(0, qcqi_lim, N_LEVELS)
    levels_w = np.linspace(-w_lim, w_lim, N_LEVELS)
    levels_qg = np.linspace(0, qg_lim, N_LEVELS)

    norm_qcqi = Normalize(vmin=0, vmax=qcqi_lim)
    norm_w = Normalize(vmin=-w_lim, vmax=w_lim)
    norm_qg = Normalize(vmin=0, vmax=qg_lim)

    n_cols = len(idxs_validos)
    fig, axes = plt.subplots(nrows=3, ncols=n_cols,
                             figsize=(3.5 * n_cols, 9),
                             sharex=True, sharey=True)
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    for i, idx in enumerate(idxs_validos):
        # --- qc+qi ---
        ax = axes[0, i]
        if qcqi is not None:
            campo = qcqi[idx].T
            im = ax.contourf(x, z, campo, levels=levels_qcqi,
                             cmap=CMAP_POS, norm=norm_qcqi, extend='max')
            fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02,
                         ticks=np.linspace(0, qcqi_lim, 5))
        ax.set_title(f"{times[i]:.0f} min", fontsize=9)
        if i == 0:
            ax.set_ylabel("altura (km)")

        # --- w (colormap balance) ---
        ax = axes[1, i]
        if w is not None:
            campo = w[idx].T
            im = ax.contourf(x, z, campo, levels=levels_w,
                             cmap=CMAP_DIV, norm=norm_w, extend='both')
            fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02,
                         ticks=np.linspace(-w_lim, w_lim, 5))
        if i == 0:
            ax.set_ylabel("altura (km)")

        # --- qg ---
        ax = axes[2, i]
        if qg is not None:
            campo = qg[idx].T
            im = ax.contourf(x, z, campo, levels=levels_qg,
                             cmap=CMAP_POS, norm=norm_qg, extend='max')
            fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02,
                         ticks=np.linspace(0, qg_lim, 5))
        ax.set_xlabel("x (km)")
        if i == 0:
            ax.set_ylabel("altura (km)")

    axes[0, 0].set_ylabel("qc+qi (g/kg)", fontsize=9)
    axes[1, 0].set_ylabel("w (m/s)", fontsize=9)
    axes[2, 0].set_ylabel("qg (g/kg)", fontsize=9)

    plt.suptitle(f"Grupo 3 - {caso}", fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    saida = OUTPUT_ROOT / caso / f"evolucao_{caso}_colorbar_fixa.png"
    plt.savefig(saida, dpi=150)
    plt.close()
    print(f"  -> {saida}")


def main():
    parser = argparse.ArgumentParser(description='Gera figuras com escala fixa para o Grupo 3')
    parser.add_argument('--percentil', type=int, default=90,
                        help='Percentil para cálculo automático (padrão: 90). Ignorado se limites manuais forem fornecidos.')
    parser.add_argument('--w-lim', type=int, default=None,
                        help='Limite manual para w (m/s). Ex.: --w-lim 15')
    parser.add_argument('--qcqi-lim', type=int, default=None,
                        help='Limite manual para qc+qi (g/kg). Ex.: --qcqi-lim 20')
    parser.add_argument('--qg-lim', type=int, default=None,
                        help='Limite manual para qg (g/kg). Ex.: --qg-lim 5')
    args = parser.parse_args()

    casos = ORDEM_CASOS

    # --- 1. Define os limites ---
    if args.w_lim is not None and args.qcqi_lim is not None and args.qg_lim is not None:
        # Modo manual: usa os valores fornecidos
        w_lim = args.w_lim
        qcqi_lim = args.qcqi_lim
        qg_lim = args.qg_lim
        print(f"Usando limites manuais:")
        print(f"  w_lim    = {w_lim} m/s")
        print(f"  qcqi_lim = {qcqi_lim} g/kg")
        print(f"  qg_lim   = {qg_lim} g/kg")
    else:
        # Modo automático: calcula via percentil
        print(f"Calculando percentil {args.percentil} dos valores...")
        w_lim, qcqi_lim, qg_lim = obter_percentis_globais(casos, percentil=args.percentil)
        print(f"  w_lim    = {w_lim} m/s  (percentil {args.percentil} do |w|)")
        print(f"  qcqi_lim = {qcqi_lim} g/kg (percentil {args.percentil} de qc+qi)")
        print(f"  qg_lim   = {qg_lim} g/kg (percentil {args.percentil} de qg)")

    # --- 2. Encontra o pico de w no CTRL ---
    print("\nLocalizando o pico de w no CTRL...")
    ctrl_data = carregar_dados("CTRL")
    if ctrl_data is None:
        print("ERRO: Dados do CTRL não encontrados.")
        return
    
    w_ctrl = ctrl_data['w']
    t_ctrl = ctrl_data['t']
    w_max_tempo = np.max(w_ctrl, axis=(1, 2))
    idx_peak = np.argmax(w_max_tempo)
    t_peak = t_ctrl[idx_peak]
    print(f"  Pico de w em t = {t_peak/60:.1f} min (índice {idx_peak})")

    # --- 3. Define os índices ao redor do pico ---
    delta_t = 600  # 10 min
    target_times = [t_peak + i * delta_t for i in range(-2, 3)]
    target_times = [max(0, min(t, t_ctrl[-1])) for t in target_times]
    idxs = []
    for tt in target_times:
        idx = np.argmin(np.abs(t_ctrl - tt))
        idxs.append(idx)
    idxs = sorted(set(idxs))
    print(f"  Tempos selecionados (min): {t_ctrl[idxs]/60}")

    # --- 4. Gera figuras ---
    print("\nGerando figuras...")
    for caso in casos:
        print(f"Processando {caso}...")
        gerar_figura_caso(caso, w_lim, qcqi_lim, qg_lim, idxs)
    
    print("\nConcluído.")


if __name__ == "__main__":
    main()