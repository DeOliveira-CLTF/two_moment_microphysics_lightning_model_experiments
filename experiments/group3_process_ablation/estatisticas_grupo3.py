# -*- coding: utf-8 -*-
"""
estatisticas_grupo3.py
=======================

Calcula estatísticas quantitativas para os experimentos do Grupo 3,
incluindo testes de significância, correlações e métricas agregadas.

Uso:
    python experiments/group3_process_ablation/estatisticas_grupo3.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuracao_casos_grupo3 import ORDEM_CASOS

OUTPUT_ROOT = REPO_ROOT / "outputs" / "group3"


def carregar_series(caso):
    """Carrega as séries temporais de w_max, qg_max, qcqi_max do caso."""
    caminho = OUTPUT_ROOT / caso / f"resultados_{caso}.npz"
    if not caminho.exists():
        return None
    data = np.load(caminho, allow_pickle=True)
    # Identifica tempo
    t = data.get('t_s', data.get('t', None))
    if t is None:
        return None
    t_min = t / 60.0
    # Extrai máximos sobre x,z
    w_max = np.max(data['w'], axis=(1, 2))
    qcqi = data.get('qc', 0) + data.get('qi', 0)
    qcqi_max = np.max(qcqi, axis=(1, 2)) * 1000  # g/kg
    qg = data.get('qg', 0) * 1000  # g/kg
    qg_max = np.max(qg, axis=(1, 2))
    return {
        't_min': t_min,
        'w_max': w_max,
        'qcqi_max': qcqi_max,
        'qg_max': qg_max,
    }


def estatisticas_descritivas(series):
    """Retorna média, desvio, máximo, tempo do pico para w_max, qg_max, qcqi_max."""
    stats_dict = {}
    for var in ['w_max', 'qcqi_max', 'qg_max']:
        s = series[var]
        stats_dict[f'{var}_mean'] = np.mean(s)
        stats_dict[f'{var}_std'] = np.std(s)
        stats_dict[f'{var}_max'] = np.max(s)
        idx_max = np.argmax(s)
        stats_dict[f'{var}_time_max_min'] = series['t_min'][idx_max]
    return stats_dict


def testes_significancia(todas_series):
    """
    Para cada caso (exceto CTRL), aplica teste t de Student comparando
    as séries temporais de w_max com o CTRL.
    Retorna um DataFrame com p-valor e estatística t.
    """
    if 'CTRL' not in todas_series:
        raise ValueError("CTRL não encontrado no dicionário de séries.")
    ctrl = todas_series['CTRL']['w_max']
    resultados = []
    for caso, series in todas_series.items():
        if caso == 'CTRL':
            continue
        dados = series['w_max']
        # Teste t para duas amostras independentes (assumindo variâncias diferentes)
        # Se os comprimentos forem diferentes, o teste ainda funciona (ttest_ind com comprimentos diferentes)
        t_stat, p_val = stats.ttest_ind(ctrl, dados, equal_var=False)
        resultados.append({
            'caso': caso,
            't_stat': t_stat,
            'p_valor': p_val,
            'significativo_5%': p_val < 0.05,
            'significativo_1%': p_val < 0.01,
        })
    return pd.DataFrame(resultados)


def correlacao_entre_variaveis(todas_series):
    """Calcula correlação de Pearson entre w_max, qg_max e qcqi_max para cada caso."""
    correlacoes = []
    for caso, series in todas_series.items():
        corr_w_qg = stats.pearsonr(series['w_max'], series['qg_max'])[0]
        corr_w_qcqi = stats.pearsonr(series['w_max'], series['qcqi_max'])[0]
        corr_qg_qcqi = stats.pearsonr(series['qg_max'], series['qcqi_max'])[0]
        correlacoes.append({
            'caso': caso,
            'corr_w_qg': corr_w_qg,
            'corr_w_qcqi': corr_w_qcqi,
            'corr_qg_qcqi': corr_qg_qcqi,
        })
    return pd.DataFrame(correlacoes).set_index('caso')


def main():
    # Carrega dados de todos os casos
    todas_series = {}
    for caso in ORDEM_CASOS:
        series = carregar_series(caso)
        if series is not None:
            todas_series[caso] = series
        else:
            print(f"[aviso] Caso {caso} não encontrado. Ignorado.")

    if 'CTRL' not in todas_series:
        raise RuntimeError("CTRL não encontrado. Execute as simulações primeiro.")

    # 1. Estatísticas descritivas
    descritivas = []
    for caso, series in todas_series.items():
        est = estatisticas_descritivas(series)
        est['caso'] = caso
        descritivas.append(est)
    df_desc = pd.DataFrame(descritivas).set_index('caso')
    print("\n=== ESTATÍSTICAS DESCRITIVAS ===")
    print(df_desc.round(3))
    df_desc.to_csv(OUTPUT_ROOT / 'estatisticas_descritivas.csv')

    # 2. Testes de significância (comparação com CTRL)
    df_sig = testes_significancia(todas_series)
    print("\n=== TESTES DE SIGNIFICÂNCIA (vs CTRL) ===")
    print(df_sig)
    df_sig.to_csv(OUTPUT_ROOT / 'testes_significancia.csv', index=False)

    # 3. Correlações entre variáveis
    df_corr = correlacao_entre_variaveis(todas_series)
    print("\n=== CORRELAÇÕES ENTRE VARIÁVEIS ===")
    print(df_corr.round(3))
    df_corr.to_csv(OUTPUT_ROOT / 'correlacoes.csv')

    # 4. Gráficos
    # 4a. Boxplot de w_max para todos os casos
    plt.figure(figsize=(10, 6))
    dados_box = [todas_series[caso]['w_max'] for caso in ORDEM_CASOS if caso in todas_series]
    labels = [c for c in ORDEM_CASOS if c in todas_series]
    plt.boxplot(dados_box, labels=labels)
    plt.ylabel('w_max (m/s)')
    plt.title('Distribuição de w_max ao longo do tempo – Grupo 3')
    plt.grid(alpha=0.3)
    plt.savefig(OUTPUT_ROOT / 'boxplot_wmax_grupo3.png', dpi=150)
    plt.close()

    # 4b. Scatter plot: w_max vs qg_max (para CTRL e SEM_RIMING)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    casos_plot = ['CTRL', 'SEM-RIMING']  # ATENÇÃO: o nome no dicionário é 'SEM-RIMING' ou 'SEM_RIMING'?
    # Verifica se o nome existe; caso contrário, usa 'SEM_RIMING'
    for i, caso in enumerate(casos_plot):
        if caso not in todas_series:
            caso = 'SEM_RIMING'  # tentativa com underscore
        if caso not in todas_series:
            continue
        s = todas_series[caso]
        axes[i].scatter(s['w_max'], s['qg_max'], alpha=0.6)
        axes[i].set_xlabel('w_max (m/s)')
        axes[i].set_ylabel('qg_max (g/kg)')
        # Obtém correlação do dataframe
        corr_val = df_corr.loc[caso, 'corr_w_qg'] if caso in df_corr.index else np.nan
        axes[i].set_title(f'{caso}\ncorr = {corr_val:.3f}')
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / 'scatter_w_vs_qg.png', dpi=150)
    plt.close()

    # 4c. Evolução temporal comparativa (w_max)
    plt.figure(figsize=(10, 6))
    casos_selecionados = ['CTRL', 'SEM-RIMING', 'SEM-DEP']
    for caso in casos_selecionados:
        if caso not in todas_series:
            caso = caso.replace('-', '_')
        if caso not in todas_series:
            continue
        s = todas_series[caso]
        plt.plot(s['t_min'], s['w_max'], label=caso)
    plt.xlabel('Tempo (min)')
    plt.ylabel('w_max (m/s)')
    plt.title('Evolução de w_max – CTRL vs ablações selecionadas')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUTPUT_ROOT / 'evolucao_wmax_selecionados.png', dpi=150)
    plt.close()

    print("\nAnálise estatística concluída. Arquivos gerados em:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()