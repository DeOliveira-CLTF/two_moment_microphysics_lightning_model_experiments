# -*- coding: utf-8 -*-
"""
estatisticas_avancadas_grupo3.py (CORRIGIDO)
=============================================

Calcula métricas avançadas para o Grupo 3:
- Partição entre gelo de nuvem (qi), neve (qs) e graupel (qg)
- Diagnósticos de McCaul (F3) e LPI* (se disponíveis)
- Testes de significância para essas métricas

Uso:
    python experiments/group3_process_ablation/estatisticas_avancadas_grupo3.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuracao_casos_grupo3 import ORDEM_CASOS

OUTPUT_ROOT = REPO_ROOT / "outputs" / "group3"


def carregar_dados_completos(caso):
    """Carrega todos os campos necessários do .npz."""
    caminho = OUTPUT_ROOT / caso / f"resultados_{caso}.npz"
    if not caminho.exists():
        return None
    data = np.load(caminho, allow_pickle=True)
    
    # Tempo
    t = data.get('t_s', data.get('t', None))
    if t is None:
        return None
    t_min = t / 60.0

    # Campos 2D (nt, nx, nz)
    w = data['w']
    qi = data.get('qi', np.zeros_like(w))
    qs = data.get('qs', np.zeros_like(w))
    qg = data.get('qg', np.zeros_like(w))
    qc = data.get('qc', np.zeros_like(w))

    # Diagnósticos de relâmpagos - podem não existir
    # Tentamos vários nomes possíveis
    f3 = data.get('lightning_mccaul_F3', None)
    if f3 is None:
        f3 = data.get('mccaul_F3', None)
    if f3 is None:
        f3 = data.get('F3', None)
    
    lpi = data.get('lightning_lpi_star', None)
    if lpi is None:
        lpi = data.get('lpi_star', None)
    if lpi is None:
        lpi = data.get('LPI', None)

    # Se não existirem, criar arrays de NaN com o mesmo shape de w (nt, nx)
    if f3 is None:
        f3 = np.full((w.shape[0], w.shape[1]), np.nan)
    if lpi is None:
        lpi = np.full((w.shape[0], w.shape[1]), np.nan)

    return {
        't_min': t_min,
        'w': w,
        'qi': qi,
        'qs': qs,
        'qg': qg,
        'qc': qc,
        'f3': f3,
        'lpi': lpi,
        'x': data['x_m'],
        'z': data['z_m'],
        'rho': data['rho0_1d'],
    }


def calcular_massas_integradas(dados):
    """Calcula massa total integrada de qi, qs, qg no domínio (kg/m)."""
    rho = dados['rho']
    dx = dados['x'][1] - dados['x'][0]
    dz = dados['z'][1] - dados['z'][0]
    area = dx * dz

    def integrar(campo):
        return np.sum(campo * rho[None, None, :], axis=(1, 2)) * area

    return {
        'qi_total': integrar(dados['qi']),
        'qs_total': integrar(dados['qs']),
        'qg_total': integrar(dados['qg']),
    }


def particao(massas):
    """Calcula frações de qi, qs, qg em relação ao total."""
    total = massas['qi_total'] + massas['qs_total'] + massas['qg_total']
    total_safe = np.maximum(total, 1e-12)
    return {
        'frac_qi': massas['qi_total'] / total_safe,
        'frac_qs': massas['qs_total'] / total_safe,
        'frac_qg': massas['qg_total'] / total_safe,
    }


def estatisticas_descritivas_series(series_dict):
    """Calcula média, desvio, máximo para cada série (ignorando NaN)."""
    stats_out = {}
    for nome, serie in series_dict.items():
        serie_clean = serie[~np.isnan(serie)]
        if len(serie_clean) == 0:
            stats_out[f'{nome}_mean'] = np.nan
            stats_out[f'{nome}_std'] = np.nan
            stats_out[f'{nome}_max'] = np.nan
            stats_out[f'{nome}_time_max'] = np.nan
        else:
            stats_out[f'{nome}_mean'] = np.mean(serie_clean)
            stats_out[f'{nome}_std'] = np.std(serie_clean)
            idx_max = np.argmax(serie_clean)
            stats_out[f'{nome}_max'] = serie_clean[idx_max]
            # Se tivermos a série de tempo, usamos o mesmo índice
            if 't_min' in series_dict:
                stats_out[f'{nome}_time_max'] = series_dict['t_min'][idx_max]
            else:
                stats_out[f'{nome}_time_max'] = np.nan
    return stats_out


def testes_significancia_avancados(series_por_caso, metrica):
    """
    Aplica teste t de Welch comparando cada ablação com o CTRL para uma métrica.
    series_por_caso: dict {caso: dict com as séries}
    """
    ctrl_series = series_por_caso['CTRL'][metrica]
    resultados = []
    for caso in series_por_caso:
        if caso == 'CTRL':
            continue
        dados = series_por_caso[caso][metrica]
        # Remove NaNs
        ctrl_clean = ctrl_series[~np.isnan(ctrl_series)]
        dados_clean = dados[~np.isnan(dados)]
        if len(ctrl_clean) < 2 or len(dados_clean) < 2:
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = stats.ttest_ind(ctrl_clean, dados_clean, equal_var=False)
        resultados.append({
            'caso': caso,
            't_stat': t_stat,
            'p_valor': p_val,
            'significativo_5%': p_val < 0.05 if not np.isnan(p_val) else False,
            'significativo_1%': p_val < 0.01 if not np.isnan(p_val) else False,
        })
    return pd.DataFrame(resultados)


def main():
    # Carrega dados de todos os casos
    todos_dados = {}
    for caso in ORDEM_CASOS:
        dados = carregar_dados_completos(caso)
        if dados is not None:
            todos_dados[caso] = dados
        else:
            print(f"[aviso] Caso {caso} não encontrado. Ignorado.")

    if 'CTRL' not in todos_dados:
        raise RuntimeError("CTRL não encontrado. Execute as simulações primeiro.")

    # Para cada caso, constrói um dicionário com as séries temporais
    series_por_caso = {}
    for caso, dados in todos_dados.items():
        # Máximos espaciais (sobre x e z)
        w_max = np.max(dados['w'], axis=(1, 2))
        qg_max = np.max(dados['qg'], axis=(1, 2)) * 1000  # g/kg
        qi_max = np.max(dados['qi'], axis=(1, 2)) * 1000
        qs_max = np.max(dados['qs'], axis=(1, 2)) * 1000

        # Massas integradas
        massas = calcular_massas_integradas(dados)
        qi_total = massas['qi_total']
        qs_total = massas['qs_total']
        qg_total = massas['qg_total']

        # Frações de partição (usando totais integrados)
        fracs = particao(massas)

        # Diagnósticos de relâmpagos: máximo em x (para cada tempo)
        # Se todos os valores forem NaN, a série ficará toda NaN
        f3_max_x = np.nanmax(dados['f3'], axis=1)
        lpi_max_x = np.nanmax(dados['lpi'], axis=1)

        series_por_caso[caso] = {
            't_min': dados['t_min'],
            'w_max': w_max,
            'qg_max': qg_max,
            'qi_max': qi_max,
            'qs_max': qs_max,
            'qi_total': qi_total,
            'qs_total': qs_total,
            'qg_total': qg_total,
            'frac_qi': fracs['frac_qi'],
            'frac_qs': fracs['frac_qs'],
            'frac_qg': fracs['frac_qg'],
            'f3_max': f3_max_x,
            'lpi_max': lpi_max_x,
        }

    # ---------- Estatísticas descritivas ----------
    desc_stats = {}
    for caso, series in series_por_caso.items():
        # Seleciona as séries para as quais queremos descritivas
        metricas = {
            'w_max': series['w_max'],
            'qg_max': series['qg_max'],
            'qi_total': series['qi_total'],
            'qs_total': series['qs_total'],
            'qg_total': series['qg_total'],
            'frac_qi': series['frac_qi'],
            'frac_qs': series['frac_qs'],
            'frac_qg': series['frac_qg'],
            'f3_max': series['f3_max'],
            'lpi_max': series['lpi_max'],
        }
        # Adiciona a série de tempo para saber o instante do pico
        metricas['t_min'] = series['t_min']
        desc_stats[caso] = estatisticas_descritivas_series(metricas)

    df_desc = pd.DataFrame(desc_stats).T
    df_desc = df_desc.round(2)

    print("\n=== ESTATÍSTICAS DESCRITIVAS AVANÇADAS ===")
    print(df_desc)

    # ---------- Testes de significância ----------
    # Para qg_max
    df_sig_qg = testes_significancia_avancados(series_por_caso, 'qg_max')
    print("\n=== TESTE DE SIGNIFICÂNCIA (qg_max vs CTRL) ===")
    print(df_sig_qg)

    # Para f3_max (se houver dados não-NaN)
    # Verifica se todos os casos têm f3_max não-NaN
    tem_f3 = any(not np.all(np.isnan(series_por_caso[caso]['f3_max'])) for caso in series_por_caso)
    if tem_f3:
        df_sig_f3 = testes_significancia_avancados(series_por_caso, 'f3_max')
        print("\n=== TESTE DE SIGNIFICÂNCIA (f3_max vs CTRL) ===")
        print(df_sig_f3)
        df_sig_f3.to_csv(OUTPUT_ROOT / 'testes_significancia_f3.csv', index=False)
    else:
        print("\n[aviso] Campos F3 não encontrados ou todos NaN. Pule teste.")

    # Para lpi_max
    tem_lpi = any(not np.all(np.isnan(series_por_caso[caso]['lpi_max'])) for caso in series_por_caso)
    if tem_lpi:
        df_sig_lpi = testes_significancia_avancados(series_por_caso, 'lpi_max')
        print("\n=== TESTE DE SIGNIFICÂNCIA (lpi_max vs CTRL) ===")
        print(df_sig_lpi)
        df_sig_lpi.to_csv(OUTPUT_ROOT / 'testes_significancia_lpi.csv', index=False)
    else:
        print("\n[aviso] Campos LPI* não encontrados ou todos NaN. Pule teste.")

    # ---------- Salvar arquivos CSV ----------
    df_desc.to_csv(OUTPUT_ROOT / 'estatisticas_descritivas_avancadas.csv')
    df_sig_qg.to_csv(OUTPUT_ROOT / 'testes_significancia_qg.csv', index=False)

    print(f"\nArquivos salvos em: {OUTPUT_ROOT}")

    # ---------- Tabela resumo para LaTeX ----------
    # Seleciona colunas principais
    colunas_para_tabela = ['qg_max_mean', 'qg_max_max', 'frac_qg_mean', 
                           'f3_max_mean', 'f3_max_max', 
                           'lpi_max_mean', 'lpi_max_max']
    # Filtra colunas que existem
    colunas_existentes = [c for c in colunas_para_tabela if c in df_desc.columns]
    tabela_latex = df_desc[colunas_existentes].copy()
    if not tabela_latex.empty:
        tabela_latex.columns = ['qg médio (g/kg)', 'qg máx (g/kg)', 'frac qg', 
                                'F3 médio', 'F3 máx', 'LPI* médio', 'LPI* máx'][:len(colunas_existentes)]
        tabela_latex.to_csv(OUTPUT_ROOT / 'tabela_resumo_latex.csv')
        print("\n=== TABELA RESUMO (médias e máximos) ===")
        print(tabela_latex.round(3))


if __name__ == "__main__":
    main()