# -*- coding: utf-8 -*-
"""
teste_conservacao_massa.py
============================

Teste de conservação de massa para os experimentos do Grupo 3.

Verifica se a massa total de água (vapor + líquido + gelo) no domínio
se conserva ao longo do tempo, considerando a perda por precipitação
que atravessa a base do domínio.

Uso:
    python experiments/group3_process_ablation/teste_conservacao_massa.py
    python experiments/group3_process_ablation/teste_conservacao_massa.py --caso CTRL
    python experiments/group3_process_ablation/teste_conservacao_massa.py --todos
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinamica_2d import campo_Vt_chuva, campo_Vt_neve, campo_Vt_graupel
from configuracao_casos_grupo3 import ORDEM_CASOS

OUTPUT_ROOT = REPO_ROOT / "outputs" / "group3"


def carregar_dados(caso):
    """Carrega os dados de um caso específico."""
    caminho = OUTPUT_ROOT / caso / f"resultados_{caso}.npz"
    if not caminho.exists():
        return None
    dados = np.load(caminho, allow_pickle=True)
    return dados


def calcular_conservacao_massa(dados, caso, plot=False):
    """
    Calcula a conservação de massa para um caso.

    Retorna:
        dict com: {
            't_min': tempos em minutos,
            'M_total': massa total no domínio (kg/m),
            'Precip_acum': precipitação acumulada (kg/m),
            'M_total_plus_precip': M_total + Precip_acum,
            'variacao_relativa_max': variação relativa máxima (%),
            'conservado': True/False (se variação < 1e-3 %)
        }
    """
    # Extrai campos
    t_s = dados['t_s']  # tempos em segundos
    x = dados['x_m']
    z = dados['z_m']
    rho = dados['rho0_1d']

    # Categorias de água
    qv = dados['qv']      # vapor
    qc = dados['qc']      # água de nuvem
    qr = dados['qr']      # chuva
    qi = dados['qi']      # gelo de nuvem
    qs = dados['qs']      # neve
    qg = dados['qg']      # graupel

    nt = len(t_s)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    area = dx * dz

    # --- 1. Calcula massa total no domínio para cada tempo ---
    # M_total(t) = ∫ ρ [qv + qc + qr + qi + qs + qg] dV
    # Soma sobre x e z, multiplica por rho(z) e área
    q_total = qv + qc + qr + qi + qs + qg  # (nt, nx, nz)
    M_total = np.sum(q_total * rho[None, None, :], axis=(1, 2)) * area  # (nt,)

    # --- 2. Calcula precipitação acumulada (fluxo que sai pela base) ---
    # Escolhe o nível mais baixo (k=0). Pode ser necessário usar k=1 se houver
    # problemas de contorno, mas vamos usar k=0.
    k_base = 0

    # Velocidades terminais na base
    Vt_qr, _ = campo_Vt_chuva(qr[:, :, k_base], dados['Nr'][:, :, k_base], rho[k_base])
    Vt_qs, _ = campo_Vt_neve(qs[:, :, k_base], dados['Ns'][:, :, k_base], rho[k_base])
    Vt_qg, _ = campo_Vt_graupel(qg[:, :, k_base], dados['Ng'][:, :, k_base], rho[k_base])

    # Fluxo de massa na base (kg m^-2 s^-1) para cada tempo e posição x
    fluxo_base = rho[k_base] * (
        qr[:, :, k_base] * Vt_qr +
        qs[:, :, k_base] * Vt_qs +
        qg[:, :, k_base] * Vt_qg
    )  # (nt, nx)

    # Integra sobre x para obter fluxo total por unidade de profundidade y (kg m^-1 s^-1)
    fluxo_total_x = np.sum(fluxo_base, axis=1) * dx  # (nt,)

    # Acumula no tempo (usando o intervalo entre saídas)
    dt_salvo = t_s[1] - t_s[0] if nt > 1 else 1.0
    Precip_acum = np.cumsum(fluxo_total_x) * dt_salvo  # (nt,)

    # Normaliza: no instante inicial, a precipitação acumulada é zero
    Precip_acum = Precip_acum - Precip_acum[0]

    # --- 3. Verifica conservação ---
    M_total_plus_precip = M_total + Precip_acum
    M_inicial = M_total[0]  # massa inicial no domínio

    # Variação relativa em relação ao valor inicial
    variacao_relativa = (M_total_plus_precip - M_inicial) / M_inicial * 100  # em %

    variacao_max = np.max(np.abs(variacao_relativa))
    conservado = variacao_max < 1e-3  # tolerância de 0.001%

    # --- 4. Resultados ---
    resultado = {
        't_min': t_s / 60.0,
        'M_total': M_total,
        'Precip_acum': Precip_acum,
        'M_total_plus_precip': M_total_plus_precip,
        'variacao_relativa': variacao_relativa,
        'variacao_relativa_max': variacao_max,
        'conservado': conservado,
        'M_inicial': M_inicial,
        'M_final': M_total_plus_precip[-1],
    }

    # --- 5. Gráfico opcional ---
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(resultado['t_min'], M_total, label='Massa no domínio', color='blue')
        axes[0].plot(resultado['t_min'], Precip_acum, label='Precipitação acumulada', color='orange')
        axes[0].plot(resultado['t_min'], M_total_plus_precip,
                     label='M_total + Precip_acum', color='green', linestyle='--')
        axes[0].axhline(M_inicial, color='red', linestyle=':', label='Valor inicial')
        axes[0].set_ylabel('Massa (kg/m)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_title(f'Conservação de massa - {caso}')

        axes[1].plot(resultado['t_min'], variacao_relativa, color='purple')
        axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[1].axhline(1e-3, color='red', linestyle='--', label='Tolerância (0.001%)')
        axes[1].axhline(-1e-3, color='red', linestyle='--')
        axes[1].set_ylabel('Variação relativa (%)')
        axes[1].set_xlabel('Tempo (min)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        saida_fig = OUTPUT_ROOT / caso / f'conservacao_massa_{caso}.png'
        plt.savefig(saida_fig, dpi=150)
        plt.close()
        print(f"  Figura salva: {saida_fig}")

    return resultado


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Teste de conservação de massa')
    parser.add_argument('--caso', type=str, default='CTRL',
                        help='Caso a testar (padrão: CTRL)')
    parser.add_argument('--todos', action='store_true',
                        help='Testa todos os casos da lista ORDEM_CASOS')
    args = parser.parse_args()

    if args.todos:
        casos = ORDEM_CASOS
    else:
        casos = [args.caso]

    print("\n" + "="*70)
    print("TESTE DE CONSERVAÇÃO DE MASSA")
    print("="*70)

    for caso in casos:
        print(f"\nProcessando caso: {caso}")
        dados = carregar_dados(caso)
        if dados is None:
            print(f"  [aviso] Dados não encontrados para {caso}. Pule.")
            continue

        resultado = calcular_conservacao_massa(dados, caso, plot=True)

        print(f"  Massa inicial no domínio: {resultado['M_inicial']:.6f} kg/m")
        print(f"  Massa final (domínio + precipitação): {resultado['M_final']:.6f} kg/m")
        print(f"  Variação relativa máxima: {resultado['variacao_relativa_max']:.6f} %")
        print(f"  Conservação: {'OK' if resultado['conservado'] else 'FALHA'}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()