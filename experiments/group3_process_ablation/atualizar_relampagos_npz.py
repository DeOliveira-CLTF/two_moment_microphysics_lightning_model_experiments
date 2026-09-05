# -*- coding: utf-8 -*-
"""
atualizar_relampagos_npz.py
============================

Atualiza os arquivos .npz já salvos pelo executar_grupo3.py,
adicionando os diagnósticos de relâmpagos (McCaul F3 e LPI*) 
sem precisar rerodar as simulações.

Uso:
    python experiments/group3_process_ablation/atualizar_relampagos_npz.py
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightning import diagnosticar_relampagos_2d
from configuracao_casos_grupo3 import ORDEM_CASOS

OUTPUT_ROOT = ROOT / "outputs" / "group3"


def atualizar_caso(caso):
    """Carrega o .npz de um caso, calcula os diagnósticos e salva novamente."""
    caminho = OUTPUT_ROOT / caso / f"resultados_{caso}.npz"
    if not caminho.exists():
        print(f"[aviso] {caminho} não encontrado. Pule.")
        return False

    # Carrega os dados
    dados = np.load(caminho, allow_pickle=True)
    
    # Verifica se já tem os campos de raio
    if "lightning_mccaul_F3" in dados:
        print(f"[info] {caso} já contém diagnósticos de raios. Pule.")
        return True

    # Reconstrói a estrutura necessária para o diagnosticar_relampagos_2d
    # O esperado é um dicionário com "frames" contendo T, w, qc, qr, qi, qs, qg
    frames = {}
    for chave in ["T", "w", "qc", "qr", "qi", "qs", "qg"]:
        if chave in dados:
            frames[chave] = dados[chave]
        else:
            # Se faltar algum campo, cria um array de zeros (mas não deveria)
            print(f"[aviso] {caso} não tem {chave}. Criando zeros.")
            frames[chave] = np.zeros(dados["w"].shape)

    # Adiciona a chave 't' (o diagnosticador espera 't' ou 't_s')
    if 't_s' in dados:
        frames['t'] = dados['t_s']
    elif 't' in dados:
        frames['t'] = dados['t']
    else:
        frames['t'] = np.arange(dados['w'].shape[0]) * 300.0  # fallback

    resultado = {
        "frames": frames,
        "x_m": dados["x_m"],
        "z_m": dados["z_m"],
        "rho0_1d": dados["rho0_1d"],
        # Outros campos não são necessários para o diagnóstico
    }

    # Calcula os diagnósticos
    diag = diagnosticar_relampagos_2d(resultado)
    if diag is None:
        print(f"[erro] diagnosticar_relampagos_2d retornou None para {caso}")
        return False

    # Converte os dados carregados para um dicionário mutável
    dados_novos = {chave: dados[chave] for chave in dados.files}
    
    # Adiciona os novos campos
    for chave, valor in diag.items():
        dados_novos[chave] = valor

    # Salva novamente (sobrescreve)
    np.savez_compressed(caminho, **dados_novos)
    print(f"[OK] {caso} atualizado com diagnósticos de raios.")
    return True


def main():
    print("Atualizando arquivos .npz com diagnósticos de relâmpagos...")
    print("(Isso NÃO reroda as simulações, apenas calcula F3 e LPI* a partir dos campos salvos.)")
    
    for caso in ORDEM_CASOS:
        atualizar_caso(caso)
    
    print("\nConcluído! Agora os .npz contêm lightning_mccaul_F3 e lightning_lpi_star.")


if __name__ == "__main__":
    main()