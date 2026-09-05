# -*- coding: utf-8 -*-
"""
atualizar_f3_offline.py
=======================

Lê os arquivos .npz já existentes, calcula os diagnósticos de raios 
(F3 de McCaul e LPI*) usando o módulo standalone da pasta do Grupo 3, 
e salva os resultados de volta no mesmo arquivo .npz.
"""

import sys
from pathlib import Path
import numpy as np

# Garante que o Python encontre os módulos locais
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configuracao_casos_grupo3 import ORDEM_CASOS
from diagnosticos_relampago_grupo3 import diagnosticos_2d_para_caso

OUTPUT_ROOT = REPO_ROOT / "outputs" / "group3"

def main():
    print("Iniciando cálculo offline de McCaul (F3) e LPI*...\n")
    
    for caso in ORDEM_CASOS:
        caminho_npz = OUTPUT_ROOT / caso / f"resultados_{caso}.npz"
        
        if not caminho_npz.exists():
            print(f"[AVISO] Caso {caso} não encontrado. Pulando...")
            continue
            
        print(f"Processando: {caso}")
        
        # 1. Carrega os dados existentes
        dados_raw = np.load(caminho_npz, allow_pickle=True)
        dados = dict(dados_raw)  # Converte para dicionário mutável
        
        # 2. Monta o dicionário de frames exigido pelo diagnosticos_2d_para_caso
        frames = {
            "T": dados["T"],
            "w": dados["w"],
            "qc": dados["qc"],
            "qr": dados["qr"],
            "qi": dados["qi"],
            "qs": dados["qs"],
            "qg": dados["qg"]
        }
        z_m = dados["z_m"]
        rho0_1d = dados["rho0_1d"]
        
        # 3. Calcula as matrizes (tempo x x)
        F3_txx, LPI_txx = diagnosticos_2d_para_caso(frames, z_m, rho0_1d)
        
        # 4. Injeta os diagnósticos de volta no dicionário
        # O script de estatísticas procura por 'lightning_mccaul_F3' e 'lightning_lpi_star'
        dados["lightning_mccaul_F3"] = F3_txx
        dados["lightning_lpi_star"] = LPI_txx
        
        # 5. Sobrescreve o .npz com os dados atualizados
        np.savez_compressed(caminho_npz, **dados)
        print(f"  -> F3 e LPI* gravados com sucesso em {caminho_npz.name}\n")

    print("Cálculo concluído. Rode estatisticas_avancadas_grupo3.py novamente.")

if __name__ == "__main__":
    main()