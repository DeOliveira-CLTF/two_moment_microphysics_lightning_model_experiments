# -*- coding: utf-8 -*-
"""Configuracao dos processos microfisicos usados nos experimentos.

O objetivo deste modulo e permitir experimentos de ablacao sem editar o
codigo das parametrizacoes. Todos os processos ficam ligados por padrao,
reproduzindo o comportamento original de ``passo_microfisica_coluna``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OpcoesMicrofisica:
    """Chaves logicas para ligar/desligar grupos de processos.

    Os nomes foram escolhidos para corresponder aos experimentos cientificos
    do Grupo 3. Quando todas as opcoes sao ``True``, a sequencia de processos
    e a mesma da versao original da coluna generica.
    """

    nucleacao_gelo: bool = True
    deposicao: bool = True
    condensacao_liquida: bool = True
    congelamento_nuvem: bool = True
    congelamento_chuva: bool = True
    riming: bool = True
    hallett_mossop: bool = True
    coleta_chuva_por_gelo: bool = True
    gelo_para_neve: bool = True
    degelo: bool = True
    chuva_quente: bool = True
