# -*- coding: utf-8 -*-
"""Teste de compatibilidade das novas opcoes microfisicas."""

from dataclasses import replace

from microfisica import OpcoesMicrofisica


def test_todas_as_opcoes_ligadas_por_padrao():
    op = OpcoesMicrofisica()
    assert all(
        (
            op.nucleacao_gelo,
            op.deposicao,
            op.condensacao_liquida,
            op.congelamento_nuvem,
            op.congelamento_chuva,
            op.riming,
            op.hallett_mossop,
            op.coleta_chuva_por_gelo,
            op.gelo_para_neve,
            op.degelo,
            op.chuva_quente,
        )
    )


def test_ablacao_nao_altera_objeto_base():
    base = OpcoesMicrofisica()
    sem_riming = replace(base, riming=False)
    assert base.riming is True
    assert sem_riming.riming is False
