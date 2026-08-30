# -*- coding: utf-8 -*-
"""Testes da interface funcional de microfisica."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from microfisica.coluna_generica import passo_microfisica_coluna
from microfisica.coluna_step3 import ColunaFaseMista


def test_passo_microfisica_coluna_reproduz_processos_locais_step3():
    coluna = ColunaFaseMista(nz=12, dz=100.0)
    coluna.inserir_nuvem(2, 8, qc_valor=8.0e-4, Nc_valor=1.0e8)
    coluna.inserir_gelo(6, 10, qi_valor=1.0e-5, Ni_valor=5.0e3)
    coluna.inserir_neve(7, 10, qs_valor=2.0e-5, Ns_valor=2.0e3)
    coluna.inserir_graupel(5, 9, qg_valor=1.0e-5, Ng_valor=1.0e3)

    estado = {
        nome: getattr(coluna, nome).copy()
        for nome in ("T", "p", "rho", "qv", "qc", "Nc", "qr", "Nr", "qi", "Ni", "qs", "Ns", "qg", "Ng")
    }

    dt = 2.0
    resultado = passo_microfisica_coluna(
        dt,
        estado["T"],
        estado["p"],
        estado["rho"],
        estado["qv"],
        estado["qc"],
        estado["Nc"],
        estado["qr"],
        estado["Nr"],
        estado["qi"],
        estado["Ni"],
        estado["qs"],
        estado["Ns"],
        estado["qg"],
        estado["Ng"],
    )

    coluna._passo_processos_locais(dt)

    nomes_retorno = ("T", "qv", "qc", "Nc", "qr", "Nr", "qi", "Ni", "qs", "Ns", "qg", "Ng")
    for nome, calculado in zip(nomes_retorno, resultado):
        assert np.allclose(calculado, getattr(coluna, nome), rtol=1.0e-12, atol=1.0e-15)


def test_passo_microfisica_coluna_nao_altera_arrays_de_entrada():
    coluna = ColunaFaseMista(nz=8, dz=100.0)
    coluna.inserir_nuvem(2, 5)

    campos = [
        coluna.T,
        coluna.p,
        coluna.rho,
        coluna.qv,
        coluna.qc,
        coluna.Nc,
        coluna.qr,
        coluna.Nr,
        coluna.qi,
        coluna.Ni,
        coluna.qs,
        coluna.Ns,
        coluna.qg,
        coluna.Ng,
    ]
    copias = [campo.copy() for campo in campos]

    passo_microfisica_coluna(1.0, *campos)

    for campo, copia in zip(campos, copias):
        assert np.array_equal(campo, copia)
