# -*- coding: utf-8 -*-
"""Verifica a associação entre estados salvos e tempo físico no Passo 3."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from microfisica.coluna_step3 import ColunaFaseMista


class HistoricoTemporalStep3Tests(unittest.TestCase):
    def test_historico_contem_estado_inicial_e_final_reais(self):
        coluna = ColunaFaseMista(nz=20, dz=100.0)
        coluna.inserir_nuvem(5, 10, qc_valor=1.0e-4, Nc_valor=1.0e8)
        qc_inicial = coluna.qc.copy()
        qg_inicial = coluna.qg.copy()
        historico = coluna.integrar(10.0, dt=2.0, salvar_a_cada=4.0)
        np.testing.assert_allclose(historico["t"], [0.0, 4.0, 8.0, 10.0])
        np.testing.assert_array_equal(historico["qc"][0], qc_inicial)
        np.testing.assert_array_equal(historico["qg"][0], qg_inicial)
        self.assertEqual(historico["t"][-1], 10.0)

    def test_passo_final_parcial_atinge_tempo_total_exato(self):
        coluna = ColunaFaseMista(nz=20, dz=100.0)
        historico = coluna.integrar(5.0, dt=2.0, salvar_a_cada=4.0)
        np.testing.assert_allclose(historico["t"], [0.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
