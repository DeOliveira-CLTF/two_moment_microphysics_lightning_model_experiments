# -*- coding: utf-8 -*-
"""
Pacote `microfisica`: modelo didatico de microfisica de nuvens.

PASSO 1 (implementado): chuva quente, dois momentos (qc,Nc,qr,Nr)
PASSO 2 (implementado): fase gelo (qi,Ni) -- nucleacao, deposicao/sublimacao,
                        congelamento, degelo, efeito Wegener-Bergeron-Findeisen
PASSO 3 (implementado): neve e graupel (qs,Ns,qg,Ng) -- riming, congelamento
                        de chuva, Hallett-Mossop, coleta cruzada, degelo

INTERFACE GENERICA: microfisica completa do Passo 3 desacoplada da classe de
                    coluna, para experimentos com dinamica externa.
"""

from . import constantes
from . import distribuicoes
from . import processos_chuva_quente
from . import processos_fase_gelo
from . import processos_fase_mista
from . import coluna_step1
from . import coluna_step2
from . import coluna_step3
from . import coluna_generica

from .coluna_generica import passo_microfisica_coluna
