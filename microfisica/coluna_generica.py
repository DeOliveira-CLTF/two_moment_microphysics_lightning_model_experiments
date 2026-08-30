# -*- coding: utf-8 -*-
"""
coluna_generica.py
====================

Versao "standalone" (independente de qualquer classe de coluna) da
fisica completa de microfisica do Passo 3 (mesma sequencia de
processos de `coluna_step3.ColunaFaseMista._passo_processos_locais`),
para ser chamada por QUALQUER modelo externo que forneca seus proprios
campos de T, p, rho a cada passo de tempo -- por exemplo um modelo
dinamico 2D (vorticidade-funcao de corrente), em vez de vir de um
perfil hidrostatico interno como as classes ColunaChuvaQuente/
ColunaFaseGelo/ColunaFaseMista fazem.

Isso permite acoplar a microfisica de 6 categorias (qv,qc,qr,qi,qs,qg,
com Nc,Nr,Ni,Ns,Ng prognosticos) a um nucleo dinamico que resolve a
circulacao explicitamente (ex.: nuvem_2d_thompson.py), em vez do
ajuste de saturacao simplificado de 1 momento usado no modelo 2D
original do curso de Conveccao Atmosferica.

USO TIPICO (uma coluna x do dominio 2D, a cada passo de tempo):

    T_novo, qv_novo, qc_novo, Nc_novo, qr_novo, Nr_novo, \\
    qi_novo, Ni_novo, qs_novo, Ns_novo, qg_novo, Ng_novo = \\
        passo_microfisica_coluna(dt, T, p, rho, qv, qc, Nc, qr, Nr,
                                  qi, Ni, qs, Ns, qg, Ng)

Todos os argumentos (exceto dt) sao arrays numpy 1D de mesmo tamanho
(nz,), representando UMA coluna vertical num dado instante de tempo.

"""
# -*- coding: utf-8 -*-
"""Microfisica completa do Passo 3 desacoplada da coluna estatica.

Esta versao preserva a sequencia fisica do arquivo original do professor e
acrescenta somente dois controles necessarios aos experimentos:

1. ``nc_ativacao_kg1``: concentracao de goticulas usada quando a agua de
   nuvem aparece pela primeira vez. Isso permite os casos N-LOW/CTRL/N-HIGH.
2. ``processos``: objeto :class:`OpcoesMicrofisica` para os experimentos de
   ablacao do Grupo 3.

Quando ``nc_ativacao_kg1=1.e8`` e todas as opcoes estao ligadas, o
comportamento e equivalente ao da versao anterior.
"""

import numpy as np

from .configuracao import OpcoesMicrofisica
from .constantes import (
    Rd,
    g,
    cp,
    Lv,
    Ls,
    rho_w,
    rho_i,
    rho_s,
    rho_g,
    MU_ICE,
    MU_SNOW,
    QMIN,
    NMIN,
    gamma_func,
)
from .processos_chuva_quente import (
    Pccnd,
    Pccnr,
    Pracw,
    Pr_self,
    Prevp,
    velocidade_terminal_chuva,
)
from .processos_fase_gelo import (
    Pidsn,
    Pidep,
    Pifzc,
    Pimlt,
    Pi_iacw,
    velocidade_terminal_gelo,
)
from .processos_fase_mista import (
    Picns,
    Psdep,
    Pgdep,
    Pgfzr,
    Pmlt,
    Pispl,
    colecao_continua,
    velocidade_terminal_neve,
    velocidade_terminal_graupel,
)
from .distribuicoes import lambda_gama, diametro_medio_numero


def passo_microfisica_coluna(
    dt,
    T,
    p,
    rho,
    qv,
    qc,
    Nc,
    qr,
    Nr,
    qi,
    Ni,
    qs,
    Ns,
    qg,
    Ng,
    evap_chuva=True,
    nc_ativacao_kg1=1.0e8,
    processos=None,
):
    """Aplica um passo local da microfisica completa a uma coluna vertical.

    Parameters
    ----------
    dt : float
        Passo de tempo [s].
    T, p, rho : array_like
        Temperatura [K], pressao [Pa] e densidade do ar [kg m-3].
    qv, qc, qr, qi, qs, qg : array_like
        Razoes de mistura [kg kg-1].
    Nc, Nr, Ni, Ns, Ng : array_like
        Concentracoes numericas [kg-1].
    evap_chuva : bool
        Liga/desliga ``Prevp`` quando os processos de chuva quente estao
        ativos.
    nc_ativacao_kg1 : float
        Concentracao atribuida a ``Nc`` quando ocorre condensacao e ainda nao
        existem goticulas. E o parametro usado no Grupo 1.
    processos : OpcoesMicrofisica or None
        Chaves de ativacao dos grupos de processos. ``None`` liga tudo.

    Returns
    -------
    tuple of ndarray
        ``T, qv, qc, Nc, qr, Nr, qi, Ni, qs, Ns, qg, Ng`` atualizados.
    """

    if processos is None:
        processos = OpcoesMicrofisica()
    if not isinstance(processos, OpcoesMicrofisica):
        raise TypeError("processos deve ser OpcoesMicrofisica ou None")

    nc_ativacao_kg1 = float(nc_ativacao_kg1)
    if not np.isfinite(nc_ativacao_kg1) or nc_ativacao_kg1 <= 0.0:
        raise ValueError("nc_ativacao_kg1 deve ser positivo e finito")

    arrays = [T, p, rho, qv, qc, Nc, qr, Nr, qi, Ni, qs, Ns, qg, Ng]
    nz = len(T)
    if any(len(a) != nz for a in arrays):
        raise ValueError("todos os perfis devem possuir o mesmo tamanho")

    T_out = np.asarray(T, dtype=float).copy()
    qv_out = np.asarray(qv, dtype=float).copy()
    qc_out = np.asarray(qc, dtype=float).copy()
    Nc_out = np.asarray(Nc, dtype=float).copy()
    qr_out = np.asarray(qr, dtype=float).copy()
    Nr_out = np.asarray(Nr, dtype=float).copy()
    qi_out = np.asarray(qi, dtype=float).copy()
    Ni_out = np.asarray(Ni, dtype=float).copy()
    qs_out = np.asarray(qs, dtype=float).copy()
    Ns_out = np.asarray(Ns, dtype=float).copy()
    qg_out = np.asarray(qg, dtype=float).copy()
    Ng_out = np.asarray(Ng, dtype=float).copy()

    p = np.asarray(p, dtype=float)
    rho = np.asarray(rho, dtype=float)

    for k in range(nz):
        Tk, pk, rhok = T_out[k], p[k], rho[k]
        qvk = qv_out[k]
        qck, Nck = qc_out[k], Nc_out[k]
        qrk, Nrk = qr_out[k], Nr_out[k]
        qik, Nik = qi_out[k], Ni_out[k]
        qsk, Nsk = qs_out[k], Ns_out[k]
        qgk, Ngk = qg_out[k], Ng_out[k]

        # Estado de trabalho: sempre inicializado, inclusive quando um
        # processo e desligado.
        T_new = Tk
        qv_new = qvk
        qc_new, Nc_new = qck, Nck
        qr_new, Nr_new = qrk, Nrk
        qi_new, Ni_new = qik, Nik
        qs_new, Ns_new = qsk, Nsk
        qg_new, Ng_new = qgk, Ngk

        # ===== 1) Pidsn: nucleacao primaria de gelo =====
        if processos.nucleacao_gelo:
            dqi_idsn, dNi_idsn = Pidsn(Ni_new, T_new, rhok, dt)
            dqi_idsn *= dt
            dNi_idsn *= dt
            qi_new += dqi_idsn
            Ni_new += dNi_idsn
            qv_new -= dqi_idsn
            T_new += (Ls / cp) * dqi_idsn

        # ===== 2) Pidep, Psdep, Pgdep: deposicao/sublimacao =====
        if processos.deposicao:
            dqi_idep = 0.0
            if Ni_new > NMIN:
                dqi_idep = max(
                    Pidep(qv_new, qi_new, Ni_new, T_new, pk, rhok) * dt,
                    -qi_new,
                )
            qi_new += dqi_idep
            qv_new -= dqi_idep
            T_new += (Ls / cp) * dqi_idep

            dqs_sdep = 0.0
            if Ns_new > NMIN:
                dqs_sdep = max(
                    Psdep(qv_new, qs_new, Ns_new, T_new, pk, rhok) * dt,
                    -qs_new,
                )
            qs_new += dqs_sdep
            qv_new -= dqs_sdep
            T_new += (Ls / cp) * dqs_sdep

            dqg_gdep = 0.0
            if Ng_new > NMIN:
                dqg_gdep = max(
                    Pgdep(qv_new, qg_new, Ng_new, T_new, pk, rhok) * dt,
                    -qg_new,
                )
            qg_new += dqg_gdep
            qv_new -= dqg_gdep
            T_new += (Ls / cp) * dqg_gdep

        # ===== 3) Pccnd: condensacao/evaporacao liquida =====
        if processos.condensacao_liquida:
            dqc_ccnd, dqv_ccnd, dT_ccnd = Pccnd(
                qv_new, qc_new, T_new, pk, dt
            )
            qc_new += dqc_ccnd
            qv_new += dqv_ccnd
            T_new += dT_ccnd
            if dqc_ccnd > 0.0 and Nc_new <= NMIN:
                Nc_new = nc_ativacao_kg1

        # ===== 4) Pifzc e Pgfzr: congelamento =====
        if processos.congelamento_nuvem:
            dqc_ifzc, dNc_ifzc = Pifzc(qc_new, Nc_new, T_new, dt)
            dqc_ifzc = max(dqc_ifzc * dt, -qc_new)
            dNc_ifzc = max(dNc_ifzc * dt, -Nc_new)
            qc_new += dqc_ifzc
            Nc_new += dNc_ifzc
            qi_new += -dqc_ifzc
            Ni_new += -dNc_ifzc
            T_new += (Ls - Lv) / cp * (-dqc_ifzc)

        if processos.congelamento_chuva:
            dqr_gfzr, dNr_gfzr = Pgfzr(qr_new, Nr_new, T_new, dt)
            dqr_gfzr = max(dqr_gfzr * dt, -qr_new)
            dNr_gfzr = max(dNr_gfzr * dt, -Nr_new)
            qr_new += dqr_gfzr
            Nr_new += dNr_gfzr
            qg_new += -dqr_gfzr
            Ng_new += -dNr_gfzr
            T_new += (Ls - Lv) / cp * (-dqr_gfzr)

        # Velocidades e diametros usados nos blocos seguintes.
        Vi, _ = velocidade_terminal_gelo(qi_new, Ni_new, rhok)
        Vs, _ = velocidade_terminal_neve(qs_new, Ns_new, rhok)
        Vg, _ = velocidade_terminal_graupel(qg_new, Ng_new, rhok)

        D_s = (
            diametro_medio_numero(
                lambda_gama(qs_new, Ns_new, rhok, rho_s, MU_SNOW),
                MU_SNOW,
            )
            if qs_new > QMIN and Ns_new > NMIN
            else 0.0
        )
        D_g = (
            diametro_medio_numero(
                lambda_gama(qg_new, Ng_new, rhok, rho_g, MU_SNOW),
                MU_SNOW,
            )
            if qg_new > QMIN and Ng_new > NMIN
            else 0.0
        )

        # ===== 5) Riming: gelo/neve/graupel coletam agua de nuvem =====
        dqc_iiacw = 0.0
        dqc_ssacw = 0.0
        dqc_gacw = 0.0
        if processos.riming:
            dqc_iiacw = max(
                Pi_iacw(qc_new, Ni_new, qi_new, T_new, rhok) * dt,
                -qc_new,
            )
            qc_new += dqc_iiacw
            qi_new += -dqc_iiacw
            T_new += (Ls - Lv) / cp * (-dqc_iiacw)

            if Ns_new > NMIN:
                dqc_ssacw = max(
                    colecao_continua(qc_new, Ns_new, D_s, Vs, 0.0, 1.0) * dt,
                    -qc_new,
                )
                qc_new += dqc_ssacw
                qs_new += -dqc_ssacw
                T_new += (Ls - Lv) / cp * (-dqc_ssacw)

            if Ng_new > NMIN:
                dqc_gacw = max(
                    colecao_continua(qc_new, Ng_new, D_g, Vg, 0.0, 1.0) * dt,
                    -qc_new,
                )
                qc_new += dqc_gacw
                qg_new += -dqc_gacw
                T_new += (Ls - Lv) / cp * (-dqc_gacw)

        # ===== 6) Pispl: Hallett-Mossop =====
        if processos.hallett_mossop:
            riming_total = (-dqc_iiacw - dqc_ssacw - dqc_gacw) / dt
            dNi_ispl = Pispl(riming_total, T_new) * dt
            Ni_new += dNi_ispl

        # ===== 7) Neve/graupel/gelo coletam chuva =====
        if processos.coleta_chuva_por_gelo:
            Vr, _ = velocidade_terminal_chuva(qr_new, Nr_new, rhok)

            if Ns_new > NMIN:
                dqr_ssacr = max(
                    colecao_continua(qr_new, Ns_new, D_s, Vs, Vr, 1.0) * dt,
                    -qr_new,
                )
                qr_new += dqr_ssacr
                qs_new += -dqr_ssacr

            if Ng_new > NMIN:
                dqr_gacr = max(
                    colecao_continua(qr_new, Ng_new, D_g, Vg, Vr, 1.0) * dt,
                    -qr_new,
                )
                qr_new += dqr_gacr
                qg_new += -dqr_gacr

            D_i = (
                diametro_medio_numero(
                    lambda_gama(qi_new, Ni_new, rhok, rho_i, MU_ICE),
                    MU_ICE,
                )
                if qi_new > QMIN and Ni_new > NMIN
                else 0.0
            )
            if Ni_new > NMIN:
                dqr_iacr = max(
                    colecao_continua(qr_new, Ni_new, D_i, Vi, Vr, 1.0) * dt,
                    -qr_new,
                )
                qr_new += dqr_iacr
                qg_new += -dqr_iacr
                T_new += (Ls - Lv) / cp * (-dqr_iacr)

        # ===== 8) Picns: autoconversao gelo -> neve =====
        if processos.gelo_para_neve:
            dqi_icns, dNi_icns, dNs_icns = Picns(qi_new, Ni_new, T_new)
            dqi_icns = max(dqi_icns * dt, -qi_new)
            dNi_icns = max(dNi_icns * dt, -Ni_new)
            dNs_icns *= dt
            qi_new += dqi_icns
            Ni_new += dNi_icns
            qs_new += -dqi_icns
            Ns_new += dNs_icns

        # ===== 9) Degelo =====
        if processos.degelo:
            dqi_imlt, dNi_imlt = Pimlt(qi_new, Ni_new, T_new, dt)
            dqi_imlt = max(dqi_imlt * dt, -qi_new)
            dNi_imlt = max(dNi_imlt * dt, -Ni_new)
            qi_new += dqi_imlt
            Ni_new += dNi_imlt
            qc_new += -dqi_imlt
            Nc_new += -dNi_imlt
            T_new -= (Ls - Lv) / cp * (-dqi_imlt)

            dqs_smlt, dNs_smlt = Pmlt(qs_new, Ns_new, T_new, dt)
            dqs_smlt = max(dqs_smlt * dt, -qs_new)
            dNs_smlt = max(dNs_smlt * dt, -Ns_new)
            qs_new += dqs_smlt
            Ns_new += dNs_smlt
            qr_new += -dqs_smlt
            Nr_new += -dNs_smlt
            T_new -= (Ls - Lv) / cp * (-dqs_smlt)

            dqg_gmlt, dNg_gmlt = Pmlt(qg_new, Ng_new, T_new, dt)
            dqg_gmlt = max(dqg_gmlt * dt, -qg_new)
            dNg_gmlt = max(dNg_gmlt * dt, -Ng_new)
            qg_new += dqg_gmlt
            Ng_new += dNg_gmlt
            qr_new += -dqg_gmlt
            Nr_new += -dNg_gmlt
            T_new -= (Ls - Lv) / cp * (-dqg_gmlt)

        # ===== 10) Processos de chuva quente =====
        if processos.chuva_quente:
            dqc_ccnr, dNc_ccnr = Pccnr(qc_new, Nc_new, rhok)
            dqc_ccnr *= dt
            dNc_ccnr *= dt
            dqc_ccnr = max(dqc_ccnr, -qc_new)

            dqc_racw = max(
                Pracw(qc_new, qr_new) * dt,
                -(qc_new + dqc_ccnr),
            )

            transferencia = -(dqc_ccnr + dqc_racw)
            qc_new = qc_new + dqc_ccnr + dqc_racw
            qr_new = qr_new + transferencia
            Nr_new = Nr_new + (-dNc_ccnr)
            Nc_new = Nc_new + dNc_ccnr

            dNr_self = Pr_self(qr_new, Nr_new, rhok) * dt
            Nr_new = max(Nr_new + dNr_self, 0.0)

            if evap_chuva:
                dqr_revp_b, dNr_revp_b = Prevp(
                    qr_new, Nr_new, qv_new, T_new, pk, rhok
                )
                dqr_revp = max(dqr_revp_b * dt, -qr_new)
                dNr_revp = max(dNr_revp_b * dt, -Nr_new)
                qr_new += dqr_revp
                Nr_new += dNr_revp
                qv_new -= dqr_revp
                T_new += (Lv / cp) * dqr_revp

        # Limpeza dos residuos muito pequenos, igual ao codigo original.
        if qc_new < QMIN:
            qc_new, Nc_new = 0.0, 0.0
        if qr_new < QMIN:
            qr_new, Nr_new = 0.0, 0.0
        if qi_new < QMIN:
            qi_new, Ni_new = 0.0, 0.0
        if qs_new < QMIN:
            qs_new, Ns_new = 0.0, 0.0
        if qg_new < QMIN:
            qg_new, Ng_new = 0.0, 0.0

        T_out[k] = T_new
        qv_out[k] = qv_new
        qc_out[k], Nc_out[k] = qc_new, Nc_new
        qr_out[k], Nr_out[k] = qr_new, Nr_new
        qi_out[k], Ni_out[k] = qi_new, Ni_new
        qs_out[k], Ns_out[k] = qs_new, Ns_new
        qg_out[k], Ng_out[k] = qg_new, Ng_new

    return (
        T_out,
        qv_out,
        qc_out,
        Nc_out,
        qr_out,
        Nr_out,
        qi_out,
        Ni_out,
        qs_out,
        Ns_out,
        qg_out,
        Ng_out,
    )
