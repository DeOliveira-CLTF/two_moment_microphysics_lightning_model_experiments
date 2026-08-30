# -*- coding: utf-8 -*-
"""
diagnostico_conservacao_dinamica_2d.py
======================================

Diagnostico nao destrutivo da conservacao de agua no nucleo 2D atual.

Objetivos:
1) verificar se a microfisica local, isoladamente, conserva qt =
   qv + qc + qr + qi + qs + qg;
2) medir a deriva do inventario total de agua causada apenas pela
   dinamica/adveccao, sem microfisica e sem difusao;
3) mostrar separadamente o inventario Boussinesq (integral de qt) e
   um diagnostico ponderado por rho0.

Este arquivo NAO altera o nucleo. Ele serve para medir o problema antes
de decidir se o transporte precisa ser reescrito em forma conservativa.

Rodar, a partir da raiz do repositorio:

    python -m pytest tests/diagnostico_conservacao_dinamica_2d.py -s -v

ou diretamente:

    python tests/diagnostico_conservacao_dinamica_2d.py
"""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinamica_2d.nucleo import (
    ConfiguracaoDinamica2D,
    criar_estado,
    diagnosticar_cfl,
    passo_dinamico,
    passo_microfisica_2d,
    velocidades,
)


CATEGORIAS_AGUA = ("qc", "qr", "qi", "qs", "qg")


def qt_total(estado):
    """Agua total em kg/kg, incluindo o vapor ambiental + perturbacao."""
    qv = estado.qv_env_1d[None, :] + estado.qvp
    qt = qv.copy()
    for nome in CATEGORIAS_AGUA:
        qt = qt + getattr(estado, nome)
    return qt


def inventario_boussinesq(estado, config):
    """
    Integral de qt no dominio.

    Para o nucleo Boussinesq, esta e a quantidade escalar diretamente
    associada ao transporte por um campo de velocidade incompressivel.
    Unidades: (kg/kg) m2 por unidade de profundidade transversal.
    """
    return float(np.sum(qt_total(estado)) * config.dx * config.dz)


def inventario_ponderado_rho0(estado, config):
    """
    Diagnostico fisico ponderado por rho0(z).

    Unidades: kg de agua por metro de profundidade transversal.
    Nao e usado como criterio primario de conservacao do transporte
    Boussinesq, pois o nucleo resolve div(v)=0, e nao div(rho0 v)=0.
    """
    rho2d = estado.rho0_1d[None, :]
    return float(np.sum(rho2d * qt_total(estado)) * config.dx * config.dz)


def erro_relativo(final, inicial):
    return abs(final - inicial) / max(abs(inicial), 1.0e-30)


def diagnostico_microfisica_local():
    """
    Um unico subpasso microfisico, sem transporte.

    Como os processos microfisicos apenas convertem agua entre categorias,
    qt deve permanecer praticamente inalterado em cada ponto.
    """
    config = ConfiguracaoDinamica2D(
        nx=30,
        nz=110,
        dt=1.5,
        bolha_k=7.0,
        microfisica="thompson",
        difusao=0.0,
        ciclo_diurno=False,
        radiacao=False,
    )
    estado = criar_estado(config)

    qt_antes = qt_total(estado)

    theta_base = estado.theta_env_1d[None, :]
    T = (theta_base + estado.thp) * estado.pi_1d[None, :]
    passo_microfisica_2d(
        estado,
        config,
        T,
        estado.qv_env_1d,
    )

    qt_depois = qt_total(estado)
    delta = qt_depois - qt_antes

    max_abs = float(np.max(np.abs(delta)))
    rms = float(np.sqrt(np.mean(delta**2)))
    delta_integrado = float(np.sum(delta) * config.dx * config.dz)

    print("\n=== MICROFISICA LOCAL ===")
    print(f"max |Delta qt| por ponto : {max_abs:.6e} kg/kg")
    print(f"RMS(Delta qt)            : {rms:.6e} kg/kg")
    print(f"Delta integral qt        : {delta_integrado:.6e} (kg/kg) m2")

    return {
        "max_abs": max_abs,
        "rms": rms,
        "delta_integrado": delta_integrado,
    }


def diagnostico_transporte_fechado(tempo_total_s=300.0):
    """
    Dinamica sem microfisica e sem difusao.

    Ha apenas a bolha termica/umida inicial e o transporte dinamico.
    Como nao existem hidrometeoros sedimentantes, nao existe precipitacao
    que deva sair pela base. Assim, qualquer deriva do inventario de qt
    revela erro numerico de transporte e/ou tratamento de fronteira.
    """
    config = ConfiguracaoDinamica2D(
        nx=90,
        nz=110,
        dx=100.0,
        dz=100.0,
        dt=1.5,
        tempo_total_s=tempo_total_s,
        salvar_a_cada_s=300.0,
        bolha_k=7.0,
        bolha_qv_kgkg=0.5e-3,
        microfisica="nenhuma",
        difusao=0.0,
        ciclo_diurno=False,
        radiacao=False,
        abortar_se_cfl_violar=True,
    )
    estado = criar_estado(config)

    inv0 = inventario_boussinesq(estado, config)
    massa0 = inventario_ponderado_rho0(estado, config)

    inv_min = inv0
    inv_max = inv0
    cfl_max = 0.0

    nsteps = int(round(tempo_total_s / config.dt))

    for step in range(nsteps):
        u, w = velocidades(estado.psi, config.dx, config.dz)
        cfl = diagnosticar_cfl(estado, config, u, w)
        cfl_max = max(cfl_max, cfl["adveccao"])

        if not cfl["estavel_adveccao"]:
            raise RuntimeError(
                f"CFL violado no passo {step}: {cfl['adveccao']:.3f}"
            )

        passo_dinamico(
            estado,
            config,
            u,
            w,
            estado.dtheta_env_dz,
            estado.dqv_env_dz,
        )

        inv = inventario_boussinesq(estado, config)
        inv_min = min(inv_min, inv)
        inv_max = max(inv_max, inv)

    invf = inventario_boussinesq(estado, config)
    massaf = inventario_ponderado_rho0(estado, config)

    err_b = erro_relativo(invf, inv0)
    err_rho = erro_relativo(massaf, massa0)
    excursao = (inv_max - inv_min) / max(abs(inv0), 1.0e-30)

    print("\n=== TRANSPORTE 2D FECHADO ===")
    print(f"tempo simulado              : {tempo_total_s:.1f} s")
    print(f"CFL maximo                  : {cfl_max:.6f}")
    print(f"inventario Boussinesq inicial: {inv0:.12e}")
    print(f"inventario Boussinesq final  : {invf:.12e}")
    print(f"erro relativo Boussinesq     : {err_b:.6e}")
    print(f"excursao min-max relativa    : {excursao:.6e}")
    print(f"massa rho0 inicial           : {massa0:.12e} kg/m")
    print(f"massa rho0 final             : {massaf:.12e} kg/m")
    print(f"erro relativo ponderado rho0 : {err_rho:.6e}")

    return {
        "erro_boussinesq": err_b,
        "erro_rho0": err_rho,
        "excursao": excursao,
        "cfl_max": cfl_max,
    }


def test_diagnostico_microfisica_local():
    """
    Teste deliberadamente permissivo nesta primeira etapa.

    O objetivo imediato e medir a ordem do erro. Se o resultado estiver
    proximo de precisao de maquina, a tolerancia pode ser apertada depois.
    """
    r = diagnostico_microfisica_local()
    assert np.isfinite(r["max_abs"])
    assert r["max_abs"] < 1.0e-8


def test_diagnostico_transporte_fechado():
    """
    Nao impoe ainda tolerancia forte de conservacao.

    Primeiro queremos medir a deriva do esquema atual sem mascarar o
    diagnostico. Depois de observar o valor, o transporte pode ser corrigido
    e este teste convertido em um criterio rigoroso de regressao.
    """
    r = diagnostico_transporte_fechado(tempo_total_s=120.0)
    assert np.isfinite(r["erro_boussinesq"])
    assert r["cfl_max"] <= 1.0


if __name__ == "__main__":
    diagnostico_microfisica_local()
    diagnostico_transporte_fechado(tempo_total_s=300.0)
