# -*- coding: utf-8 -*-
"""
diagnostico_origem_deriva_transporte.py
=======================================

Diagnostico para localizar a origem da deriva de agua no nucleo 2D atual.

Compara tres situacoes:

1) bolha umida original, centrada em z0 = 500 m;
2) a mesma bolha elevada para z0 = 3000 m, longe das fronteiras;
3) um tracador gaussiano artificial, longe das fronteiras, transportado
   por um campo de velocidade incompressivel prescrito.

No terceiro caso sao comparados:
- o operador upwind sozinho;
- o mesmo operador seguido de aplicar_bordas(..., zero_grad=True).

O objetivo NAO e impor ainda uma tolerancia rigorosa. Primeiro queremos
medir se a deriva vem principalmente:
(a) das condicoes de contorno, ou
(b) do operador advectivo em forma nao conservativa.

Rodar a partir da raiz do repositorio:

    python -m pytest tests/diagnostico_origem_deriva_transporte.py -s -v

ou diretamente:

    python tests/diagnostico_origem_deriva_transporte.py
"""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dinamica_2d.nucleo import (
    ConfiguracaoDinamica2D,
    aplicar_bordas,
    criar_estado,
    diagnosticar_cfl,
    passo_dinamico,
    upwind_advect,
    velocidades,
)


def inventario_qv_total(estado, config):
    """
    Integral horizontal-vertical do vapor total:
        qv_total = qv_env(z) + qvp(x,z)

    Unidades: (kg/kg) m2 por unidade de profundidade transversal.
    """
    qv_total = estado.qv_env_1d[None, :] + estado.qvp
    return float(np.sum(qv_total) * config.dx * config.dz)


def inventario_anomalia_umida_inicial(estado, config):
    """
    Integral da anomalia positiva de vapor inserida pela bolha.

    Serve como denominador mais sensivel do que o enorme inventario
    de vapor ambiental do dominio.
    """
    return float(np.sum(np.maximum(estado.qvp, 0.0)) * config.dx * config.dz)


def rodar_bolha_sem_microfisica(z0_m, tempo_total_s=120.0):
    """
    Roda apenas dinamica, sem microfisica e sem difusao.

    Como nao ha hidrometeoros, nao existe precipitacao saindo pela base.
    Qualquer mudanca do inventario de vapor total e numerica.
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
        bolha_z0_m=float(z0_m),
        bolha_qv_kgkg=0.5e-3,
        microfisica="nenhuma",
        difusao=0.0,
        ciclo_diurno=False,
        radiacao=False,
        abortar_se_cfl_violar=True,
    )

    estado = criar_estado(config)

    inventario0 = inventario_qv_total(estado, config)
    anomalia0 = inventario_anomalia_umida_inicial(estado, config)

    maior_desvio = 0.0
    cfl_max = 0.0

    nsteps = int(round(tempo_total_s / config.dt))

    for _ in range(nsteps):
        u, w = velocidades(estado.psi, config.dx, config.dz)

        cfl = diagnosticar_cfl(estado, config, u, w)
        cfl_max = max(cfl_max, cfl["adveccao"])
        if not cfl["estavel_adveccao"]:
            raise RuntimeError(
                f"CFL advectivo violado: {cfl['adveccao']:.6f}"
            )

        passo_dinamico(
            estado,
            config,
            u,
            w,
            estado.dtheta_env_dz,
            estado.dqv_env_dz,
        )

        inventario = inventario_qv_total(estado, config)
        maior_desvio = max(maior_desvio, abs(inventario - inventario0))

    inventariof = inventario_qv_total(estado, config)
    delta = inventariof - inventario0

    erro_total = abs(delta) / max(abs(inventario0), 1.0e-30)
    erro_anomalia = abs(delta) / max(abs(anomalia0), 1.0e-30)
    excursao_anomalia = maior_desvio / max(abs(anomalia0), 1.0e-30)

    return {
        "z0_m": float(z0_m),
        "inventario0": inventario0,
        "inventariof": inventariof,
        "anomalia0": anomalia0,
        "delta": delta,
        "erro_total": erro_total,
        "erro_anomalia": erro_anomalia,
        "excursao_anomalia": excursao_anomalia,
        "cfl_max": cfl_max,
    }


def campo_incompressivel_prescrito(nx=90, nz=110, dx=100.0, dz=100.0):
    """
    Constroi psi = A sin(pi x/Lx) sin(pi z/Lz).

    Como u=-dpsi/dz e w=dpsi/dx sao diagnosticados pela mesma funcao
    usada pelo nucleo, o campo e discretamente associado a uma funcao
    de corrente e nao possui fluxo normal imposto pelas bordas.
    """
    x = np.arange(nx, dtype=float) * dx
    z = np.arange(nz, dtype=float) * dz
    X, Z = np.meshgrid(x, z, indexing="ij")

    lx = x[-1] - x[0]
    lz = z[-1] - z[0]

    amplitude_psi = 1.5e4  # m2/s
    psi = (
        amplitude_psi
        * np.sin(np.pi * X / lx)
        * np.sin(np.pi * Z / lz)
    )

    u, w = velocidades(psi, dx, dz)
    return x, z, X, Z, u, w


def tracador_inicial(X, Z):
    """
    Tracador positivo e bem afastado das fronteiras.
    """
    x0 = 0.5 * (X.min() + X.max())
    z0 = 3000.0
    rx = 600.0
    rz = 600.0

    r2 = ((X - x0) / rx) ** 2 + ((Z - z0) / rz) ** 2
    return 1.0e-3 * np.exp(-r2)


def integrar_tracador(aplicar_condicao_contorno, tempo_total_s=120.0):
    """
    Integra o mesmo tracador com o operador upwind atual.

    Se aplicar_condicao_contorno=False:
        mede principalmente a deriva do operador.

    Se aplicar_condicao_contorno=True:
        mede operador + copiar gradiente zero nas bordas.
    """
    nx = 90
    nz = 110
    dx = 100.0
    dz = 100.0
    dt = 1.5

    _x, _z, X, Z, u, w = campo_incompressivel_prescrito(
        nx=nx, nz=nz, dx=dx, dz=dz
    )
    f = tracador_inicial(X, Z)

    inventario0 = float(np.sum(f) * dx * dz)
    nsteps = int(round(tempo_total_s / dt))

    cfl = float(
        np.max(np.abs(u) * dt / dx + np.abs(w) * dt / dz)
    )
    if cfl > 1.0:
        raise RuntimeError(f"CFL do tracador prescrito = {cfl:.6f} > 1")

    maior_desvio = 0.0

    for _ in range(nsteps):
        f = f + dt * upwind_advect(f, u, w, dx, dz)

        if aplicar_condicao_contorno:
            f = aplicar_bordas(f, zero_grad=True)

        inventario = float(np.sum(f) * dx * dz)
        maior_desvio = max(maior_desvio, abs(inventario - inventario0))

    inventariof = float(np.sum(f) * dx * dz)
    delta = inventariof - inventario0

    return {
        "inventario0": inventario0,
        "inventariof": inventariof,
        "delta": delta,
        "erro_relativo": abs(delta) / max(abs(inventario0), 1.0e-30),
        "excursao_relativa": maior_desvio / max(abs(inventario0), 1.0e-30),
        "cfl": cfl,
        "min_final": float(np.min(f)),
        "max_final": float(np.max(f)),
    }


def imprimir_bolha(rotulo, r):
    print(f"\n=== {rotulo} ===")
    print(f"z0 da bolha                   : {r['z0_m']:.0f} m")
    print(f"CFL maximo                    : {r['cfl_max']:.6f}")
    print(f"inventario qv inicial         : {r['inventario0']:.12e}")
    print(f"inventario qv final           : {r['inventariof']:.12e}")
    print(f"Delta inventario              : {r['delta']:.12e}")
    print(f"erro / inventario total       : {r['erro_total']:.6e}")
    print(f"anomalia umida inicial        : {r['anomalia0']:.12e}")
    print(f"|Delta| / anomalia inicial    : {r['erro_anomalia']:.6e}")
    print(f"excursao max / anomalia       : {r['excursao_anomalia']:.6e}")


def imprimir_tracador(rotulo, r):
    print(f"\n=== {rotulo} ===")
    print(f"CFL                           : {r['cfl']:.6f}")
    print(f"inventario inicial            : {r['inventario0']:.12e}")
    print(f"inventario final              : {r['inventariof']:.12e}")
    print(f"Delta inventario              : {r['delta']:.12e}")
    print(f"erro relativo                 : {r['erro_relativo']:.6e}")
    print(f"excursao relativa max         : {r['excursao_relativa']:.6e}")
    print(f"minimo final                  : {r['min_final']:.6e}")
    print(f"maximo final                  : {r['max_final']:.6e}")


def executar_diagnostico(tempo_total_s=120.0):
    baixo = rodar_bolha_sem_microfisica(
        z0_m=500.0,
        tempo_total_s=tempo_total_s,
    )
    alto = rodar_bolha_sem_microfisica(
        z0_m=3000.0,
        tempo_total_s=tempo_total_s,
    )
    operador = integrar_tracador(
        aplicar_condicao_contorno=False,
        tempo_total_s=tempo_total_s,
    )
    operador_borda = integrar_tracador(
        aplicar_condicao_contorno=True,
        tempo_total_s=tempo_total_s,
    )

    imprimir_bolha("BOLHA ORIGINAL, PROXIMA AO SOLO", baixo)
    imprimir_bolha("BOLHA ELEVADA, LONGE DAS BORDAS", alto)
    imprimir_tracador("TRACADOR: OPERADOR UPWIND APENAS", operador)
    imprimir_tracador(
        "TRACADOR: UPWIND + BORDA ZERO-GRAD",
        operador_borda,
    )

    print("\n=== COMPARACOES ===")

    if baixo["erro_anomalia"] > 0.0:
        razao_bolhas = alto["erro_anomalia"] / baixo["erro_anomalia"]
    else:
        razao_bolhas = np.nan

    if operador["erro_relativo"] > 0.0:
        razao_borda = (
            operador_borda["erro_relativo"]
            / operador["erro_relativo"]
        )
    else:
        razao_borda = np.nan

    print(
        "erro bolha alta / erro bolha baixa : "
        f"{razao_bolhas:.6e}"
    )
    print(
        "erro (upwind+borda) / erro upwind   : "
        f"{razao_borda:.6e}"
    )

    print("\nInterpretacao sugerida:")
    print(
        "- Se elevar a bolha reduzir fortemente o erro, "
        "a fronteira inferior e importante."
    )
    print(
        "- Se o tracador perder massa mesmo sem aplicar_bordas, "
        "o operador upwind tambem contribui."
    )
    print(
        "- Se o erro aumentar muito ao aplicar zero_grad, "
        "o tratamento de borda e uma fonte importante da deriva."
    )

    return baixo, alto, operador, operador_borda


def test_diagnostico_origem_deriva():
    """
    Teste diagnostico: por enquanto exige apenas estabilidade e finitude.

    Nao deve ser transformado em teste rigoroso de regressao antes de
    decidirmos qual formulacao conservativa sera adotada.
    """
    baixo, alto, operador, operador_borda = executar_diagnostico(
        tempo_total_s=120.0
    )

    resultados = (baixo, alto, operador, operador_borda)

    for r in resultados:
        for valor in r.values():
            if isinstance(valor, (float, np.floating)):
                assert np.isfinite(valor)

    assert baixo["cfl_max"] <= 1.0
    assert alto["cfl_max"] <= 1.0
    assert operador["cfl"] <= 1.0
    assert operador_borda["cfl"] <= 1.0


if __name__ == "__main__":
    executar_diagnostico(tempo_total_s=120.0)
