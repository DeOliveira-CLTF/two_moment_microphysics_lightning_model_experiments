# -*- coding: utf-8 -*-
"""Nucleo dinamico 2D para nuvem convectiva idealizada.

Este modulo preserva o nucleo Boussinesq vorticidade-funcao de corrente do
modelo fornecido pelo professor e acrescenta os controles necessarios para os
experimentos cientificos:

- aquecimento uniforme do ambiente em temperatura real (Grupo 2);
- preservacao opcional de umidade relativa no ambiente aquecido;
- amplitude e geometria configuraveis da bolha termica;
- concentracao de goticulas configuravel (Grupo 1);
- chaves de processos microfisicos (Grupo 3);
- diagnostico de CFL a cada passo;
- armazenamento das categorias microfisicas separadamente.
"""

from dataclasses import dataclass, field
import time

import numpy as np

from microfisica.coluna_generica import passo_microfisica_coluna
from microfisica.configuracao import OpcoesMicrofisica
from microfisica.constantes import (
    Rd,
    cp,
    Lv,
    rho_w,
    rho_i,
    rho_s,
    rho_g,
    MU_RAIN,
    MU_ICE,
    MU_SNOW,
    QMIN,
    NMIN,
    gamma_func,
)

G = 9.81
RV = 461.5
EPS_R = Rd / RV
P0_HPA = 1000.0
THETA0 = 300.0
RAD_COOL_RATE = 1.5 / 86400.0


@dataclass(frozen=True)
class ConfiguracaoDinamica2D:
    """Parametros do nucleo 2D idealizado."""

    nx: int = 90
    nz: int = 110
    dx: float = 100.0
    dz: float = 100.0
    dt: float = 1.5
    tempo_total_s: float = 60.0 * 60.0
    salvar_a_cada_s: float = 300.0

    # Disparo convectivo.
    bolha_k: float = 3.0
    bolha_z0_m: float = 500.0
    bolha_rx_m: float = 1100.0
    bolha_rz_m: float = 600.0
    bolha_qv_kgkg: float = 0.5e-3

    # Experimento de aquecimento.
    delta_t_ambiente_k: float = 0.0
    preservar_rh: bool = True

    # Microfisica.
    microfisica: str = "thompson"  # "nenhuma" ou "thompson"
    evap_chuva: bool = True
    nc_ativacao_kg1: float = 1.0e8
    processos: OpcoesMicrofisica = field(default_factory=OpcoesMicrofisica)

    # Outras fisicas/numerica.
    radiacao: bool = False
    ciclo_diurno: bool = False
    difusao: float = 25.0
    iteracoes_poisson: int = 120
    cenario: str = "bolha"

    # Seguranca numerica.
    cfl_aviso: float = 0.80
    cfl_limite: float = 1.00
    abortar_se_cfl_violar: bool = True


@dataclass
class EstadoDinamica2D:
    """Campos prognosticos e ambientais do modelo 2D."""

    x: np.ndarray
    z: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    theta_env_1d: np.ndarray
    T_env_1d: np.ndarray
    qv_env_1d: np.ndarray
    rh_env_1d: np.ndarray
    pi_1d: np.ndarray
    p_hpa_1d: np.ndarray
    p_pa_1d: np.ndarray
    rho0_1d: np.ndarray
    dtheta_env_dz: np.ndarray
    dqv_env_dz: np.ndarray
    zeta: np.ndarray
    psi: np.ndarray
    thp: np.ndarray
    qvp: np.ndarray
    qc: np.ndarray
    Nc: np.ndarray
    qr: np.ndarray
    Nr: np.ndarray
    qi: np.ndarray
    Ni: np.ndarray
    qs: np.ndarray
    Ns: np.ndarray
    qg: np.ndarray
    Ng: np.ndarray
    h_clc: float = 200.0
    theta_ml: float | None = None
    q_ml: float = 11.0e-3
    ultimo_disparo_s: float = -1.0e9


def validar_configuracao(config: ConfiguracaoDinamica2D) -> None:
    if config.nx < 5 or config.nz < 5:
        raise ValueError("nx e nz devem ser pelo menos 5")
    for nome in ("dx", "dz", "dt", "tempo_total_s", "salvar_a_cada_s"):
        if getattr(config, nome) <= 0.0:
            raise ValueError(f"{nome} deve ser positivo")
    if config.microfisica not in {"nenhuma", "thompson"}:
        raise ValueError("microfisica deve ser 'nenhuma' ou 'thompson'")
    if config.nc_ativacao_kg1 <= 0.0:
        raise ValueError("nc_ativacao_kg1 deve ser positivo")
    if config.cfl_aviso <= 0.0 or config.cfl_limite <= 0.0:
        raise ValueError("limites de CFL devem ser positivos")
    if config.cfl_aviso > config.cfl_limite:
        raise ValueError("cfl_aviso nao pode exceder cfl_limite")


def p_of_z(z_m, escala_m=8000.0):
    return P0_HPA * np.exp(-z_m / escala_m)


def exner(z_m):
    return (p_of_z(z_m) / P0_HPA) ** (Rd / cp)


def qsat_liq(T_k, p_hpa):
    Tc = T_k - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    return EPS_R * es / np.maximum(p_hpa - es, 1.0e-3)


def dtheta_dz_env(z_m):
    return np.where(
        z_m < 1000.0,
        3.0e-3,
        np.where(z_m < 2000.0, 6.5e-3, np.where(z_m < 8500.0, 2.0e-3, 6.0e-3)),
    )


def RH_env_profile(z_m):
    return np.where(
        z_m < 1000.0,
        0.70,
        np.where(z_m < 2000.0, 0.35, np.where(z_m < 8500.0, 0.55, 0.20)),
    )


def fluxo_sensivel(t_s, maximo=250.0, duracao_dia_s=12.0 * 3600.0):
    if 0.0 <= t_s <= duracao_dia_s:
        return max(0.0, maximo * np.sin(np.pi * t_s / duracao_dia_s))
    return 0.0


def fluxo_latente(t_s, maximo=300.0, duracao_dia_s=12.0 * 3600.0):
    if 0.0 <= t_s <= duracao_dia_s:
        return max(0.0, maximo * np.sin(np.pi * t_s / duracao_dia_s))
    return 0.0


def construir_ambiente_clc(
    z,
    theta_env_1d,
    qv_env_1d,
    h_clc,
    theta_ml,
    q_ml,
    dz,
):
    theta_now = np.where(z < h_clc, theta_ml, theta_env_1d)
    qv_now = np.where(z < h_clc, q_ml, qv_env_1d)
    return (
        theta_now,
        qv_now,
        np.gradient(theta_now, dz),
        np.gradient(qv_now, dz),
    )


def criar_estado(config: ConfiguracaoDinamica2D) -> EstadoDinamica2D:
    """Cria ambiente, bolha termica inicial e campos prognosticos zerados."""

    validar_configuracao(config)

    x = np.arange(config.nx, dtype=float) * config.dx
    z = np.arange(config.nz, dtype=float) * config.dz
    X, Z = np.meshgrid(x, z, indexing="ij")

    # Perfil de referencia original do professor em temperatura potencial.
    theta_original = np.zeros(config.nz, dtype=float)
    theta_original[0] = THETA0
    grad = dtheta_dz_env(z)
    for k in range(1, config.nz):
        theta_original[k] = theta_original[k - 1] + grad[k - 1] * config.dz

    pi_1d = exner(z)
    p_hpa_1d = p_of_z(z)
    p_pa_1d = p_hpa_1d * 100.0

    # O WARM e definido em temperatura real: T_warm(z) = T_ctrl(z) + DeltaT.
    T_original_1d = theta_original * pi_1d
    T_env_1d = T_original_1d + config.delta_t_ambiente_k
    theta_env_1d = T_env_1d / pi_1d

    rh_ctrl_1d = RH_env_profile(z)
    if config.preservar_rh:
        qv_env_1d = rh_ctrl_1d * qsat_liq(T_env_1d, p_hpa_1d)
    else:
        # Mantem qv do CTRL mesmo quando T e alterada.
        qv_env_1d = rh_ctrl_1d * qsat_liq(T_original_1d, p_hpa_1d)

    # RH efetivamente resultante, util para auditoria dos experimentos WARM.
    rh_env_1d = qv_env_1d / np.maximum(qsat_liq(T_env_1d, p_hpa_1d), 1.0e-12)
    rho0_1d = p_pa_1d / (Rd * T_env_1d)

    shape = (config.nx, config.nz)
    zeta = np.zeros(shape)
    psi = np.zeros(shape)
    thp = np.zeros(shape)
    qvp = np.zeros(shape)

    if not config.ciclo_diurno:
        # Centralizar a termica no dominio torna nx configuravel sem deslocar a
        # bolha para fora da grade.
        x0 = 0.5 * (x[0] + x[-1])
        z0 = config.bolha_z0_m
        rx = config.bolha_rx_m
        rz = config.bolha_rz_m
        r2 = ((X - x0) / rx) ** 2 + ((Z - z0) / rz) ** 2
        mascara = r2 < 6.0
        thp += config.bolha_k * np.exp(-r2) * mascara
        qvp += config.bolha_qv_kgkg * np.exp(-r2) * mascara

    return EstadoDinamica2D(
        x=x,
        z=z,
        X=X,
        Z=Z,
        theta_env_1d=theta_env_1d,
        T_env_1d=T_env_1d,
        qv_env_1d=qv_env_1d,
        rh_env_1d=rh_env_1d,
        pi_1d=pi_1d,
        p_hpa_1d=p_hpa_1d,
        p_pa_1d=p_pa_1d,
        rho0_1d=rho0_1d,
        dtheta_env_dz=np.gradient(theta_env_1d, config.dz),
        dqv_env_dz=np.gradient(qv_env_1d, config.dz),
        zeta=zeta,
        psi=psi,
        thp=thp,
        qvp=qvp,
        qc=np.zeros(shape),
        Nc=np.zeros(shape),
        qr=np.zeros(shape),
        Nr=np.zeros(shape),
        qi=np.zeros(shape),
        Ni=np.zeros(shape),
        qs=np.zeros(shape),
        Ns=np.zeros(shape),
        qg=np.zeros(shape),
        Ng=np.zeros(shape),
    )


def poisson_jacobi(zeta_field, psi_guess, dx, dz, niter=120, omega=1.0):
    psi = psi_guess.copy()
    dx2, dz2 = dx * dx, dz * dz
    denom = 2.0 * (1.0 / dx2 + 1.0 / dz2)
    for _ in range(niter):
        rhs = (
            (psi[2:, 1:-1] + psi[:-2, 1:-1]) / dx2
            + (psi[1:-1, 2:] + psi[1:-1, :-2]) / dz2
            - zeta_field[1:-1, 1:-1]
        )
        novo = rhs / denom
        psi[1:-1, 1:-1] = (1.0 - omega) * psi[1:-1, 1:-1] + omega * novo
    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0
    return psi


def velocidades(psi, dx, dz):
    u = np.zeros_like(psi)
    w = np.zeros_like(psi)
    u[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dz)
    w[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
    return u, w


def upwind_advect(f, u, w, dx, dz):
    dfdx = np.zeros_like(f)
    dfdz = np.zeros_like(f)
    dfdx[1:-1, :] = np.where(
        u[1:-1, :] > 0.0,
        (f[1:-1, :] - f[:-2, :]) / dx,
        (f[2:, :] - f[1:-1, :]) / dx,
    )
    dfdz[:, 1:-1] = np.where(
        w[:, 1:-1] > 0.0,
        (f[:, 1:-1] - f[:, :-2]) / dz,
        (f[:, 2:] - f[:, 1:-1]) / dz,
    )
    return -(u * dfdx + w * dfdz)


def laplacian(f, dx, dz):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2
        + (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dz**2
    )
    return lap


def aplicar_bordas(f, zero_grad=True):
    if zero_grad:
        f[0, :] = f[1, :]
        f[-1, :] = f[-2, :]
        f[:, 0] = f[:, 1]
        f[:, -1] = f[:, -2]
    else:
        f[0, :] = 0.0
        f[-1, :] = 0.0
        f[:, 0] = 0.0
        f[:, -1] = 0.0
    return f


def _campo_Vt(q, N, rho, rho_x, mu, a, b, vmax, correcao_rho=None):
    valido = (q > QMIN) & (N > NMIN)
    q_safe = np.maximum(q, QMIN)
    N_safe = np.maximum(N, NMIN)
    lam = (
        np.pi
        * rho_x
        * gamma_func(mu + 4.0)
        * N_safe
        / (6.0 * rho * gamma_func(mu + 1.0) * q_safe)
    ) ** (1.0 / 3.0)
    Vq = a * (gamma_func(mu + 4.0 + b) / gamma_func(mu + 4.0)) * lam ** (-b)
    Vn = a * (gamma_func(mu + 1.0 + b) / gamma_func(mu + 1.0)) * lam ** (-b)
    if correcao_rho is not None:
        Vq *= correcao_rho
        Vn *= correcao_rho
    return (
        np.where(valido, np.minimum(Vq, vmax), 0.0),
        np.where(valido, np.minimum(Vn, vmax), 0.0),
    )


def campo_Vt_chuva(q, N, rho):
    return _campo_Vt(
        q,
        N,
        rho,
        rho_w,
        MU_RAIN,
        842.0,
        0.8,
        9.5,
        (1.2 / rho) ** 0.5,
    )


def campo_Vt_gelo(q, N, rho):
    return _campo_Vt(q, N, rho, rho_i, MU_ICE, 700.0, 1.0, 1.5)


def campo_Vt_neve(q, N, rho):
    return _campo_Vt(q, N, rho, rho_s, MU_SNOW, 11.72, 0.41, 3.0)


def campo_Vt_graupel(q, N, rho):
    return _campo_Vt(q, N, rho, rho_g, MU_SNOW, 19.3, 0.37, 12.0)


def disparar_termica(
    estado: EstadoDinamica2D,
    config: ConfiguracaoDinamica2D,
    t_s,
    amp_scale,
    intervalo_s=600.0,
    amp_base_k=4.0,
):
    if t_s - estado.ultimo_disparo_s < intervalo_s:
        return
    estado.ultimo_disparo_s = t_s
    x0 = 0.5 * (estado.x[0] + estado.x[-1])
    rx = config.bolha_rx_m
    amp = amp_base_k * max(amp_scale, 0.05)
    z0 = max(estado.h_clc * 0.5, 100.0)
    rz = max(estado.h_clc * 0.4, 150.0)
    r2 = ((estado.X - x0) / rx) ** 2 + ((estado.Z - z0) / rz) ** 2
    mascara = r2 < 6.0
    estado.thp += amp * np.exp(-r2) * mascara
    estado.qvp += config.bolha_qv_kgkg * np.exp(-r2) * mascara


def atualizar_clc(estado: EstadoDinamica2D, config: ConfiguracaoDinamica2D, t_s):
    shf = fluxo_sensivel(t_s)
    lhf = fluxo_latente(t_s)
    if estado.theta_ml is None:
        estado.theta_ml = float(
            np.interp(estado.h_clc, estado.z, estado.theta_env_1d)
        )
    gamma_local = max(
        float(
            np.interp(
                estado.h_clc,
                estado.z,
                np.gradient(estado.theta_env_1d, config.dz),
            )
        ),
        1.0e-4,
    )
    dh = (
        config.dt * shf / (1.15 * cp * estado.h_clc * gamma_local)
        if shf > 0.0
        else 0.0
    )
    estado.h_clc = max(estado.h_clc + dh, 200.0)
    estado.theta_ml = float(
        np.interp(estado.h_clc, estado.z, estado.theta_env_1d)
    )
    q_env_top = float(np.interp(estado.h_clc, estado.z, estado.qv_env_1d))
    mistura = dh / estado.h_clc if estado.h_clc > 0.0 else 0.0
    estado.q_ml = max(
        estado.q_ml
        + config.dt * (lhf / (1.15 * Lv * estado.h_clc))
        + (q_env_top - estado.q_ml) * mistura,
        1.0e-4,
    )
    if shf > 0.0:
        disparar_termica(estado, config, t_s, shf / 250.0)
    return shf, lhf


def passo_microfisica_2d(
    estado: EstadoDinamica2D,
    config: ConfiguracaoDinamica2D,
    T,
    qv_env_now_1d,
):
    if config.microfisica != "thompson":
        return

    qv_total = qv_env_now_1d[None, :] + estado.qvp
    for i in range(config.nx):
        (
            T_novo,
            qv_novo,
            estado.qc[i, :],
            estado.Nc[i, :],
            estado.qr[i, :],
            estado.Nr[i, :],
            estado.qi[i, :],
            estado.Ni[i, :],
            estado.qs[i, :],
            estado.Ns[i, :],
            estado.qg[i, :],
            estado.Ng[i, :],
        ) = passo_microfisica_coluna(
            config.dt,
            T[i, :],
            estado.p_pa_1d,
            estado.rho0_1d,
            qv_total[i, :],
            estado.qc[i, :],
            estado.Nc[i, :],
            estado.qr[i, :],
            estado.Nr[i, :],
            estado.qi[i, :],
            estado.Ni[i, :],
            estado.qs[i, :],
            estado.Ns[i, :],
            estado.qg[i, :],
            estado.Ng[i, :],
            evap_chuva=config.evap_chuva,
            nc_ativacao_kg1=config.nc_ativacao_kg1,
            processos=config.processos,
        )
        estado.thp[i, :] += (T_novo - T[i, :]) / estado.pi_1d
        estado.qvp[i, :] += qv_novo - qv_total[i, :]


def _velocidades_terminais(estado: EstadoDinamica2D):
    rho2d = estado.rho0_1d[None, :]
    Vtq_r, Vtn_r = campo_Vt_chuva(estado.qr, estado.Nr, rho2d)
    Vtq_i, Vtn_i = campo_Vt_gelo(estado.qi, estado.Ni, rho2d)
    Vtq_s, Vtn_s = campo_Vt_neve(estado.qs, estado.Ns, rho2d)
    Vtq_g, Vtn_g = campo_Vt_graupel(estado.qg, estado.Ng, rho2d)
    return Vtq_r, Vtn_r, Vtq_i, Vtn_i, Vtq_s, Vtn_s, Vtq_g, Vtn_g


def diagnosticar_cfl(estado: EstadoDinamica2D, config: ConfiguracaoDinamica2D, u, w):
    """Retorna numeros de Courant para adveccao/sedimentacao e difusao.

    Para o esquema upwind 2D, usa-se a condicao suficiente

        |u| dt/dx + |w_eff| dt/dz <= 1,

    onde ``w_eff = w - Vt`` para hidrometeoros sedimentantes. Para a difusao
    explicita 2D, monitora-se

        K dt (1/dx^2 + 1/dz^2) <= 1/2.
    """

    velocidades_z = [w]
    if config.microfisica == "thompson":
        for vt in _velocidades_terminais(estado):
            velocidades_z.append(w - vt)

    cfl_adv = max(
        float(
            np.max(
                np.abs(u) * config.dt / config.dx
                + np.abs(vz) * config.dt / config.dz
            )
        )
        for vz in velocidades_z
    )
    cfl_diff = float(
        config.difusao
        * config.dt
        * (1.0 / config.dx**2 + 1.0 / config.dz**2)
    )

    return {
        "adveccao": cfl_adv,
        "difusao": cfl_diff,
        "estavel_adveccao": cfl_adv <= config.cfl_limite,
        "estavel_difusao": cfl_diff <= 0.5,
    }


def passo_dinamico(
    estado: EstadoDinamica2D,
    config: ConfiguracaoDinamica2D,
    u,
    w,
    dtheta_dz_now,
    dqv_dz_now,
):
    if config.microfisica == "thompson":
        (
            Vtq_r,
            Vtn_r,
            Vtq_i,
            Vtn_i,
            Vtq_s,
            Vtn_s,
            Vtq_g,
            Vtn_g,
        ) = _velocidades_terminais(estado)
    else:
        zero = np.zeros_like(estado.qc)
        Vtq_r = Vtn_r = Vtq_i = Vtn_i = Vtq_s = Vtn_s = Vtq_g = Vtn_g = zero

    thv = (
        estado.thp
        + 0.61 * THETA0 * estado.qvp
        - THETA0 * (estado.qc + estado.qi + estado.qr + estado.qs + estado.qg)
    )
    dthv_dx = np.zeros_like(estado.thp)
    dthv_dx[1:-1, :] = (thv[2:, :] - thv[:-2, :]) / (2.0 * config.dx)
    buoy_torque = (G / THETA0) * dthv_dx

    dzeta = (
        upwind_advect(estado.zeta, u, w, config.dx, config.dz)
        + buoy_torque
        + config.difusao * laplacian(estado.zeta, config.dx, config.dz)
    )
    dthp = (
        upwind_advect(estado.thp, u, w, config.dx, config.dz)
        - w * dtheta_dz_now[None, :]
        + config.difusao * laplacian(estado.thp, config.dx, config.dz)
    )
    dqvp = (
        upwind_advect(estado.qvp, u, w, config.dx, config.dz)
        - w * dqv_dz_now[None, :]
        + config.difusao * laplacian(estado.qvp, config.dx, config.dz)
    )
    dqc = (
        upwind_advect(estado.qc, u, w, config.dx, config.dz)
        + config.difusao * laplacian(estado.qc, config.dx, config.dz)
    )
    dNc = (
        upwind_advect(estado.Nc, u, w, config.dx, config.dz)
        + config.difusao * laplacian(estado.Nc, config.dx, config.dz)
    )

    estado.zeta += config.dt * dzeta
    estado.thp += config.dt * dthp
    estado.qvp += config.dt * dqvp
    estado.qc = np.maximum(estado.qc + config.dt * dqc, 0.0)
    estado.Nc = np.maximum(estado.Nc + config.dt * dNc, 0.0)

    if config.microfisica == "thompson":
        pares = (
            ("qr", "Nr", Vtq_r, Vtn_r),
            ("qi", "Ni", Vtq_i, Vtn_i),
            ("qs", "Ns", Vtq_s, Vtn_s),
            ("qg", "Ng", Vtq_g, Vtn_g),
        )
        for qnome, nnome, Vtq, Vtn in pares:
            qcampo = getattr(estado, qnome)
            Ncampo = getattr(estado, nnome)
            dq = upwind_advect(
                qcampo, u, w - Vtq, config.dx, config.dz
            ) + config.difusao * laplacian(qcampo, config.dx, config.dz)
            dN = upwind_advect(
                Ncampo, u, w - Vtn, config.dx, config.dz
            ) + config.difusao * laplacian(Ncampo, config.dx, config.dz)
            setattr(estado, qnome, np.maximum(qcampo + config.dt * dq, 0.0))
            setattr(estado, nnome, np.maximum(Ncampo + config.dt * dN, 0.0))

    estado.zeta = aplicar_bordas(estado.zeta, zero_grad=False)
    for nome in (
        "thp",
        "qvp",
        "qc",
        "Nc",
        "qr",
        "Nr",
        "qi",
        "Ni",
        "qs",
        "Ns",
        "qg",
        "Ng",
    ):
        setattr(estado, nome, aplicar_bordas(getattr(estado, nome), zero_grad=True))

    estado.psi = poisson_jacobi(
        estado.zeta,
        estado.psi,
        config.dx,
        config.dz,
        niter=config.iteracoes_poisson,
    )


def _novo_dicionario_frames():
    return {
        "t": [],
        "T": [],
        "qv": [],
        "qc": [],
        "Nc": [],
        "qr": [],
        "Nr": [],
        "qi": [],
        "Ni": [],
        "qs": [],
        "Ns": [],
        "qg": [],
        "Ng": [],
        "qc_qi": [],
        "qr_qs_qg": [],
        "w": [],
        "u": [],
        "thp": [],
        "qvp": [],
        "cfl_adv": [],
        "cfl_diff": [],
    }


def rodar_thompson_2d(config: ConfiguracaoDinamica2D | None = None, verbose=False):
    """Executa o modelo 2D acoplado a microfisica de dois momentos."""

    config = config or ConfiguracaoDinamica2D()
    validar_configuracao(config)
    estado = criar_estado(config)
    nsteps = int(config.tempo_total_s / config.dt)
    save_every = max(1, int(round(config.salvar_a_cada_s / config.dt)))
    inicio = time.time()

    frames = _novo_dicionario_frames()
    maior_cfl_adv = 0.0
    maior_cfl_diff = 0.0

    for step in range(nsteps + 1):
        u, w = velocidades(estado.psi, config.dx, config.dz)
        t_s = step * config.dt

        if config.ciclo_diurno:
            shf, _lhf = atualizar_clc(estado, config, t_s)
            theta_env_now, qv_env_now, dtheta_now, dqv_now = construir_ambiente_clc(
                estado.z,
                estado.theta_env_1d,
                estado.qv_env_1d,
                estado.h_clc,
                estado.theta_ml,
                estado.q_ml,
                config.dz,
            )
        else:
            shf = 0.0
            theta_env_now = estado.theta_env_1d
            qv_env_now = estado.qv_env_1d
            dtheta_now = estado.dtheta_env_dz
            dqv_now = estado.dqv_env_dz

        theta_base = theta_env_now[None, :] - (
            RAD_COOL_RATE * t_s if config.radiacao else 0.0
        )
        T_antes_microfisica = (theta_base + estado.thp) * estado.pi_1d[None, :]
        passo_microfisica_2d(estado, config, T_antes_microfisica, qv_env_now)

        # Recalcula os campos apos o subpasso microfisico para que as saidas
        # tenham T/qv consistentes com qc, qr, qi, qs e qg salvos.
        T_atual = (theta_base + estado.thp) * estado.pi_1d[None, :]
        qv_atual = qv_env_now[None, :] + estado.qvp

        cfl = diagnosticar_cfl(estado, config, u, w)
        maior_cfl_adv = max(maior_cfl_adv, cfl["adveccao"])
        maior_cfl_diff = max(maior_cfl_diff, cfl["difusao"])

        if not cfl["estavel_difusao"]:
            raise RuntimeError(
                "CFL difusivo violado: "
                f"K*dt*(1/dx^2+1/dz^2)={cfl['difusao']:.3f} > 0.5"
            )
        if not cfl["estavel_adveccao"] and config.abortar_se_cfl_violar:
            raise RuntimeError(
                f"CFL advectivo/sedimentacao violado em t={t_s:.1f}s: "
                f"{cfl['adveccao']:.3f} > {config.cfl_limite:.3f}. "
                "Reduza dt e rode novamente."
            )
        if (
            verbose
            and cfl["adveccao"] > config.cfl_aviso
            and step % save_every == 0
        ):
            print(
                f"AVISO CFL em t={t_s/60:.1f} min: "
                f"C={cfl['adveccao']:.3f}"
            )

        if step % save_every == 0:
            frames["t"].append(t_s)
            frames["T"].append(T_atual.copy())
            frames["qv"].append(qv_atual.copy())
            for nome in ("qc", "Nc", "qr", "Nr", "qi", "Ni", "qs", "Ns", "qg", "Ng"):
                frames[nome].append(getattr(estado, nome).copy())
            frames["qc_qi"].append((estado.qc + estado.qi).copy())
            frames["qr_qs_qg"].append((estado.qr + estado.qs + estado.qg).copy())
            frames["w"].append(w.copy())
            frames["u"].append(u.copy())
            frames["thp"].append(estado.thp.copy())
            frames["qvp"].append(estado.qvp.copy())
            frames["cfl_adv"].append(cfl["adveccao"])
            frames["cfl_diff"].append(cfl["difusao"])

            if verbose:
                condensado_nuvem = estado.qc + estado.qi
                if condensado_nuvem.max() > 1.0e-5:
                    niveis = np.where(condensado_nuvem.max(axis=0) > 1.0e-5)[0]
                    topo = estado.z[niveis.max()]
                else:
                    topo = 0.0
                print(
                    f"t={t_s/60:5.1f} min | "
                    f"w_max={w.max():6.2f} m/s | "
                    f"qg_max={estado.qg.max()*1000:7.4f} g/kg | "
                    f"topo~{topo:6.0f} m | "
                    f"CFL={cfl['adveccao']:.3f} | "
                    f"SHF={shf:5.0f} W/m2 | "
                    f"tempo_real={(time.time()-inicio)/60:5.1f} min"
                )

        if step == nsteps:
            break

        passo_dinamico(estado, config, u, w, dtheta_now, dqv_now)

    for chave, valores in frames.items():
        frames[chave] = np.asarray(valores)

    return {
        "config": config,
        "estado_final": estado,
        "x_m": estado.x,
        "z_m": estado.z,
        "p_pa_1d": estado.p_pa_1d,
        "p_hpa_1d": estado.p_hpa_1d,
        "rho0_1d": estado.rho0_1d,
        "theta_env_1d": estado.theta_env_1d,
        "T_env_1d": estado.T_env_1d,
        "qv_env_1d": estado.qv_env_1d,
        "rh_env_1d": estado.rh_env_1d,
        "frames": frames,
        "cfl_max_adv": maior_cfl_adv,
        "cfl_max_diff": maior_cfl_diff,
    }
