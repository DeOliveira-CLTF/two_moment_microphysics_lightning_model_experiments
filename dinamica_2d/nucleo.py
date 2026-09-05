# -*- coding: utf-8 -*-
"""Nucleo dinamico 2D para nuvem convectiva idealizada.

Este modulo preserva o nucleo Boussinesq vorticidade-funcao de corrente do
modelo fornecido pelo professor e acrescenta os controles necessarios para os
experimentos cientificos:

- perfis ambientais independentes para os experimentos de iniciacao (Grupo 2)
  e de sensibilidade microfisica (Grupos 1 e 3);
- aquecimento uniforme do ambiente em temperatura real (Grupo 2);
- preservacao opcional de umidade relativa no ambiente aquecido;
- forcamento mecanico externo de levantamento (Grupo 2), aplicado como
  aceleracao vertical e convertido em tendencia de vorticidade;
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

    # Perfil ambiental de referencia.
    #
    # 'referencia': perfil original do nucleo/professor, usado no Grupo 2
    #               para os experimentos de iniciacao convectiva.
    # 'microfisica': perfil mais umido e menos estavel adotado nos
    #                 experimentos de sensibilidade microfisica dos
    #                 Grupos 1 e 3.
    perfil_ambiente: str = "referencia"

    # Experimento de aquecimento.
    delta_t_ambiente_k: float = 0.0
    preservar_rh: bool = True

    # Forcamento mecanico externo de levantamento.
    #
    # A amplitude tem unidades de aceleracao vertical [m s-2]. O campo
    # espacial e gaussiano e sua variacao temporal e uma janela suave
    # sen^2. O forcamento NAO prescreve w: ele entra na equacao da
    # vorticidade por meio de d(a_dyn)/dx, e u/w continuam prognosticados.
    #
    # O valor zero preserva exatamente o comportamento anterior do nucleo.
    forc_dyn_amp_m_s2: float = 0.0
    forc_dyn_x0_m: float | None = None
    forc_dyn_z0_m: float = 800.0
    forc_dyn_rx_m: float = 2000.0
    forc_dyn_rz_m: float = 700.0
    forc_dyn_inicio_s: float = 0.0
    forc_dyn_duracao_s: float = 900.0

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
    if config.perfil_ambiente not in {"referencia", "microfisica"}:
        raise ValueError(
            "perfil_ambiente deve ser 'referencia' ou 'microfisica'"
        )
    if config.nc_ativacao_kg1 <= 0.0:
        raise ValueError("nc_ativacao_kg1 deve ser positivo")
    if config.cfl_aviso <= 0.0 or config.cfl_limite <= 0.0:
        raise ValueError("limites de CFL devem ser positivos")
    if config.cfl_aviso > config.cfl_limite:
        raise ValueError("cfl_aviso nao pode exceder cfl_limite")

    if (
        not np.isfinite(config.forc_dyn_amp_m_s2)
        or config.forc_dyn_amp_m_s2 < 0.0
    ):
        raise ValueError(
            "forc_dyn_amp_m_s2 deve ser finito e maior ou igual a zero"
        )

    for nome in ("forc_dyn_rx_m", "forc_dyn_rz_m"):
        valor = getattr(config, nome)
        if not np.isfinite(valor) or valor <= 0.0:
            raise ValueError(f"{nome} deve ser finito e positivo")

    if not np.isfinite(config.forc_dyn_z0_m) or config.forc_dyn_z0_m < 0.0:
        raise ValueError("forc_dyn_z0_m deve ser finito e maior ou igual a zero")

    if (
        not np.isfinite(config.forc_dyn_inicio_s)
        or config.forc_dyn_inicio_s < 0.0
    ):
        raise ValueError(
            "forc_dyn_inicio_s deve ser finito e maior ou igual a zero"
        )

    if (
        not np.isfinite(config.forc_dyn_duracao_s)
        or config.forc_dyn_duracao_s < 0.0
    ):
        raise ValueError(
            "forc_dyn_duracao_s deve ser finito e maior ou igual a zero"
        )

    if (
        config.forc_dyn_amp_m_s2 > 0.0
        and config.forc_dyn_duracao_s <= 0.0
    ):
        raise ValueError(
            "forc_dyn_duracao_s deve ser positiva quando o forcamento "
            "dinamico estiver ativo"
        )

    if (
        config.forc_dyn_x0_m is not None
        and not np.isfinite(config.forc_dyn_x0_m)
    ):
        raise ValueError("forc_dyn_x0_m deve ser None ou um valor finito")


def p_of_z(z_m, escala_m=8000.0):
    return P0_HPA * np.exp(-z_m / escala_m)


def exner(z_m):
    return (p_of_z(z_m) / P0_HPA) ** (Rd / cp)


def qsat_liq(T_k, p_hpa):
    Tc = T_k - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    return EPS_R * es / np.maximum(p_hpa - es, 1.0e-3)


def dtheta_dz_env(z_m, perfil_ambiente="referencia"):
    """Retorna dtheta/dz [K m-1] para o perfil ambiental selecionado.

    Perfis disponiveis
    ------------------
    referencia
        Perfil original do nucleo/professor. Deve ser usado pelo Grupo 2
        quando o objetivo e estudar a iniciacao convectiva sob aquecimento
        e diferentes intensidades de forcamento dinamico.

    microfisica
        Perfil mais umido e menos estavel introduzido para sustentar uma
        tempestade estabelecida nos experimentos de sensibilidade
        microfisica dos Grupos 1 e 3.
    """

    if perfil_ambiente == "referencia":
        return np.where(
            z_m < 1000.0,
            3.0e-3,
            np.where(
                z_m < 2000.0,
                6.5e-3,
                np.where(z_m < 8500.0, 2.0e-3, 6.0e-3),
            ),
        )

    if perfil_ambiente == "microfisica":
        return np.where(
            z_m < 1000.0,
            2.5e-3,
            np.where(
                z_m < 3000.0,
                4.0e-3,
                np.where(z_m < 8500.0, 1.5e-3, 6.0e-3),
            ),
        )

    raise ValueError(
        "perfil_ambiente deve ser 'referencia' ou 'microfisica'"
    )


def RH_env_profile(z_m, perfil_ambiente="referencia"):
    """Retorna o perfil de umidade relativa do ambiente de controle.

    O aquecimento do Grupo 2 e aplicado posteriormente. Quando
    preservar_rh=False, qv do CTRL e mantido e a RH efetiva diminui
    automaticamente no ambiente aquecido.
    """

    if perfil_ambiente == "referencia":
        return np.where(
            z_m < 1000.0,
            0.70,
            np.where(
                z_m < 2000.0,
                0.35,
                np.where(z_m < 8500.0, 0.55, 0.20),
            ),
        )

    if perfil_ambiente == "microfisica":
        return np.where(
            z_m < 1500.0,
            0.95,
            np.where(
                z_m < 4000.0,
                0.70,
                np.where(z_m < 8500.0, 0.40, 0.20),
            ),
        )

    raise ValueError(
        "perfil_ambiente deve ser 'referencia' ou 'microfisica'"
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

    # Perfil ambiental de controle selecionado para este experimento.
    theta_original = np.zeros(config.nz, dtype=float)
    theta_original[0] = THETA0
    grad = dtheta_dz_env(z, config.perfil_ambiente)
    for k in range(1, config.nz):
        theta_original[k] = theta_original[k - 1] + grad[k - 1] * config.dz

    pi_1d = exner(z)
    p_hpa_1d = p_of_z(z)
    p_pa_1d = p_hpa_1d * 100.0

    # O WARM e definido em temperatura real: T_warm(z) = T_ctrl(z) + DeltaT.
    T_original_1d = theta_original * pi_1d
    T_env_1d = T_original_1d + config.delta_t_ambiente_k
    theta_env_1d = T_env_1d / pi_1d

    rh_ctrl_1d = RH_env_profile(z, config.perfil_ambiente)
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



def janela_forcamento_dinamico(
    t_s,
    inicio_s,
    duracao_s,
):
    """
    Retorna uma janela temporal suave entre 0 e 1 para o forcamento mecanico.

    A forma usada e sen^2(pi*tau), com tau variando de 0 a 1 durante o
    intervalo ativo. Assim, o forcamento cresce e decai suavemente, evitando
    uma descontinuidade temporal no termo-fonte de vorticidade.
    """

    if duracao_s <= 0.0:
        return 0.0

    tau = (float(t_s) - float(inicio_s)) / float(duracao_s)

    if tau <= 0.0 or tau >= 1.0:
        return 0.0

    return float(np.sin(np.pi * tau) ** 2)


def campo_forcamento_dinamico(
    estado: EstadoDinamica2D,
    config: ConfiguracaoDinamica2D,
    t_s,
):
    """
    Constroi a aceleracao vertical mecanica externa a_dyn(x,z,t) [m s-2].

    O campo e gaussiano no espaco e suave no tempo. Ele representa um
    levantamento mecanico idealizado. Nao deve ser interpretado como uma
    frente fria explicitamente resolvida.

    Importante:
    - nao altera theta;
    - nao altera qv;
    - nao prescreve w;
    - apenas fornece uma aceleracao vertical externa que, via curl, entra
      como tendencia da vorticidade.
    """

    if config.forc_dyn_amp_m_s2 == 0.0:
        return np.zeros_like(estado.zeta)

    janela = janela_forcamento_dinamico(
        t_s=t_s,
        inicio_s=config.forc_dyn_inicio_s,
        duracao_s=config.forc_dyn_duracao_s,
    )

    if janela == 0.0:
        return np.zeros_like(estado.zeta)

    if config.forc_dyn_x0_m is None:
        x0 = 0.5 * (estado.x[0] + estado.x[-1])
    else:
        x0 = float(config.forc_dyn_x0_m)

    z0 = float(config.forc_dyn_z0_m)

    r2 = (
        ((estado.X - x0) / config.forc_dyn_rx_m) ** 2
        + ((estado.Z - z0) / config.forc_dyn_rz_m) ** 2
    )

    return (
        config.forc_dyn_amp_m_s2
        * janela
        * np.exp(-r2)
    )


def torque_forcamento_dinamico(
    a_dyn,
    dx,
):
    """
    Converte a aceleracao vertical externa em tendencia de vorticidade.

    Para o sistema 2D x-z e a convencao de sinais usada neste nucleo,
    uma aceleracao vertical a_dyn entra na equacao da vorticidade pelo
    gradiente horizontal:

        d(zeta)/dt |_dyn = d(a_dyn)/dx

    Como a_dyn tem unidade [m s-2], o gradiente tem unidade [s-2],
    coerente com uma tendencia de vorticidade.
    """

    a_dyn = np.asarray(a_dyn, dtype=float)
    torque = np.zeros_like(a_dyn)

    torque[1:-1, :] = (
        a_dyn[2:, :]
        - a_dyn[:-2, :]
    ) / (2.0 * dx)

    return torque


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


def diagnosticar_cfl(
    estado: EstadoDinamica2D,
    config: ConfiguracaoDinamica2D,
    u,
    w,
):
    """
    Retorna numeros de Courant para adveccao/sedimentacao e difusao.

    Para o esquema upwind 2D:

        C = |u| dt/dx + |w_eff| dt/dz

    onde:

        w_eff = w

    para campos nao sedimentantes, e

        w_eff = w - Vt

    para hidrometeoros sedimentantes.

    Alem do CFL maximo, retorna a categoria que controla o CFL
    e a localizacao espacial do maximo.
    """

    # ------------------------------------------------------------------
    # Componentes que serao testados
    # ------------------------------------------------------------------

    componentes = [
        {
            "nome": "adveccao_sem_sedimentacao",
            "w_eff": w,
            "vt": np.zeros_like(w),
        }
    ]

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

        componentes.extend(
            [
                {
                    "nome": "chuva_massa_qr",
                    "w_eff": w - Vtq_r,
                    "vt": Vtq_r,
                },
                {
                    "nome": "chuva_numero_Nr",
                    "w_eff": w - Vtn_r,
                    "vt": Vtn_r,
                },
                {
                    "nome": "gelo_massa_qi",
                    "w_eff": w - Vtq_i,
                    "vt": Vtq_i,
                },
                {
                    "nome": "gelo_numero_Ni",
                    "w_eff": w - Vtn_i,
                    "vt": Vtn_i,
                },
                {
                    "nome": "neve_massa_qs",
                    "w_eff": w - Vtq_s,
                    "vt": Vtq_s,
                },
                {
                    "nome": "neve_numero_Ns",
                    "w_eff": w - Vtn_s,
                    "vt": Vtn_s,
                },
                {
                    "nome": "graupel_massa_qg",
                    "w_eff": w - Vtq_g,
                    "vt": Vtq_g,
                },
                {
                    "nome": "graupel_numero_Ng",
                    "w_eff": w - Vtn_g,
                    "vt": Vtn_g,
                },
            ]
        )

    # ------------------------------------------------------------------
    # Procura o maior CFL entre todos os componentes
    # ------------------------------------------------------------------

    melhor = None

    for componente in componentes:

        w_eff = componente["w_eff"]

        cfl_x = (
            np.abs(u)
            * config.dt
            / config.dx
        )

        cfl_z = (
            np.abs(w_eff)
            * config.dt
            / config.dz
        )

        campo_cfl = cfl_x + cfl_z

        indice = np.unravel_index(
            np.nanargmax(campo_cfl),
            campo_cfl.shape,
        )

        ix, iz = indice

        valor = float(
            campo_cfl[ix, iz]
        )

        if melhor is None or valor > melhor["valor"]:

            melhor = {
                "valor": valor,
                "nome": componente["nome"],
                "ix": int(ix),
                "iz": int(iz),
                "x_m": float(estado.x[ix]),
                "z_m": float(estado.z[iz]),
                "u": float(u[ix, iz]),
                "w": float(w[ix, iz]),
                "vt": float(
                    componente["vt"][ix, iz]
                ),
                "w_eff": float(
                    w_eff[ix, iz]
                ),
                "cfl_x": float(
                    cfl_x[ix, iz]
                ),
                "cfl_z": float(
                    cfl_z[ix, iz]
                ),
            }

    cfl_adv = melhor["valor"]

    # ------------------------------------------------------------------
    # CFL difusivo
    # ------------------------------------------------------------------

    cfl_diff = float(
        config.difusao
        * config.dt
        * (
            1.0 / config.dx**2
            + 1.0 / config.dz**2
        )
    )

    # ------------------------------------------------------------------
    # Retorno
    # ------------------------------------------------------------------

    return {
        "adveccao": cfl_adv,
        "difusao": cfl_diff,

        "estavel_adveccao": (
            cfl_adv <= config.cfl_limite
        ),

        "estavel_difusao": (
            cfl_diff <= 0.5
        ),

        # Diagnosticos adicionais
        "componente_max": melhor["nome"],
        "ix_max": melhor["ix"],
        "iz_max": melhor["iz"],
        "x_max_m": melhor["x_m"],
        "z_max_m": melhor["z_m"],

        "u_local_m_s": melhor["u"],
        "w_local_m_s": melhor["w"],
        "vt_local_m_s": melhor["vt"],
        "w_eff_local_m_s": melhor["w_eff"],

        "cfl_x_local": melhor["cfl_x"],
        "cfl_z_local": melhor["cfl_z"],
    }


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
        "a_dyn": [],
        "torque_dyn": [],
        "cfl_adv": [],
        "cfl_diff": [],
    }


def passo_dinamico(
    estado: EstadoDinamica2D,
    config: ConfiguracaoDinamica2D,
    u,
    w,
    dtheta_dz_now,
    dqv_dz_now,
    t_s=0.0,
):
    """
    Avanca a dinamica e transporta os campos microfisicos por um passo de tempo.

    Inclui:
    - adveccao de vorticidade;
    - forca de empuxo;
    - forcamento mecanico externo de levantamento;
    - adveccao de theta' e qv';
    - difusao;
    - transporte dos hidrometeoros;
    - sedimentacao de chuva, gelo, neve e graupel;
    - atualizacao da funcao de corrente.
    """

    # ------------------------------------------------------------------
    # 1. Velocidades terminais dos hidrometeoros
    # ------------------------------------------------------------------

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

        Vtq_r = zero
        Vtn_r = zero
        Vtq_i = zero
        Vtn_i = zero
        Vtq_s = zero
        Vtn_s = zero
        Vtq_g = zero
        Vtn_g = zero

    # ------------------------------------------------------------------
    # 2. Temperatura potencial virtual perturbada
    # ------------------------------------------------------------------

    thv = (
        estado.thp
        + 0.61 * THETA0 * estado.qvp
        - THETA0
        * (
            estado.qc
            + estado.qi
            + estado.qr
            + estado.qs
            + estado.qg
        )
    )

    # Gradiente horizontal.
    dthv_dx = np.zeros_like(estado.thp)

    dthv_dx[1:-1, :] = (
        thv[2:, :]
        - thv[:-2, :]
    ) / (2.0 * config.dx)

    # Torque associado ao empuxo.
    buoy_torque = (
        G / THETA0
    ) * dthv_dx

    # Forcamento mecanico externo.
    #
    # a_dyn e uma aceleracao vertical [m s-2]. Seu gradiente horizontal
    # produz uma tendencia adicional de vorticidade [s-2]. Isso permite
    # intensificar o levantamento sem aquecer artificialmente a parcela
    # e sem prescrever diretamente a velocidade vertical.
    a_dyn = campo_forcamento_dinamico(
        estado=estado,
        config=config,
        t_s=t_s,
    )

    dyn_torque = torque_forcamento_dinamico(
        a_dyn=a_dyn,
        dx=config.dx,
    )

    # ------------------------------------------------------------------
    # 3. Tendencia da vorticidade
    # ------------------------------------------------------------------

    dzeta = (
        upwind_advect(
            estado.zeta,
            u,
            w,
            config.dx,
            config.dz,
        )
        + buoy_torque
        + dyn_torque
        + config.difusao
        * laplacian(
            estado.zeta,
            config.dx,
            config.dz,
        )
    )

    # ------------------------------------------------------------------
    # 4. Tendencia de theta'
    # ------------------------------------------------------------------

    dthp = (
        upwind_advect(
            estado.thp,
            u,
            w,
            config.dx,
            config.dz,
        )
        - w
        * dtheta_dz_now[None, :]
        + config.difusao
        * laplacian(
            estado.thp,
            config.dx,
            config.dz,
        )
    )

    # ------------------------------------------------------------------
    # 5. Tendencia de qv'
    # ------------------------------------------------------------------

    dqvp = (
        upwind_advect(
            estado.qvp,
            u,
            w,
            config.dx,
            config.dz,
        )
        - w
        * dqv_dz_now[None, :]
        + config.difusao
        * laplacian(
            estado.qvp,
            config.dx,
            config.dz,
        )
    )

    # ------------------------------------------------------------------
    # 6. Agua de nuvem
    # ------------------------------------------------------------------

    dqc = (
        upwind_advect(
            estado.qc,
            u,
            w,
            config.dx,
            config.dz,
        )
        + config.difusao
        * laplacian(
            estado.qc,
            config.dx,
            config.dz,
        )
    )

    dNc = (
        upwind_advect(
            estado.Nc,
            u,
            w,
            config.dx,
            config.dz,
        )
        + config.difusao
        * laplacian(
            estado.Nc,
            config.dx,
            config.dz,
        )
    )

    # ------------------------------------------------------------------
    # 7. Atualizacao dos campos nao sedimentantes
    # ------------------------------------------------------------------

    estado.zeta += (
        config.dt * dzeta
    )

    estado.thp += (
        config.dt * dthp
    )

    estado.qvp += (
        config.dt * dqvp
    )

    estado.qc = np.maximum(
        estado.qc
        + config.dt * dqc,
        0.0,
    )

    estado.Nc = np.maximum(
        estado.Nc
        + config.dt * dNc,
        0.0,
    )

    # ------------------------------------------------------------------
    # 8. Hidrometeoros sedimentantes
    # ------------------------------------------------------------------

    if config.microfisica == "thompson":

        pares = (
            ("qr", "Nr", Vtq_r, Vtn_r),
            ("qi", "Ni", Vtq_i, Vtn_i),
            ("qs", "Ns", Vtq_s, Vtn_s),
            ("qg", "Ng", Vtq_g, Vtn_g),
        )

        for (
            qnome,
            nnome,
            Vtq,
            Vtn,
        ) in pares:

            qcampo = getattr(
                estado,
                qnome,
            )

            Ncampo = getattr(
                estado,
                nnome,
            )

            # Massa:
            # velocidade vertical efetiva = w - Vtq
            dq = (
                upwind_advect(
                    qcampo,
                    u,
                    w - Vtq,
                    config.dx,
                    config.dz,
                )
                + config.difusao
                * laplacian(
                    qcampo,
                    config.dx,
                    config.dz,
                )
            )

            # Numero:
            # velocidade vertical efetiva = w - Vtn
            dN = (
                upwind_advect(
                    Ncampo,
                    u,
                    w - Vtn,
                    config.dx,
                    config.dz,
                )
                + config.difusao
                * laplacian(
                    Ncampo,
                    config.dx,
                    config.dz,
                )
            )

            setattr(
                estado,
                qnome,
                np.maximum(
                    qcampo
                    + config.dt * dq,
                    0.0,
                ),
            )

            setattr(
                estado,
                nnome,
                np.maximum(
                    Ncampo
                    + config.dt * dN,
                    0.0,
                ),
            )

    # ------------------------------------------------------------------
    # 9. Condicoes de contorno
    # ------------------------------------------------------------------

    estado.zeta = aplicar_bordas(
        estado.zeta,
        zero_grad=False,
    )

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

        setattr(
            estado,
            nome,
            aplicar_bordas(
                getattr(
                    estado,
                    nome,
                ),
                zero_grad=True,
            ),
        )

    # ------------------------------------------------------------------
    # 10. Recupera psi a partir da nova vorticidade
    # ------------------------------------------------------------------

    estado.psi = poisson_jacobi(
        estado.zeta,
        estado.psi,
        config.dx,
        config.dz,
        niter=config.iteracoes_poisson,
    )

def rodar_thompson_2d(config: ConfiguracaoDinamica2D | None = None, verbose=False):
    """Executa o modelo 2D acoplado a microfisica de dois momentos.

    O forcamento mecanico externo, quando ativado, modifica a tendencia de
    vorticidade. A velocidade vertical continua sendo obtida da funcao de
    corrente, portanto nao e prescrita diretamente.
    """

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
            print()
            print("=" * 78)
            print("DIAGNOSTICO DA VIOLACAO DE CFL")
            print("=" * 78)

            print(
                f"tempo                  = {t_s:.1f} s "
                f"({t_s / 60.0:.2f} min)"
            )

            print(
                f"CFL total              = {cfl['adveccao']:.4f}"
            )

            print(
                f"componente controlador = {cfl['componente_max']}"
            )

            print(
                f"x                      = {cfl['x_max_m']:.1f} m"
            )

            print(
                f"z                      = {cfl['z_max_m']:.1f} m"
            )

            print(
                f"u local                = {cfl['u_local_m_s']:.3f} m/s"
            )

            print(
                f"w local                = {cfl['w_local_m_s']:.3f} m/s"
            )

            print(
                f"Vt local               = {cfl['vt_local_m_s']:.3f} m/s"
            )

            print(
                f"w_eff = w - Vt         = {cfl['w_eff_local_m_s']:.3f} m/s"
            )

            print(
                f"CFL horizontal         = {cfl['cfl_x_local']:.4f}"
            )

            print(
                f"CFL vertical           = {cfl['cfl_z_local']:.4f}"
            )

            print("=" * 78)
            print()

            raise RuntimeError(
                f"CFL advectivo/sedimentacao violado em t={t_s:.1f}s: "
                f"{cfl['adveccao']:.3f} > {config.cfl_limite:.3f}. "
                f"Componente controlador: {cfl['componente_max']}. "
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

            # Salva tambem o forcamento mecanico aplicado naquele instante.
            # Isso facilita a auditoria dos casos D0/D1 do Grupo 2.
            a_dyn_atual = campo_forcamento_dinamico(
                estado=estado,
                config=config,
                t_s=t_s,
            )
            torque_dyn_atual = torque_forcamento_dinamico(
                a_dyn=a_dyn_atual,
                dx=config.dx,
            )

            frames["a_dyn"].append(a_dyn_atual.copy())
            frames["torque_dyn"].append(torque_dyn_atual.copy())

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
                    f"a_dyn={np.max(a_dyn_atual):.4f} m/s2 | "
                    f"SHF={shf:5.0f} W/m2 | "
                    f"tempo_real={(time.time()-inicio)/60:5.1f} min"
                )

        if step == nsteps:
            break

        passo_dinamico(
            estado,
            config,
            u,
            w,
            dtheta_now,
            dqv_now,
            t_s=t_s,
        )

    for chave, valores in frames.items():
        frames[chave] = np.asarray(valores)

    return {
        "config": config,
        "perfil_ambiente": config.perfil_ambiente,
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
