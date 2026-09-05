# -*- coding: utf-8 -*-
"""
Grupo 2 - Aquecimento, forcamento dinamico e diagnosticos eletricos
===================================================================

Este driver implementa a nova formulacao do Grupo Experimental 2 do repositorio:

    DeOliveira-CLTF/two_moment_microphysics_lightning_model_experiments

HIPOTESE CIENTIFICA
-------------------
O objetivo e separar dois fatores:

1. aquecimento ambiental com qv inicial preservado;
2. intensidade de um forcamento mecanico externo de levantamento.

O Grupo 2 usa explicitamente o perfil ambiental de referencia do nucleo:

    perfil_ambiente = "referencia"

Esse perfil fica separado do perfil "microfisica", usado pelos Grupos 1 e 3.
Assim, alteracoes feitas nos experimentos microfisicos nao redefinem
silenciosamente o ambiente-base do Grupo 2.

Nos casos WARM:

    T_warm(z) = T_ctrl(z) + DeltaT
    qv_warm(z) = qv_ctrl(z)

Portanto, a umidade relativa NAO e preservada. Como qsat aumenta com a
 temperatura, a RH tende a diminuir no ambiente aquecido. Isso permite testar
se o aquecimento aumenta a inibicao convectiva e torna a conveccao mais
dependente de um mecanismo externo de levantamento.

A intensidade dinamica NAO e representada por uma bolha mais quente. A bolha
termica e a perturbacao adicional de vapor sao desligadas neste grupo:

    bolha_k = 0
    bolha_qv_kgkg = 0

O segundo fator experimental e a amplitude da aceleracao vertical mecanica
externa implementada no nucleo:

    forc_dyn_amp_m_s2

Essa aceleracao entra na equacao prognostica da vorticidade por meio de seu
gradiente horizontal. Assim, w continua sendo prognosticado pelo nucleo e nao
prescrito diretamente.

MATRIZ FINAL
------------

    CTRL             : DeltaT = 0 K,  forcamento = D0
    WARM             : DeltaT = +4 K, forcamento = D0
    DYN_PLUS         : DeltaT = 0 K,  forcamento = D1
    WARM_DYN_PLUS    : DeltaT = +4 K, forcamento = D1

com D1 > D0.

FLUXO DE TRABALHO
-----------------

ETAPA 1 - varredura preliminar do forcamento dinamico

    python experiments/group2_warming_lightning/experimento_grupo2.py varredura

Por padrao sao testadas amplitudes preliminares:

    0.001, 0.002, 0.005, 0.010 e 0.020 m s-2

Esses valores sao apenas uma grade inicial de sensibilidade. A varredura serve
para identificar uma faixa de forcamento que:

- produza conveccao profunda;
- alcance a fase mista;
- produza graupel em torno de -15 C;
- gere movimento ascendente em -15 C;
- permaneça numericamente estavel segundo CFL.

A escolha de D0 e D1 NAO deve ser feita com base no maior F3 ou LPI*. Esses
proxies sao resultados do experimento, nao criterios para calibrar o forcamento.

ETAPA 2 - escolha de D0 e D1

Escolher:

    D0 = forcamento de referencia, robusto mas nao excessivo
    D1 = forcamento claramente mais intenso que D0

A geometria e a duracao do forcamento devem permanecer identicas; apenas a
amplitude muda.

ETAPA 3 - matriz final

Exemplo de sintaxe, APENAS depois da varredura:

    python experiments/group2_warming_lightning/experimento_grupo2.py final \
        --d0 0.002 --d1 0.010

IMPORTANTE
----------
McCaul F1/F2/F3 e LPI* sao proxies de atividade eletrica. Este modelo nao
simula explicitamente flashes observados.
"""

# ============================================================================
# 1. IMPORTACOES
# ============================================================================

import argparse
import csv
import json
import subprocess
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np


# ============================================================================
# 2. RAIZ DO REPOSITORIO
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# 3. API DO REPOSITORIO
# ============================================================================

from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d
from microfisica.constantes import QMIN
from lightning import diagnosticar_relampagos_2d, resumir_diagnosticos_2d


# ============================================================================
# 4. CONFIGURACAO COMUM DO GRUPO 2
# ============================================================================

# Perfil ambiental do Grupo 2.
#
# "referencia" recupera o sounding-base usado nos experimentos de iniciacao.
# O perfil "microfisica" fica reservado aos Grupos 1 e 3.
PERFIL_AMBIENTE_GRUPO2 = "referencia"

# Aquecimento dos casos WARM.
DELTA_T_WARM_K = 4.0

# Nc mantido fixo neste grupo.
NC_CONTROLE_KG1 = 2.0e8

# Grade.
NX = 90
NZ = 110
DX_M = 100.0
DZ_M = 100.0

# Integracao temporal.
DT_S = 1.0
SALVAR_A_CADA_S = 300.0
TEMPO_PADRAO_MIN = 40.0

# Limiar usado pelo LPI*.
W_LPI_THRESHOLD_M_S = 0.5

# ---------------------------------------------------------------------------
# Geometria e duracao FIXAS do forcamento dinamico.
# Somente forc_dyn_amp_m_s2 varia entre D0 e D1.
# ---------------------------------------------------------------------------

FORC_DYN_X0_M = None       # None = centro horizontal do dominio
FORC_DYN_Z0_M = 800.0      # centro vertical do levantamento
FORC_DYN_RX_M = 2000.0     # escala horizontal gaussiana
FORC_DYN_RZ_M = 700.0      # escala vertical gaussiana
FORC_DYN_INICIO_S = 0.0
FORC_DYN_DURACAO_S = 900.0 # 15 min

# Sem bolha termodinamica no novo Grupo 2.
BOLHA_K_GRUPO2 = 0.0
BOLHA_QV_GRUPO2_KGKG = 0.0

# Diretorio de saida.
OUTPUT_BASE = ROOT / "outputs" / "group2"


# ============================================================================
# 5. RASTREABILIDADE
# ============================================================================

def obter_commit_git():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "commit_indisponivel"


def comando_executado():
    return " ".join([sys.executable, *sys.argv])


def escrever_texto(caminho, texto):
    caminho.write_text(str(texto) + "\n", encoding="utf-8")


def salvar_json(caminho, objeto):
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(objeto, arquivo, indent=2, ensure_ascii=False)


def rotulo_forcamento(valor):
    texto = f"{float(valor):g}".replace(".", "p")
    return f"{texto}ms2"


# ============================================================================
# 6. CONFIGURACAO DE UM CASO
# ============================================================================

def criar_configuracao(
    caso,
    delta_t_ambiente_k,
    forc_dyn_amp_m_s2,
    tempo_min,
):
    """
    Constroi um caso do novo Grupo 2.

    Entre os quatro casos finais mudam apenas:

        delta_t_ambiente_k
        forc_dyn_amp_m_s2

    O perfil ambiental permanece explicitamente fixo em "referencia".
    Todo o restante permanece fixo.
    """

    return ConfiguracaoDinamica2D(
        nx=NX,
        nz=NZ,
        dx=DX_M,
        dz=DZ_M,
        dt=DT_S,
        tempo_total_s=float(tempo_min) * 60.0,
        salvar_a_cada_s=SALVAR_A_CADA_S,

        # Grupo 2 novo: sem bolha termodinamica.
        bolha_k=BOLHA_K_GRUPO2,
        bolha_qv_kgkg=BOLHA_QV_GRUPO2_KGKG,

        # Ambiente-base proprio do Grupo 2.
        perfil_ambiente=PERFIL_AMBIENTE_GRUPO2,

        # Aquecimento ambiental.
        delta_t_ambiente_k=float(delta_t_ambiente_k),

        # Mantem qv do CTRL. A RH pode diminuir quando T aumenta.
        preservar_rh=False,

        # Forcamento mecanico externo.
        forc_dyn_amp_m_s2=float(forc_dyn_amp_m_s2),
        forc_dyn_x0_m=FORC_DYN_X0_M,
        forc_dyn_z0_m=FORC_DYN_Z0_M,
        forc_dyn_rx_m=FORC_DYN_RX_M,
        forc_dyn_rz_m=FORC_DYN_RZ_M,
        forc_dyn_inicio_s=FORC_DYN_INICIO_S,
        forc_dyn_duracao_s=FORC_DYN_DURACAO_S,

        # Microfisica.
        nc_ativacao_kg1=NC_CONTROLE_KG1,
        microfisica="thompson",
        evap_chuva=True,

        # Outras fisicas.
        radiacao=False,
        ciclo_diurno=False,

        # Identificador.
        cenario=caso,

        # Seguranca numerica.
        cfl_aviso=0.80,
        cfl_limite=1.00,
        abortar_se_cfl_violar=True,
    )


# ============================================================================
# 7. FUNCOES NUMERICAS SEGURAS
# ============================================================================

def maximo_seguro(campo):
    campo = np.asarray(campo, dtype=float)
    if campo.size == 0 or np.all(np.isnan(campo)):
        return np.nan
    return float(np.nanmax(campo))


def media_segura(campo):
    campo = np.asarray(campo, dtype=float)
    if campo.size == 0 or np.all(np.isnan(campo)):
        return np.nan
    return float(np.nanmean(campo))


def razao_segura(numerador, denominador):
    if not np.isfinite(numerador):
        return np.nan
    if not np.isfinite(denominador):
        return np.nan
    if denominador == 0.0:
        return np.nan
    return float(numerador / denominador)


# ============================================================================
# 8. SALVAMENTO COMPLETO
# ============================================================================

def salvar_npz(resultado, diagnosticos, caminho):
    frames = resultado["frames"]

    dados = {
        "t_s": np.asarray(frames["t"]),
        "x_m": np.asarray(resultado["x_m"]),
        "z_m": np.asarray(resultado["z_m"]),
        "p_pa_1d": np.asarray(resultado["p_pa_1d"]),
        "rho0_1d": np.asarray(resultado["rho0_1d"]),
        "theta_env_1d": np.asarray(resultado["theta_env_1d"]),
        "T_env_1d": np.asarray(resultado["T_env_1d"]),
        "qv_env_1d": np.asarray(resultado["qv_env_1d"]),
        "rh_env_1d": np.asarray(resultado["rh_env_1d"]),
        "cfl_max_adv": np.asarray(resultado["cfl_max_adv"]),
        "cfl_max_diff": np.asarray(resultado["cfl_max_diff"]),
    }

    # Preserva automaticamente a_dyn, torque_dyn e qualquer outro novo campo
    # que o nucleo salve em frames.
    for nome, valores in frames.items():
        if nome == "t":
            continue
        dados[nome] = np.asarray(valores)

    for nome, valores in diagnosticos.items():
        if nome in {"t_s", "x_m"}:
            continue
        dados[f"lightning_{nome}"] = np.asarray(valores)

    np.savez_compressed(caminho, **dados)


# ============================================================================
# 9. DIAGNOSTICOS DO AMBIENTE INICIAL
# ============================================================================

def diagnosticos_ambiente_inicial(resultado):
    z = np.asarray(resultado["z_m"], dtype=float)
    T = np.asarray(resultado["T_env_1d"], dtype=float)
    qv = np.asarray(resultado["qv_env_1d"], dtype=float)
    rh = np.asarray(resultado["rh_env_1d"], dtype=float)

    camada_0_2km = z <= 2000.0

    return {
        "T_superficie_K": float(T[0]),
        "qv_superficie_kgkg": float(qv[0]),
        "RH_superficie": float(rh[0]),
        "RH_0_2km_media": media_segura(rh[camada_0_2km]),
        "RH_min_perfil": float(np.nanmin(rh)),
        "RH_max_perfil": float(np.nanmax(rh)),
    }


# ============================================================================
# 10. DIAGNOSTICOS DINAMICOS E MICROFISICOS
# ============================================================================

def diagnosticos_dinamicos_microfisicos(resultado, diagnosticos_eletricos):
    frames = resultado["frames"]

    z_m = np.asarray(resultado["z_m"], dtype=float)
    t_s = np.asarray(frames["t"], dtype=float)

    w = np.asarray(frames["w"], dtype=float)
    T = np.asarray(frames["T"], dtype=float)

    qc = np.asarray(frames["qc"], dtype=float)
    qr = np.asarray(frames["qr"], dtype=float)
    qi = np.asarray(frames["qi"], dtype=float)
    qs = np.asarray(frames["qs"], dtype=float)
    qg = np.asarray(frames["qg"], dtype=float)

    # Movimento ascendente maximo.
    w_max = maximo_seguro(w)
    indice_w = np.unravel_index(np.nanargmax(w), w.shape)
    it_w, _ix_w, iz_w = indice_w

    z_w_max_m = float(z_m[iz_w])
    tempo_w_max_s = float(t_s[it_w])

    # Topo da nuvem.
    q_total = qc + qr + qi + qs + qg
    mascara_nuvem = q_total > QMIN

    if np.any(mascara_nuvem):
        indices_z = np.where(np.any(mascara_nuvem, axis=(0, 1)))[0]
        topo_nuvem_m = float(z_m[indices_z[-1]])
    else:
        topo_nuvem_m = np.nan

    # Fase mista: 0 a -20 C.
    mascara_fase_mista = (T <= 273.15) & (T >= 253.15)

    q_liquido = qc + qr
    q_congelado = qi + qs + qg

    q_liquido_fm = np.where(mascara_fase_mista, q_liquido, np.nan)
    q_congelado_fm = np.where(mascara_fase_mista, q_congelado, np.nan)
    qg_fm = np.where(mascara_fase_mista, qg, np.nan)

    coexistencia = (
        mascara_fase_mista
        & (q_liquido > QMIN)
        & (q_congelado > QMIN)
    )

    fase_mista_ativa = bool(np.any(coexistencia))

    # Diagnosticos em -15 C fornecidos pelo modulo lightning.
    w_minus15 = np.asarray(
        diagnosticos_eletricos["w_minus15_m_s"],
        dtype=float,
    )
    qg_minus15 = np.asarray(
        diagnosticos_eletricos["qg_minus15_kgkg"],
        dtype=float,
    )

    w_minus15_max = maximo_seguro(w_minus15)
    qg_minus15_max = maximo_seguro(qg_minus15)

    graupel_minus15_ativo = bool(
        np.isfinite(qg_minus15_max)
        and qg_minus15_max > QMIN
    )

    updraft_minus15_ativo = bool(
        np.isfinite(w_minus15_max)
        and w_minus15_max > W_LPI_THRESHOLD_M_S
    )

    return {
        "w_max_m_s": w_max,
        "tempo_w_max_s": tempo_w_max_s,
        "z_w_max_m": z_w_max_m,
        "topo_nuvem_m": topo_nuvem_m,

        "qc_max_kgkg": maximo_seguro(qc),
        "qr_max_kgkg": maximo_seguro(qr),
        "qi_max_kgkg": maximo_seguro(qi),
        "qs_max_kgkg": maximo_seguro(qs),
        "qg_max_kgkg": maximo_seguro(qg),

        "q_liquido_fase_mista_max_kgkg": maximo_seguro(q_liquido_fm),
        "q_congelado_fase_mista_max_kgkg": maximo_seguro(q_congelado_fm),
        "qg_fase_mista_max_kgkg": maximo_seguro(qg_fm),

        "fase_mista_ativa": fase_mista_ativa,

        "w_minus15_max_m_s": w_minus15_max,
        "qg_minus15_max_kgkg": qg_minus15_max,

        "graupel_minus15_ativo": graupel_minus15_ativo,
        "updraft_minus15_ativo": updraft_minus15_ativo,
    }


# ============================================================================
# 11. CRITERIOS DE ADEQUACAO PARA D0
# ============================================================================

def avaliar_criterios_d0(resumo):
    """
    Identifica forcamentos capazes de produzir uma tempestade de referencia.

    F3 e LPI* sao registrados, mas NAO entram no criterio de calibracao de D0.
    Isso evita escolher a intensidade dinamica com base no proprio resultado
    eletrico que sera analisado cientificamente depois.
    """

    cfl_adv_ok = bool(
        np.isfinite(resumo["cfl_max_adv"])
        and resumo["cfl_max_adv"] <= 1.0
    )

    cfl_diff_ok = bool(
        np.isfinite(resumo["cfl_max_diff"])
        and resumo["cfl_max_diff"] <= 0.5
    )

    cfl_ok = bool(cfl_adv_ok and cfl_diff_ok)

    f3_ativo = bool(
        np.isfinite(resumo["f3_max"])
        and resumo["f3_max"] > 0.0
    )

    lpi_ativo = bool(
        np.isfinite(resumo["lpi_star_max"])
        and resumo["lpi_star_max"] > 0.0
    )

    candidato_d0 = bool(
        resumo["fase_mista_ativa"]
        and resumo["graupel_minus15_ativo"]
        and resumo["updraft_minus15_ativo"]
        and cfl_ok
    )

    return {
        "cfl_adv_ok": cfl_adv_ok,
        "cfl_diff_ok": cfl_diff_ok,
        "cfl_ok": cfl_ok,
        "f3_ativo": f3_ativo,
        "lpi_ativo": lpi_ativo,
        "candidato_D0": candidato_d0,
    }


# ============================================================================
# 12. EXECUCAO DE UM CASO
# ============================================================================

def executar_caso(
    caso,
    delta_t_ambiente_k,
    forc_dyn_amp_m_s2,
    tempo_min,
    output_base,
):
    print()
    print("=" * 78)
    print(f"INICIANDO CASO: {caso}")
    print(f"Perfil ambiente   = {PERFIL_AMBIENTE_GRUPO2}")
    print(f"Delta T ambiente = {delta_t_ambiente_k:.3f} K")
    print(f"Forcamento dyn    = {forc_dyn_amp_m_s2:.6f} m/s2")
    print("qv ambiental      = preservado em relacao ao CTRL")
    print("bolha termica     = desligada")
    print("=" * 78)

    config = criar_configuracao(
        caso=caso,
        delta_t_ambiente_k=delta_t_ambiente_k,
        forc_dyn_amp_m_s2=forc_dyn_amp_m_s2,
        tempo_min=tempo_min,
    )

    pasta_caso = output_base / caso
    pasta_caso.mkdir(parents=True, exist_ok=True)

    resultado = rodar_thompson_2d(config, verbose=True)

    perfil_resultado = resultado.get("perfil_ambiente", PERFIL_AMBIENTE_GRUPO2)
    if perfil_resultado != PERFIL_AMBIENTE_GRUPO2:
        raise RuntimeError(
            "O nucleo retornou um perfil ambiental diferente do solicitado "
            f"pelo Grupo 2: esperado={PERFIL_AMBIENTE_GRUPO2!r}, "
            f"recebido={perfil_resultado!r}."
        )

    diagnosticos = diagnosticar_relampagos_2d(resultado)
    resumo_lightning = resumir_diagnosticos_2d(diagnosticos)

    resumo_dm = diagnosticos_dinamicos_microfisicos(
        resultado=resultado,
        diagnosticos_eletricos=diagnosticos,
    )

    resumo_amb = diagnosticos_ambiente_inicial(resultado)

    resumo = {
        "caso": caso,
        "perfil_ambiente": PERFIL_AMBIENTE_GRUPO2,
        "delta_t_ambiente_k": float(delta_t_ambiente_k),
        "preservar_rh": False,
        "qv_inicial_preservado": True,
        "bolha_k": BOLHA_K_GRUPO2,
        "bolha_qv_kgkg": BOLHA_QV_GRUPO2_KGKG,
        "forc_dyn_amp_m_s2": float(forc_dyn_amp_m_s2),
        "forc_dyn_z0_m": FORC_DYN_Z0_M,
        "forc_dyn_rx_m": FORC_DYN_RX_M,
        "forc_dyn_rz_m": FORC_DYN_RZ_M,
        "forc_dyn_inicio_s": FORC_DYN_INICIO_S,
        "forc_dyn_duracao_s": FORC_DYN_DURACAO_S,
        "tempo_total_min": float(tempo_min),

        "cfl_max_adv": float(resultado["cfl_max_adv"]),
        "cfl_max_diff": float(resultado["cfl_max_diff"]),

        **resumo_amb,
        **resumo_dm,
        **resumo_lightning,
    }

    resumo.update(avaliar_criterios_d0(resumo))

    salvar_npz(
        resultado=resultado,
        diagnosticos=diagnosticos,
        caminho=pasta_caso / f"resultados_{caso}.npz",
    )

    salvar_json(pasta_caso / "configuracao.json", asdict(config))
    salvar_json(pasta_caso / "resumo.json", resumo)
    escrever_texto(pasta_caso / "comando.txt", comando_executado())
    escrever_texto(pasta_caso / "commit.txt", obter_commit_git())

    print(f"Caso {caso} concluido.")
    print(f"RH superficie    = {100.0 * resumo['RH_superficie']:.1f} %")
    print(f"RH media 0-2 km  = {100.0 * resumo['RH_0_2km_media']:.1f} %")
    print(f"CFL max adv/sed  = {resumo['cfl_max_adv']:.4f}")
    print(f"CFL max diff     = {resumo['cfl_max_diff']:.6f}")
    print(f"w max            = {resumo['w_max_m_s']:.3f} m/s")
    print(f"w max em -15 C   = {resumo['w_minus15_max_m_s']:.3f} m/s")
    print(f"qg max em -15 C  = {resumo['qg_minus15_max_kgkg']:.6e} kg/kg")
    print(f"F3 max           = {resumo['f3_max']:.6g}")
    print(f"LPI* max         = {resumo['lpi_star_max']:.6g}")
    print(f"fase mista       = {resumo['fase_mista_ativa']}")
    print(f"candidato D0     = {resumo['candidato_D0']}")
    print(f"Saida            = {pasta_caso}")

    return resultado, diagnosticos, resumo


# ============================================================================
# 13. TABELAS
# ============================================================================

def salvar_tabela_resumo(resumos, caminho):
    if not resumos:
        return

    colunas = []
    for resumo in resumos:
        for chave in resumo:
            if chave not in colunas:
                colunas.append(chave)

    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(resumos)


# ============================================================================
# 14. FIGURA DA VARREDURA DE FORCAMENTO
# ============================================================================

def salvar_figura_varredura(resumos, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resumos = sorted(resumos, key=lambda r: r["forc_dyn_amp_m_s2"])

    forc = np.asarray([r["forc_dyn_amp_m_s2"] for r in resumos], dtype=float)
    wmax = np.asarray([r["w_max_m_s"] for r in resumos], dtype=float)
    topo = np.asarray([r["topo_nuvem_m"] / 1000.0 for r in resumos], dtype=float)
    w15 = np.asarray([r["w_minus15_max_m_s"] for r in resumos], dtype=float)
    qg15 = np.asarray(
        [r["qg_minus15_max_kgkg"] * 1000.0 for r in resumos],
        dtype=float,
    )
    f3 = np.asarray([r["f3_max"] for r in resumos], dtype=float)
    lpi = np.asarray([r["lpi_star_max"] for r in resumos], dtype=float)
    rh0 = np.asarray([100.0 * r["RH_superficie"] for r in resumos], dtype=float)
    cfl = np.asarray([r["cfl_max_adv"] for r in resumos], dtype=float)

    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True)

    axes[0, 0].plot(forc, wmax, marker="o")
    axes[0, 0].set_title("(a) Movimento ascendente maximo")
    axes[0, 0].set_ylabel("w max [m s$^{-1}$]")

    axes[0, 1].plot(forc, topo, marker="o")
    axes[0, 1].set_title("(b) Topo maximo da nuvem")
    axes[0, 1].set_ylabel("altura [km]")

    axes[1, 0].plot(forc, w15, marker="o")
    axes[1, 0].set_title("(c) Movimento ascendente em -15 °C")
    axes[1, 0].set_ylabel("w max [m s$^{-1}$]")

    axes[1, 1].plot(forc, qg15, marker="o")
    axes[1, 1].set_title("(d) Graupel em -15 °C")
    axes[1, 1].set_ylabel("qg max [g kg$^{-1}$]")

    axes[2, 0].plot(forc, f3, marker="o")
    axes[2, 0].set_title("(e) McCaul F3")
    axes[2, 0].set_ylabel("F3 max")

    axes[2, 1].plot(forc, lpi, marker="o")
    axes[2, 1].set_title("(f) Lightning Potential Index")
    axes[2, 1].set_ylabel("LPI* max")

    axes[3, 0].plot(forc, rh0, marker="o")
    axes[3, 0].set_title("(g) RH inicial na superficie")
    axes[3, 0].set_ylabel("RH [%]")
    axes[3, 0].set_xlabel("forcamento dinamico [m s$^{-2}$]")

    axes[3, 1].plot(forc, cfl, marker="o")
    axes[3, 1].axhline(1.0, linestyle="--")
    axes[3, 1].axhline(0.8, linestyle=":")
    axes[3, 1].set_title("(h) CFL advectivo/sedimentacao")
    axes[3, 1].set_ylabel("CFL max")
    axes[3, 1].set_xlabel("forcamento dinamico [m s$^{-2}$]")

    fig.tight_layout()
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


# ============================================================================
# 15. FIGURA COMPARATIVA FINAL
# ============================================================================

def nanmax_por_tempo(campo):
    campo = np.asarray(campo, dtype=float)
    saida = np.full(campo.shape[0], np.nan, dtype=float)

    for it in range(campo.shape[0]):
        fatia = campo[it]
        if not np.all(np.isnan(fatia)):
            saida[it] = np.nanmax(fatia)

    return saida


def salvar_figura_comparativa_final(
    resultados_por_caso,
    diagnosticos_por_caso,
    caminho,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    for caso, resultado in resultados_por_caso.items():
        frames = resultado["frames"]
        t_min = np.asarray(frames["t"], dtype=float) / 60.0

        w_t = np.nanmax(np.asarray(frames["w"], dtype=float), axis=(1, 2))
        qg_t = (
            np.nanmax(np.asarray(frames["qg"], dtype=float), axis=(1, 2))
            * 1000.0
        )

        diag = diagnosticos_por_caso[caso]
        f3_t = nanmax_por_tempo(diag["f3"])
        lpi_t = nanmax_por_tempo(diag["lpi_star"])

        axes[0, 0].plot(t_min, w_t, label=caso)
        axes[0, 1].plot(t_min, qg_t, label=caso)
        axes[1, 0].plot(t_min, f3_t, label=caso)
        axes[1, 1].plot(t_min, lpi_t, label=caso)

    axes[0, 0].set_title("(a) Movimento ascendente maximo")
    axes[0, 0].set_ylabel("w max [m s$^{-1}$]")

    axes[0, 1].set_title("(b) Graupel maximo")
    axes[0, 1].set_ylabel("qg max [g kg$^{-1}$]")

    axes[1, 0].set_title("(c) McCaul F3")
    axes[1, 0].set_ylabel("F3")
    axes[1, 0].set_xlabel("tempo [min]")

    axes[1, 1].set_title("(d) LPI*")
    axes[1, 1].set_ylabel("LPI*")
    axes[1, 1].set_xlabel("tempo [min]")

    axes[0, 0].legend()

    fig.tight_layout()
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


# ============================================================================
# 16. VARREDURA PRELIMINAR
# ============================================================================

def modo_varredura(args):
    pasta = OUTPUT_BASE / "varredura_forcamento_dinamico"
    pasta.mkdir(parents=True, exist_ok=True)

    forcamentos = sorted(set(float(v) for v in args.forcamentos))

    if any(valor <= 0.0 for valor in forcamentos):
        raise ValueError("Todos os forcamentos devem ser positivos.")

    commit_inicial = obter_commit_git()
    resumos = []

    for forcamento in forcamentos:
        caso = "SCAN_D_" + rotulo_forcamento(forcamento)

        _, _, resumo = executar_caso(
            caso=caso,
            delta_t_ambiente_k=0.0,
            forc_dyn_amp_m_s2=forcamento,
            tempo_min=args.tempo_varredura,
            output_base=pasta,
        )

        resumos.append(resumo)

        if obter_commit_git() != commit_inicial:
            raise RuntimeError(
                "O commit Git mudou durante a varredura. "
                "Repita a bateria usando um unico commit."
            )

    tabela = pasta / "resumo_varredura_forcamento.csv"
    salvar_tabela_resumo(resumos, tabela)

    figura = pasta / "comparacao_varredura_forcamento.png"
    salvar_figura_varredura(resumos, figura)

    candidatos_d0 = [
        r["forc_dyn_amp_m_s2"]
        for r in resumos
        if r["candidato_D0"]
    ]

    salvar_json(
        pasta / "metadados_varredura.json",
        {
            "commit": commit_inicial,
            "forcamentos_testados_m_s2": forcamentos,
            "tempo_varredura_min": float(args.tempo_varredura),
            "delta_T_ambiente_K": 0.0,
            "perfil_ambiente": PERFIL_AMBIENTE_GRUPO2,
            "preservar_rh": False,
            "bolha_k": BOLHA_K_GRUPO2,
            "bolha_qv_kgkg": BOLHA_QV_GRUPO2_KGKG,
            "geometria_forcamento": {
                "x0_m": FORC_DYN_X0_M,
                "z0_m": FORC_DYN_Z0_M,
                "rx_m": FORC_DYN_RX_M,
                "rz_m": FORC_DYN_RZ_M,
                "inicio_s": FORC_DYN_INICIO_S,
                "duracao_s": FORC_DYN_DURACAO_S,
            },
            "criterios_candidato_D0": {
                "fase_mista_ativa": True,
                "graupel_minus15_ativo": True,
                "updraft_minus15_ativo": True,
                "CFL_adv_menor_igual_1": True,
                "CFL_diff_menor_igual_0p5": True,
                "F3_e_LPI_nao_usados_na_selecao": True,
            },
            "candidatos_D0_m_s2": candidatos_d0,
        },
    )

    print()
    print("=" * 78)
    print("VARREDURA DE FORCAMENTO DINAMICO CONCLUIDA")
    print("=" * 78)
    print(f"Tabela: {tabela}")
    print(f"Figura: {figura}")

    if candidatos_d0:
        print()
        print("Forcamentos que satisfizeram os criterios minimos para D0:")
        print(", ".join(f"{v:g} m/s2" for v in candidatos_d0))
        print()
        print(
            "Escolha um D0 robusto e depois um D1 > D0 que produza uma "
            "resposta dinamica claramente mais intensa, sem usar F3/LPI* "
            "como criterio de calibracao."
        )
    else:
        print()
        print("Nenhum forcamento satisfez os criterios minimos para D0.")
        print("Amplie ou refine a faixa e repita a varredura.")

    print()
    print("Depois execute, por exemplo:")
    print(
        "python experiments/group2_warming_lightning/experimento_grupo2.py "
        "final --d0 <D0> --d1 <D1>"
    )
    print("=" * 78)


# ============================================================================
# 17. COMPARACOES RELATIVAS
# ============================================================================

def adicionar_razoes_relativas(resumos):
    ctrl = next(r for r in resumos if r["caso"] == "CTRL")
    saida = []

    for resumo in resumos:
        novo = dict(resumo)

        novo["w_max_sobre_CTRL"] = razao_segura(
            resumo["w_max_m_s"],
            ctrl["w_max_m_s"],
        )

        novo["qg_minus15_sobre_CTRL"] = razao_segura(
            resumo["qg_minus15_max_kgkg"],
            ctrl["qg_minus15_max_kgkg"],
        )

        novo["F3_max_sobre_CTRL"] = razao_segura(
            resumo["f3_max"],
            ctrl["f3_max"],
        )

        novo["LPI_star_max_sobre_CTRL"] = razao_segura(
            resumo["lpi_star_max"],
            ctrl["lpi_star_max"],
        )

        novo["RH_superficie_sobre_CTRL"] = razao_segura(
            resumo["RH_superficie"],
            ctrl["RH_superficie"],
        )

        saida.append(novo)

    return saida


# ============================================================================
# 18. MATRIZ FINAL 2x2
# ============================================================================

def modo_final(args):
    if args.d0 <= 0.0:
        raise ValueError("D0 deve ser positivo.")

    if args.d1 <= args.d0:
        raise ValueError(
            "D1 deve ser maior que D0. "
            f"Recebido: D0={args.d0:g}, D1={args.d1:g} m/s2."
        )

    matriz = {
        "CTRL": {
            "delta_t_ambiente_k": 0.0,
            "forc_dyn_amp_m_s2": float(args.d0),
        },
        "WARM": {
            "delta_t_ambiente_k": DELTA_T_WARM_K,
            "forc_dyn_amp_m_s2": float(args.d0),
        },
        "DYN_PLUS": {
            "delta_t_ambiente_k": 0.0,
            "forc_dyn_amp_m_s2": float(args.d1),
        },
        "WARM_DYN_PLUS": {
            "delta_t_ambiente_k": DELTA_T_WARM_K,
            "forc_dyn_amp_m_s2": float(args.d1),
        },
    }

    commit_inicial = obter_commit_git()

    resultados_por_caso = {}
    diagnosticos_por_caso = {}
    resumos = []

    for caso, fatores in matriz.items():
        resultado, diagnosticos, resumo = executar_caso(
            caso=caso,
            delta_t_ambiente_k=fatores["delta_t_ambiente_k"],
            forc_dyn_amp_m_s2=fatores["forc_dyn_amp_m_s2"],
            tempo_min=args.tempo_final,
            output_base=OUTPUT_BASE,
        )

        resultados_por_caso[caso] = resultado
        diagnosticos_por_caso[caso] = diagnosticos
        resumos.append(resumo)

        if obter_commit_git() != commit_inicial:
            raise RuntimeError(
                "O commit Git mudou durante a bateria final. "
                "Os quatro casos devem usar um unico commit."
            )

    # ----------------------------------------------------------------------
    # Verificacoes da logica experimental.
    # ----------------------------------------------------------------------

    resumo_ctrl = next(r for r in resumos if r["caso"] == "CTRL")
    resumo_warm = next(r for r in resumos if r["caso"] == "WARM")
    resumo_dyn = next(r for r in resumos if r["caso"] == "DYN_PLUS")

    if not resumo_ctrl["candidato_D0"]:
        warnings.warn(
            "O CTRL final nao satisfez todos os criterios minimos definidos "
            "para D0. Revise a escolha antes da interpretacao cientifica.",
            RuntimeWarning,
        )

    # O WARM deve ficar relativamente mais seco quando qv e mantido fixo.
    if resumo_warm["RH_0_2km_media"] >= resumo_ctrl["RH_0_2km_media"]:
        warnings.warn(
            "A RH media de 0-2 km do WARM nao ficou menor que a do CTRL. "
            "Isso contradiz a hipotese experimental esperada e deve ser "
            "verificado antes da interpretacao.",
            RuntimeWarning,
        )

    # D1 deve produzir resposta dinamica maior que D0 no ambiente CTRL.
    if resumo_dyn["w_max_m_s"] <= resumo_ctrl["w_max_m_s"]:
        warnings.warn(
            "DYN_PLUS nao produziu w_max maior que CTRL. "
            "D1 pode nao estar representando um forcamento dinamico "
            "claramente mais intenso na resposta do modelo.",
            RuntimeWarning,
        )

    # qv dos ambientes CTRL/WARM deve ser o mesmo quando preservar_rh=False.
    qv_ctrl = np.asarray(resultados_por_caso["CTRL"]["qv_env_1d"], dtype=float)
    qv_warm = np.asarray(resultados_por_caso["WARM"]["qv_env_1d"], dtype=float)

    if not np.allclose(qv_ctrl, qv_warm, rtol=0.0, atol=1.0e-12):
        warnings.warn(
            "qv ambiental do WARM difere do CTRL, embora o Grupo 2 novo "
            "exija qv inicial preservado. Verifique o nucleo/configuracao.",
            RuntimeWarning,
        )

    resumos_relativos = adicionar_razoes_relativas(resumos)

    tabela = OUTPUT_BASE / "resumo_grupo2.csv"
    salvar_tabela_resumo(resumos_relativos, tabela)

    salvar_json(
        OUTPUT_BASE / "matriz_experimental.json",
        {
            "D0_m_s2": float(args.d0),
            "D1_m_s2": float(args.d1),
            "delta_T_WARM_K": float(DELTA_T_WARM_K),
            "perfil_ambiente": PERFIL_AMBIENTE_GRUPO2,
            "preservar_rh": False,
            "qv_inicial_preservado": True,
            "bolha_k": BOLHA_K_GRUPO2,
            "bolha_qv_kgkg": BOLHA_QV_GRUPO2_KGKG,
            "geometria_forcamento": {
                "x0_m": FORC_DYN_X0_M,
                "z0_m": FORC_DYN_Z0_M,
                "rx_m": FORC_DYN_RX_M,
                "rz_m": FORC_DYN_RZ_M,
                "inicio_s": FORC_DYN_INICIO_S,
                "duracao_s": FORC_DYN_DURACAO_S,
            },
            "commit": commit_inicial,
            "casos": matriz,
        },
    )

    figura = OUTPUT_BASE / "comparacao_grupo2.png"
    salvar_figura_comparativa_final(
        resultados_por_caso=resultados_por_caso,
        diagnosticos_por_caso=diagnosticos_por_caso,
        caminho=figura,
    )

    print()
    print("=" * 78)
    print("MATRIZ FINAL DO GRUPO 2 CONCLUIDA")
    print("=" * 78)
    print(f"Perfil ambiente = {PERFIL_AMBIENTE_GRUPO2}")
    print(f"D0 = {args.d0:.6f} m/s2")
    print(f"D1 = {args.d1:.6f} m/s2")
    print(f"Delta T WARM = {DELTA_T_WARM_K:.2f} K")
    print("qv WARM = qv CTRL; RH livre para diminuir")
    print(f"RH 0-2 km CTRL = {100.0 * resumo_ctrl['RH_0_2km_media']:.1f} %")
    print(f"RH 0-2 km WARM = {100.0 * resumo_warm['RH_0_2km_media']:.1f} %")
    print(f"Resumo CSV: {tabela}")
    print(f"Figura: {figura}")
    print(f"Commit: {commit_inicial}")
    print("=" * 78)


# ============================================================================
# 19. INTERFACE DE LINHA DE COMANDO
# ============================================================================

def construir_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Grupo 2: aquecimento com qv preservado x forcamento dinamico "
            "externo, com diagnosticos microfisicos e eletricos."
        )
    )

    subparsers = parser.add_subparsers(dest="modo", required=True)

    # Varredura.
    p_scan = subparsers.add_parser(
        "varredura",
        help=(
            "Testa amplitudes do forcamento dinamico no ambiente CTRL "
            "para orientar a escolha de D0 e D1."
        ),
    )

    p_scan.add_argument(
        "--forcamentos",
        type=float,
        nargs="+",
        default=[0.50, 0.55, 0.60],
        help=(
            "Amplitudes do forcamento mecanico [m s-2]. "
            "Padrao atual: 0.10 0.15 0.20."
        ),
    )

    p_scan.add_argument(
        "--tempo-varredura",
        type=float,
        default=TEMPO_PADRAO_MIN,
        help=(
            "Duracao de cada simulacao da varredura [min]. "
            f"Padrao: {TEMPO_PADRAO_MIN:g}."
        ),
    )

    # Matriz final.
    p_final = subparsers.add_parser(
        "final",
        help="Executa CTRL, WARM, DYN_PLUS e WARM_DYN_PLUS.",
    )

    p_final.add_argument(
        "--d0",
        type=float,
        required=True,
        help="Forcamento dinamico de referencia D0 [m s-2].",
    )

    p_final.add_argument(
        "--d1",
        type=float,
        required=True,
        help="Forcamento dinamico forte D1 [m s-2]. Deve ser maior que D0.",
    )

    p_final.add_argument(
        "--tempo-final",
        type=float,
        default=TEMPO_PADRAO_MIN,
        help=(
            "Duracao de cada experimento final [min]. "
            f"Padrao: {TEMPO_PADRAO_MIN:g}."
        ),
    )

    return parser


# ============================================================================
# 20. MAIN
# ============================================================================

def main():
    parser = construir_parser()
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Raiz do repositorio: {ROOT}")
    print(f"Saidas do Grupo 2:   {OUTPUT_BASE}")
    print(f"Commit atual:        {obter_commit_git()}")
    print(f"Perfil ambiente:     {PERFIL_AMBIENTE_GRUPO2}")
    print("Grupo 2 novo: qv preservado, RH nao preservada, sem bolha termica.")

    if args.modo == "varredura":
        modo_varredura(args)
    elif args.modo == "final":
        modo_final(args)
    else:
        raise ValueError(f"Modo desconhecido: {args.modo}")


if __name__ == "__main__":
    main()
