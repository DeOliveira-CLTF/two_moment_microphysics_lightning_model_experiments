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

MATRIZ FINAL DECOMPOSTA
-----------------------

A matriz final completa separa tres componentes:

1. aquecimento ambiental;
2. intensidade do forcamento dinamico;
3. tratamento da umidade no ambiente aquecido.

Os seis casos unicos sao:

    CTRL
        DeltaT = 0 K, D0, ambiente de referencia

    DYN_PLUS
        DeltaT = 0 K, D1, ambiente de referencia

    WARM_QV
        DeltaT = +4 K, D0, qv inicial fixo e RH variavel

    WARM_QV_DYN_PLUS
        DeltaT = +4 K, D1, qv inicial fixo e RH variavel

    WARM_RH
        DeltaT = +4 K, D0, RH inicial fixa e qv ajustado

    WARM_RH_DYN_PLUS
        DeltaT = +4 K, D1, RH inicial fixa e qv ajustado

CTRL e DYN_PLUS nao sao duplicados entre os regimes de umidade porque, sem
aquecimento, os dois tratamentos produzem o mesmo ambiente inicial.

A execucao final gera automaticamente uma tabela CSV do desenho experimental
e uma tabela CSV completa com fatores e diagnosticos de cada subexperimento.

FLUXO DE TRABALHO
-----------------

ETAPA 1 - varredura preliminar do forcamento dinamico

A varredura pode ser feita no CTRL, no WARM ou nos dois ambientes:

    python experiments/group2_warming_lightning/experimento_grupo2.py \
        varredura --ambiente ambos --umidade qv_fixo

Por padrao sao testadas amplitudes:

    0.50, 0.55, 0.60, 0.65, 0.70, 0.75 e 0.80 m s-2

No CTRL:
    DeltaT = 0 K

No WARM:
    DeltaT = +4 K

A varredura serve para estimar separadamente:

    Dcrit_CTRL  = menor D que produz conveccao profunda adequada no CTRL
    Dcrit_WARM  = menor D que produz conveccao profunda adequada no WARM

Os criterios de adequacao sao:

- fase mista ativa;
- graupel em torno de -15 C;
- movimento ascendente em -15 C;
- estabilidade numerica segundo CFL.

A escolha de D0 e D1 NAO deve ser feita com base no maior F3 ou LPI*. Esses
proxies sao resultados do experimento, nao criterios para calibrar o forcamento.

ETAPA 2 - escolha de D0 e D1

Escolher:

    D0 = forcamento de referencia que sustenta conveccao no CTRL;
    D1 = forcamento mais intenso, preferencialmente suficiente para superar
         o limiar de iniciacao do WARM.

Se houver uma faixa em que:

    Dcrit_CTRL <= D0 < Dcrit_WARM <= D1

o experimento evidencia diretamente que o ambiente aquecido exige um
forcamento mais forte para desenvolver conveccao profunda.

A geometria e a duracao do forcamento permanecem identicas; apenas a
amplitude muda.

ETAPA 3 - matriz final decomposta

Por padrao o modo final executa os dois tratamentos de umidade:

    python experimento_grupo2_reformulado.py final --d0 0.55 --d1 0.65

Tambem e possivel executar apenas uma familia:

    --umidade qv_fixo
    --umidade rh_fixa

ou manter o padrao:

    --umidade ambas

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
NZ = 151
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

# Raiz dos diretorios de saida.
OUTPUT_ROOT = ROOT / "outputs" / "group2"

# Regimes de umidade:
# qv_fixo -> qv do CTRL e preservado; RH varia no WARM.
# rh_fixa -> RH do CTRL e preservada; qv aumenta no WARM.
MODO_UMIDADE_PADRAO = "qv_fixo"
MODOS_UMIDADE_VALIDOS = ("qv_fixo", "rh_fixa")

# Ambientes disponiveis na varredura.
AMBIENTES_VARREDURA_VALIDOS = ("ctrl", "warm", "ambos")

# Modos de umidade aceitos na matriz final.
MODOS_UMIDADE_FINAL_VALIDOS = ("ambas", "qv_fixo", "rh_fixa")
MODO_UMIDADE_FINAL_PADRAO = "ambas"


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


def preservar_rh_do_modo(modo_umidade):
    if modo_umidade == "qv_fixo":
        return False
    if modo_umidade == "rh_fixa":
        return True
    raise ValueError(
        f"Modo de umidade invalido: {modo_umidade!r}. "
        f"Use um de: {', '.join(MODOS_UMIDADE_VALIDOS)}."
    )


def rotulo_modo_umidade(modo_umidade):
    if modo_umidade == "qv_fixo":
        return "qv fixo; RH variavel"
    if modo_umidade == "rh_fixa":
        return "RH fixa; qv ajustado com o aquecimento"
    raise ValueError(f"Modo de umidade invalido: {modo_umidade!r}")


def obter_output_base(modo_umidade):
    if modo_umidade == "qv_fixo":
        return OUTPUT_ROOT / "qv_fixo_rh_variavel"
    if modo_umidade == "rh_fixa":
        return OUTPUT_ROOT / "rh_fixa"
    raise ValueError(f"Modo de umidade invalido: {modo_umidade!r}")


def ambientes_da_varredura(opcao):
    """Retorna os ambientes que serao executados na varredura."""

    if opcao == "ctrl":
        return ("CTRL",)

    if opcao == "warm":
        return ("WARM",)

    if opcao == "ambos":
        return ("CTRL", "WARM")

    raise ValueError(
        f"Ambiente de varredura invalido: {opcao!r}. "
        f"Use um de: {', '.join(AMBIENTES_VARREDURA_VALIDOS)}."
    )


def delta_t_do_ambiente(nome_ambiente):
    if nome_ambiente == "CTRL":
        return 0.0

    if nome_ambiente == "WARM":
        return float(DELTA_T_WARM_K)

    raise ValueError(f"Ambiente desconhecido: {nome_ambiente!r}")


def obter_output_final(modo_umidade_final):
    """Retorna uma pasta exclusiva para a matriz final solicitada."""

    if modo_umidade_final == "ambas":
        return OUTPUT_ROOT / "final_decomposto"

    if modo_umidade_final == "qv_fixo":
        return OUTPUT_ROOT / "final_qv_fixo_rh_variavel"

    if modo_umidade_final == "rh_fixa":
        return OUTPUT_ROOT / "final_rh_fixa"

    raise ValueError(
        f"Modo final de umidade invalido: {modo_umidade_final!r}"
    )


# ============================================================================
# 6. CONFIGURACAO DE UM CASO
# ============================================================================

def criar_configuracao(
    caso,
    delta_t_ambiente_k,
    forc_dyn_amp_m_s2,
    tempo_min,
    modo_umidade,
):
    """
    Constroi um caso do novo Grupo 2.

    Entre os quatro casos finais mudam apenas:

        delta_t_ambiente_k
        forc_dyn_amp_m_s2

    O perfil ambiental permanece explicitamente fixo em "referencia".
    Todo o restante permanece fixo.
    """

    preservar_rh = preservar_rh_do_modo(modo_umidade)

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

        # Regime de umidade escolhido na linha de comando.
        preservar_rh=preservar_rh,

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
    modo_umidade,
):
    print()
    print("=" * 78)
    print(f"INICIANDO CASO: {caso}")
    preservar_rh = preservar_rh_do_modo(modo_umidade)

    print(f"Perfil ambiente   = {PERFIL_AMBIENTE_GRUPO2}")
    print(f"Regime de umidade = {rotulo_modo_umidade(modo_umidade)}")
    print(f"Delta T ambiente  = {delta_t_ambiente_k:.3f} K")
    print(f"Forcamento dyn     = {forc_dyn_amp_m_s2:.6f} m/s2")
    print("bolha termica     = desligada")
    print("=" * 78)

    config = criar_configuracao(
        caso=caso,
        delta_t_ambiente_k=delta_t_ambiente_k,
        forc_dyn_amp_m_s2=forc_dyn_amp_m_s2,
        tempo_min=tempo_min,
        modo_umidade=modo_umidade,
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
        "modo_umidade": modo_umidade,
        "descricao_umidade": rotulo_modo_umidade(modo_umidade),
        "delta_t_ambiente_k": float(delta_t_ambiente_k),
        "preservar_rh": bool(preservar_rh),
        "qv_inicial_preservado": bool(not preservar_rh),
        "rh_inicial_preservada": bool(preservar_rh),
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


def salvar_tabela_desenho_experimental(matriz, caminho):
    """Salva uma tabela enxuta com os fatores de cada subexperimento."""

    linhas = []

    for ordem, (caso, fatores) in enumerate(matriz.items(), start=1):
        preservar_rh = preservar_rh_do_modo(fatores["modo_umidade"])
        referencia = fatores["regime_umidade"] == "referencia"

        linhas.append(
            {
                "ordem": ordem,
                "caso": caso,
                "delta_T_K": float(fatores["delta_t_ambiente_k"]),
                "nivel_forcamento": fatores["nivel_forcamento"],
                "forcamento_m_s2": float(fatores["forc_dyn_amp_m_s2"]),
                "regime_umidade": fatores["regime_umidade"],
                "modo_umidade_codigo": (
                    "nao_aplicavel" if referencia else fatores["modo_umidade"]
                ),
                "preservar_RH": None if referencia else bool(preservar_rh),
                "qv_inicial_preservado": (
                    None if referencia else bool(not preservar_rh)
                ),
                "componente_aquecimento": bool(
                    fatores["delta_t_ambiente_k"] > 0.0
                ),
                "componente_dinamica_intensificada": bool(
                    fatores["nivel_forcamento"] == "D1"
                ),
                "descricao": fatores["descricao"],
            }
        )

    salvar_tabela_resumo(linhas, caminho)


# ============================================================================
# 14. FIGURA DA VARREDURA DE FORCAMENTO
# ============================================================================

def salvar_figura_varredura(resumos, caminho, titulo=None):
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

    if titulo:
        fig.suptitle(titulo, fontsize=12, y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    else:
        fig.tight_layout()

    fig.savefig(caminho, dpi=180)
    plt.close(fig)



def salvar_figura_varredura_ctrl_warm(resumos_ctrl, resumos_warm, caminho):
    """
    Compara CTRL e WARM em funcao da amplitude do forcamento.

    F3 e LPI* aparecem apenas como diagnosticos; nao sao usados para
    determinar Dcrit.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conjuntos = {
        "CTRL": sorted(
            resumos_ctrl,
            key=lambda r: r["forc_dyn_amp_m_s2"],
        ),
        "WARM": sorted(
            resumos_warm,
            key=lambda r: r["forc_dyn_amp_m_s2"],
        ),
    }

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11, 10),
        sharex=True,
    )

    for ambiente, resumos in conjuntos.items():
        if not resumos:
            continue

        forc = np.asarray(
            [r["forc_dyn_amp_m_s2"] for r in resumos],
            dtype=float,
        )

        wmax = np.asarray(
            [r["w_max_m_s"] for r in resumos],
            dtype=float,
        )

        topo = np.asarray(
            [r["topo_nuvem_m"] / 1000.0 for r in resumos],
            dtype=float,
        )

        w15 = np.asarray(
            [r["w_minus15_max_m_s"] for r in resumos],
            dtype=float,
        )

        qg15 = np.asarray(
            [r["qg_minus15_max_kgkg"] * 1000.0 for r in resumos],
            dtype=float,
        )

        f3 = np.asarray(
            [r["f3_max"] for r in resumos],
            dtype=float,
        )

        lpi = np.asarray(
            [r["lpi_star_max"] for r in resumos],
            dtype=float,
        )

        axes[0, 0].plot(forc, wmax, marker="o", label=ambiente)
        axes[0, 1].plot(forc, topo, marker="o", label=ambiente)
        axes[1, 0].plot(forc, w15, marker="o", label=ambiente)
        axes[1, 1].plot(forc, qg15, marker="o", label=ambiente)
        axes[2, 0].plot(forc, f3, marker="o", label=ambiente)
        axes[2, 1].plot(forc, lpi, marker="o", label=ambiente)

    axes[0, 0].set_title("(a) Movimento ascendente maximo")
    axes[0, 0].set_ylabel("w max [m s$^{-1}$]")

    axes[0, 1].set_title("(b) Topo maximo da nuvem")
    axes[0, 1].set_ylabel("altura [km]")

    axes[1, 0].set_title("(c) Movimento ascendente em -15 °C")
    axes[1, 0].set_ylabel("w max [m s$^{-1}$]")

    axes[1, 1].set_title("(d) Graupel em -15 °C")
    axes[1, 1].set_ylabel("qg max [g kg$^{-1}$]")

    axes[2, 0].set_title("(e) McCaul F3")
    axes[2, 0].set_ylabel("F3 max")
    axes[2, 0].set_xlabel("forcamento dinamico [m s$^{-2}$]")

    axes[2, 1].set_title("(f) Lightning Potential Index")
    axes[2, 1].set_ylabel("LPI* max")
    axes[2, 1].set_xlabel("forcamento dinamico [m s$^{-2}$]")

    axes[0, 0].legend()

    for ax in axes.flat:
        ax.grid(linestyle=":", alpha=0.35)

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
    """
    Executa a varredura de D no CTRL, no WARM ou nos dois ambientes.

    Os resultados sao separados por ambiente:

        .../varredura_forcamento_dinamico/CTRL/
        .../varredura_forcamento_dinamico/WARM/

    Portanto os casos de mesma amplitude nunca se sobrescrevem.
    """

    output_base = obter_output_base(args.umidade)
    pasta_raiz = output_base / "varredura_forcamento_dinamico"
    pasta_raiz.mkdir(parents=True, exist_ok=True)

    forcamentos = sorted(
        set(float(v) for v in args.forcamentos)
    )

    if any(valor <= 0.0 for valor in forcamentos):
        raise ValueError("Todos os forcamentos devem ser positivos.")

    ambientes = ambientes_da_varredura(args.ambiente)
    commit_inicial = obter_commit_git()

    resumos_por_ambiente = {
        "CTRL": [],
        "WARM": [],
    }

    falhas = []

    for ambiente in ambientes:
        delta_t = delta_t_do_ambiente(ambiente)
        pasta_ambiente = pasta_raiz / ambiente
        pasta_ambiente.mkdir(parents=True, exist_ok=True)

        print()
        print("#" * 78)
        print(f"VARREDURA DO AMBIENTE {ambiente}")
        print(f"Delta T = {delta_t:.2f} K")
        print(f"Umidade = {rotulo_modo_umidade(args.umidade)}")
        print("#" * 78)

        for forcamento in forcamentos:
            caso = (
                f"SCAN_{ambiente}_D_"
                + rotulo_forcamento(forcamento)
            )

            try:
                _, _, resumo = executar_caso(
                    caso=caso,
                    delta_t_ambiente_k=delta_t,
                    forc_dyn_amp_m_s2=forcamento,
                    tempo_min=args.tempo_varredura,
                    output_base=pasta_ambiente,
                    modo_umidade=args.umidade,
                )

                resumo["ambiente_varredura"] = ambiente
                resumos_por_ambiente[ambiente].append(resumo)

            except RuntimeError as exc:
                mensagem = str(exc)

                # Em uma varredura, uma amplitude excessiva pode violar CFL.
                # Registramos essa amplitude e seguimos para as demais.
                if "CFL" not in mensagem.upper():
                    raise

                falha = {
                    "ambiente": ambiente,
                    "forc_dyn_amp_m_s2": float(forcamento),
                    "erro": mensagem,
                }
                falhas.append(falha)

                print()
                print("!" * 78)
                print(
                    f"CASO {caso} INTERROMPIDO POR CFL; "
                    "A VARREDURA CONTINUARA."
                )
                print(mensagem)
                print("!" * 78)

            if obter_commit_git() != commit_inicial:
                raise RuntimeError(
                    "O commit Git mudou durante a varredura. "
                    "Repita a bateria usando um unico commit."
                )

        # Produtos individuais de cada ambiente.
        resumos_amb = resumos_por_ambiente[ambiente]

        if resumos_amb:
            tabela_amb = (
                pasta_raiz
                / f"resumo_varredura_{ambiente}.csv"
            )
            salvar_tabela_resumo(
                resumos_amb,
                tabela_amb,
            )

            figura_amb = (
                pasta_raiz
                / f"comparacao_varredura_{ambiente}.png"
            )
            salvar_figura_varredura(
                resumos_amb,
                figura_amb,
                titulo=(
                    f"Varredura {ambiente} - "
                    f"{rotulo_modo_umidade(args.umidade)}"
                ),
            )

    # ----------------------------------------------------------------------
    # Estimativa dos limiares dinamicos.
    # ----------------------------------------------------------------------

    candidatos_ctrl = [
        r["forc_dyn_amp_m_s2"]
        for r in resumos_por_ambiente["CTRL"]
        if r["candidato_D0"]
    ]

    candidatos_warm = [
        r["forc_dyn_amp_m_s2"]
        for r in resumos_por_ambiente["WARM"]
        if r["candidato_D0"]
    ]

    dcrit_ctrl = (
        min(candidatos_ctrl)
        if candidatos_ctrl
        else None
    )

    dcrit_warm = (
        min(candidatos_warm)
        if candidatos_warm
        else None
    )

    # Figura comparativa se os dois ambientes foram executados.
    figura_comparativa = None

    if (
        resumos_por_ambiente["CTRL"]
        and resumos_por_ambiente["WARM"]
    ):
        figura_comparativa = (
            pasta_raiz
            / "comparacao_varredura_CTRL_WARM.png"
        )

        salvar_figura_varredura_ctrl_warm(
            resumos_ctrl=resumos_por_ambiente["CTRL"],
            resumos_warm=resumos_por_ambiente["WARM"],
            caminho=figura_comparativa,
        )

    # Tabela combinada.
    todos_resumos = (
        resumos_por_ambiente["CTRL"]
        + resumos_por_ambiente["WARM"]
    )

    if todos_resumos:
        salvar_tabela_resumo(
            todos_resumos,
            pasta_raiz / "resumo_varredura_CTRL_WARM.csv",
        )

    metadados = {
        "commit": commit_inicial,
        "ambientes_executados": list(ambientes),
        "forcamentos_testados_m_s2": forcamentos,
        "tempo_varredura_min": float(args.tempo_varredura),
        "delta_T_CTRL_K": 0.0,
        "delta_T_WARM_K": float(DELTA_T_WARM_K),
        "perfil_ambiente": PERFIL_AMBIENTE_GRUPO2,
        "modo_umidade": args.umidade,
        "descricao_umidade": rotulo_modo_umidade(args.umidade),
        "preservar_rh": preservar_rh_do_modo(args.umidade),
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
        "criterios_limiar_convectivo": {
            "fase_mista_ativa": True,
            "graupel_minus15_ativo": True,
            "updraft_minus15_ativo": True,
            "CFL_adv_menor_igual_1": True,
            "CFL_diff_menor_igual_0p5": True,
            "F3_e_LPI_nao_usados_na_selecao": True,
        },
        "candidatos_CTRL_m_s2": candidatos_ctrl,
        "candidatos_WARM_m_s2": candidatos_warm,
        "Dcrit_CTRL_aprox_m_s2": dcrit_ctrl,
        "Dcrit_WARM_aprox_m_s2": dcrit_warm,
        "falhas_CFL": falhas,
    }

    salvar_json(
        pasta_raiz / "metadados_varredura.json",
        metadados,
    )

    # ----------------------------------------------------------------------
    # Relatorio.
    # ----------------------------------------------------------------------

    print()
    print("=" * 78)
    print("VARREDURA DE FORCAMENTO DINAMICO CONCLUIDA")
    print("=" * 78)
    print(f"Diretorio: {pasta_raiz}")

    if dcrit_ctrl is not None:
        print(
            f"Dcrit CTRL aproximado = {dcrit_ctrl:g} m/s2"
        )
    elif "CTRL" in ambientes:
        print(
            "Dcrit CTRL nao foi encontrado na faixa testada."
        )

    if dcrit_warm is not None:
        print(
            f"Dcrit WARM aproximado = {dcrit_warm:g} m/s2"
        )
    elif "WARM" in ambientes:
        print(
            "Dcrit WARM nao foi encontrado na faixa testada."
        )

    if (
        dcrit_ctrl is not None
        and dcrit_warm is not None
    ):
        if dcrit_warm > dcrit_ctrl:
            print()
            print(
                "O WARM exigiu forcamento maior que o CTRL "
                "segundo os criterios dinamicos/microfisicos."
            )
            print(
                "Isto e consistente com a hipotese de maior "
                "limiar de iniciacao no ambiente aquecido."
            )
        elif dcrit_warm == dcrit_ctrl:
            print()
            print(
                "CTRL e WARM apresentaram o mesmo limiar na "
                "resolucao da varredura. Refine a faixa se necessario."
            )
        else:
            print()
            print(
                "O WARM apresentou limiar menor que o CTRL. "
                "A hipotese de maior limiar no WARM nao e suportada "
                "por esta varredura."
            )

    if figura_comparativa is not None:
        print(
            f"Figura CTRL x WARM: {figura_comparativa}"
        )

    if falhas:
        print()
        print(
            f"{len(falhas)} caso(s) foram interrompidos por CFL "
            "e registrados em metadados_varredura.json."
        )

    print()
    print("Depois execute a matriz final, por exemplo:")
    print(
        "python experiments/group2_warming_lightning/"
        "experimento_grupo2.py final "
        "--d0 <D0> --d1 <D1> "
        f"--umidade {args.umidade}"
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

def construir_matriz_final(d0, d1, modo_umidade_final):
    """Constroi a matriz final sem duplicar CTRL e DYN_PLUS."""

    matriz = {
        "CTRL": {
            "delta_t_ambiente_k": 0.0,
            "forc_dyn_amp_m_s2": float(d0),
            "nivel_forcamento": "D0",
            "modo_umidade": "qv_fixo",
            "regime_umidade": "referencia",
            "descricao": "Ambiente de referencia com forcamento D0.",
        },
        "DYN_PLUS": {
            "delta_t_ambiente_k": 0.0,
            "forc_dyn_amp_m_s2": float(d1),
            "nivel_forcamento": "D1",
            "modo_umidade": "qv_fixo",
            "regime_umidade": "referencia",
            "descricao": "Ambiente de referencia com forcamento D1.",
        },
    }

    if modo_umidade_final in {"ambas", "qv_fixo"}:
        matriz.update(
            {
                "WARM_QV": {
                    "delta_t_ambiente_k": float(DELTA_T_WARM_K),
                    "forc_dyn_amp_m_s2": float(d0),
                    "nivel_forcamento": "D0",
                    "modo_umidade": "qv_fixo",
                    "regime_umidade": "qv fixo; RH variavel",
                    "descricao": (
                        "Aquecimento com qv inicial preservado e RH livre "
                        "para diminuir, sob forcamento D0."
                    ),
                },
                "WARM_QV_DYN_PLUS": {
                    "delta_t_ambiente_k": float(DELTA_T_WARM_K),
                    "forc_dyn_amp_m_s2": float(d1),
                    "nivel_forcamento": "D1",
                    "modo_umidade": "qv_fixo",
                    "regime_umidade": "qv fixo; RH variavel",
                    "descricao": (
                        "Aquecimento com qv inicial preservado e RH livre "
                        "para diminuir, sob forcamento D1."
                    ),
                },
            }
        )

    if modo_umidade_final in {"ambas", "rh_fixa"}:
        matriz.update(
            {
                "WARM_RH": {
                    "delta_t_ambiente_k": float(DELTA_T_WARM_K),
                    "forc_dyn_amp_m_s2": float(d0),
                    "nivel_forcamento": "D0",
                    "modo_umidade": "rh_fixa",
                    "regime_umidade": "RH fixa; qv ajustado",
                    "descricao": (
                        "Aquecimento com RH inicial preservada e qv "
                        "ajustado, sob forcamento D0."
                    ),
                },
                "WARM_RH_DYN_PLUS": {
                    "delta_t_ambiente_k": float(DELTA_T_WARM_K),
                    "forc_dyn_amp_m_s2": float(d1),
                    "nivel_forcamento": "D1",
                    "modo_umidade": "rh_fixa",
                    "regime_umidade": "RH fixa; qv ajustado",
                    "descricao": (
                        "Aquecimento com RH inicial preservada e qv "
                        "ajustado, sob forcamento D1."
                    ),
                },
            }
        )

    return matriz


def modo_final(args):
    """Executa a matriz final decomposta e gera CSVs prontos."""

    if args.d0 <= 0.0:
        raise ValueError("D0 deve ser positivo.")

    if args.d1 <= args.d0:
        raise ValueError(
            "D1 deve ser maior que D0. "
            f"Recebido: D0={args.d0:g}, D1={args.d1:g} m/s2."
        )

    output_base = obter_output_final(args.umidade)
    output_base.mkdir(parents=True, exist_ok=True)

    matriz = construir_matriz_final(
        d0=args.d0,
        d1=args.d1,
        modo_umidade_final=args.umidade,
    )

    tabela_desenho = output_base / "tabela_desenho_experimental_grupo2.csv"
    salvar_tabela_desenho_experimental(matriz, tabela_desenho)

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
            output_base=output_base,
            modo_umidade=fatores["modo_umidade"],
        )

        resumo["nivel_forcamento"] = fatores["nivel_forcamento"]
        resumo["regime_umidade_experimental"] = fatores["regime_umidade"]
        resumo["descricao_subexperimento"] = fatores["descricao"]

        resultados_por_caso[caso] = resultado
        diagnosticos_por_caso[caso] = diagnosticos
        resumos.append(resumo)

        if obter_commit_git() != commit_inicial:
            raise RuntimeError(
                "O commit Git mudou durante a bateria final. "
                "Todos os subexperimentos devem usar um unico commit."
            )

    resumo_ctrl = next(r for r in resumos if r["caso"] == "CTRL")
    resumo_dyn = next(r for r in resumos if r["caso"] == "DYN_PLUS")

    if not resumo_ctrl["candidato_D0"]:
        warnings.warn(
            "O CTRL final nao satisfez os criterios minimos para D0.",
            RuntimeWarning,
        )

    if resumo_dyn["w_max_m_s"] <= resumo_ctrl["w_max_m_s"]:
        warnings.warn(
            "DYN_PLUS nao produziu w_max maior que CTRL.",
            RuntimeWarning,
        )

    qv_ctrl = np.asarray(
        resultados_por_caso["CTRL"]["qv_env_1d"], dtype=float
    )
    rh_ctrl = np.asarray(
        resultados_por_caso["CTRL"]["rh_env_1d"], dtype=float
    )

    if "WARM_QV" in resultados_por_caso:
        resumo_warm_qv = next(r for r in resumos if r["caso"] == "WARM_QV")
        qv_warm_qv = np.asarray(
            resultados_por_caso["WARM_QV"]["qv_env_1d"], dtype=float
        )

        if not np.allclose(qv_ctrl, qv_warm_qv, rtol=0.0, atol=1.0e-12):
            warnings.warn(
                "WARM_QV nao preservou o perfil inicial de qv do CTRL.",
                RuntimeWarning,
            )

        if resumo_warm_qv["RH_0_2km_media"] >= resumo_ctrl["RH_0_2km_media"]:
            warnings.warn(
                "WARM_QV nao apresentou reducao de RH em 0-2 km.",
                RuntimeWarning,
            )

    if "WARM_RH" in resultados_por_caso:
        resumo_warm_rh = next(r for r in resumos if r["caso"] == "WARM_RH")
        qv_warm_rh = np.asarray(
            resultados_por_caso["WARM_RH"]["qv_env_1d"], dtype=float
        )
        rh_warm_rh = np.asarray(
            resultados_por_caso["WARM_RH"]["rh_env_1d"], dtype=float
        )

        if not np.allclose(rh_ctrl, rh_warm_rh, rtol=0.0, atol=1.0e-12):
            warnings.warn(
                "WARM_RH nao preservou o perfil inicial de RH do CTRL.",
                RuntimeWarning,
            )

        if np.nanmean(qv_warm_rh) <= np.nanmean(qv_ctrl):
            warnings.warn(
                "WARM_RH nao apresentou aumento medio de qv.",
                RuntimeWarning,
            )

        if not np.isclose(
            resumo_warm_rh["RH_0_2km_media"],
            resumo_ctrl["RH_0_2km_media"],
            rtol=0.0,
            atol=1.0e-10,
        ):
            warnings.warn(
                "A RH media de 0-2 km de WARM_RH difere da do CTRL.",
                RuntimeWarning,
            )

    resumos_relativos = adicionar_razoes_relativas(resumos)
    tabela_resultados = output_base / "tabela_resultados_experimentos_grupo2.csv"
    salvar_tabela_resumo(resumos_relativos, tabela_resultados)

    salvar_json(
        output_base / "matriz_experimental.json",
        {
            "D0_m_s2": float(args.d0),
            "D1_m_s2": float(args.d1),
            "delta_T_WARM_K": float(DELTA_T_WARM_K),
            "perfil_ambiente": PERFIL_AMBIENTE_GRUPO2,
            "modo_umidade_final": args.umidade,
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

    figura = output_base / "comparacao_grupo2_final.png"
    salvar_figura_comparativa_final(
        resultados_por_caso=resultados_por_caso,
        diagnosticos_por_caso=diagnosticos_por_caso,
        caminho=figura,
    )

    print()
    print("=" * 78)
    print("MATRIZ FINAL DECOMPOSTA DO GRUPO 2 CONCLUIDA")
    print("=" * 78)
    print(f"Perfil ambiente = {PERFIL_AMBIENTE_GRUPO2}")
    print(f"D0 = {args.d0:.6f} m/s2")
    print(f"D1 = {args.d1:.6f} m/s2")
    print(f"Delta T WARM = {DELTA_T_WARM_K:.2f} K")
    print(f"Familias de umidade = {args.umidade}")
    print(f"Numero de subexperimentos = {len(matriz)}")
    print(f"Diretorio = {output_base}")
    print(f"Tabela desenho = {tabela_desenho}")
    print(f"Tabela resultados = {tabela_resultados}")
    print(f"Figura = {figura}")
    print(f"Commit = {commit_inicial}")
    print("=" * 78)


# ============================================================================
# 19. INTERFACE DE LINHA DE COMANDO
# ============================================================================

def construir_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Grupo 2: aquecimento x forcamento dinamico externo, "
            "com varredura CTRL/WARM e diagnosticos microfisicos e eletricos."
        )
    )

    subparsers = parser.add_subparsers(dest="modo", required=True)

    # Varredura.
    p_scan = subparsers.add_parser(
        "varredura",
        help=(
            "Testa amplitudes do forcamento no CTRL, no WARM ou em ambos "
            "para estimar os limiares dinamicos de iniciacao."
        ),
    )

    p_scan.add_argument(
        "--forcamentos",
        type=float,
        nargs="+",
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        help=(
            "Amplitudes do forcamento mecanico [m s-2]. "
            "Padrao atual: 0.50 0.55 0.60 0.65 0.70 0.75 0.80."
        ),
    )

    p_scan.add_argument(
        "--ambiente",
        choices=AMBIENTES_VARREDURA_VALIDOS,
        default="ambos",
        help=(
            "Ambiente usado na varredura: 'ctrl', 'warm' ou 'ambos'. "
            "Padrao: ambos."
        ),
    )

    p_scan.add_argument(
        "--umidade",
        choices=MODOS_UMIDADE_VALIDOS,
        default=MODO_UMIDADE_PADRAO,
        help=(
            "'qv_fixo' mantem qv e deixa RH variar; "
            "'rh_fixa' mantem RH e ajusta qv."
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
        help=(
            "Executa a matriz final decomposta. Por padrao inclui "
            "qv fixo/RH variavel e RH fixa."
        ),
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
        "--umidade",
        choices=MODOS_UMIDADE_FINAL_VALIDOS,
        default=MODO_UMIDADE_FINAL_PADRAO,
        help=(
            "Familias da matriz final: 'ambas', 'qv_fixo' ou 'rh_fixa'. "
            f"Padrao: {MODO_UMIDADE_FINAL_PADRAO}."
        ),
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

    print(f"Raiz do repositorio: {ROOT}")
    print(f"Commit atual:        {obter_commit_git()}")
    print(f"Perfil ambiente:     {PERFIL_AMBIENTE_GRUPO2}")
    print("Grupo 2: sem bolha termica; forcing dinamico prognostico.")

    if args.modo == "varredura":
        output_base = obter_output_base(args.umidade)
        output_base.mkdir(parents=True, exist_ok=True)
        print(f"Saidas do Grupo 2:   {output_base}")
        print(f"Regime de umidade:   {rotulo_modo_umidade(args.umidade)}")
        modo_varredura(args)

    elif args.modo == "final":
        output_base = obter_output_final(args.umidade)
        output_base.mkdir(parents=True, exist_ok=True)
        print(f"Saidas do Grupo 2:   {output_base}")
        print(f"Familias de umidade: {args.umidade}")
        modo_final(args)

    else:
        raise ValueError(f"Modo desconhecido: {args.modo}")


if __name__ == "__main__":
    main()
