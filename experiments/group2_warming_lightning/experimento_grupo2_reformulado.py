# -*- coding: utf-8 -*-
"""
Grupo 2 - Aquecimento, intensidade convectiva e diagnosticos eletricos
======================================================================

Este driver implementa o Grupo Experimental 2 do repositorio:

    DeOliveira-CLTF/two_moment_microphysics_lightning_model_experiments

A versao foi reformulada para que B0 e B1 NAO sejam assumidos previamente.

IDEIA CENTRAL
-------------
Antes da matriz cientifica 2x2, o script executa uma VARREDURA PRELIMINAR
da amplitude da bolha termica, sempre no ambiente de controle (Delta T = 0 K).

Essa varredura serve para identificar:

1. quais amplitudes geram conveccao profunda;
2. quais amplitudes alcancam uma regiao de fase mista ativa;
3. quais amplitudes produzem graupel em torno de -15 graus C;
4. quais amplitudes geram McCaul F3 e LPI* diferentes de zero;
5. quais amplitudes permanecem numericamente estaveis segundo CFL;
6. quais valores podem ser usados como B0;
7. quais valores maiores podem ser usados como B1.

B0
--
B0 e a amplitude da bolha termica usada nos casos CTRL e WARM.

Ela deve produzir uma tempestade de referencia suficientemente desenvolvida
para que a fase mista e os proxies eletricos possam ser analisados.

B1
--
B1 e uma amplitude MAIOR que B0, usada nos casos BUBBLE_PLUS e
WARM_BUBBLE_PLUS.

B1 nao e uma velocidade vertical imposta. O parametro bolha_k apenas altera
a perturbacao termica inicial. O movimento vertical w continua sendo
prognosticado pelo nucleo dinamico.

IMPORTANTE
----------
McCaul F1/F2/F3 e LPI* sao PROXIES de atividade eletrica. Este modelo nao
simula explicitamente flashes observados. Portanto, neste arquivo evitamos
dizer que uma simulacao "gera raios"; dizemos que ela produz um sinal
eletrico diagnostico mensuravel.

FLUXO DE TRABALHO
-----------------

ETAPA 1 - varredura preliminar

    python experiments/group2_warming_lightning/experimento_grupo2.py \
        varredura

Por padrao sao testadas as amplitudes:

    4, 6, 8, 10, 12 e 14 K

Esses valores sao apenas uma grade inicial de sensibilidade. Podem ser
substituidos pela linha de comando:

    python experiments/group2_warming_lightning/experimento_grupo2.py \
        varredura --amplitudes 5 6 7 8 9 10 11 12

A saida principal sera:

    outputs/group2/varredura_bolha/resumo_varredura_bolha.csv

O CSV inclui, entre outros:

    w_max
    topo da nuvem
    conteudo de fase mista
    graupel em -15 C
    w em -15 C
    F3 maximo
    LPI* maximo
    CFL maximo
    candidato_B0

ETAPA 2 - escolha cientifica de B0 e B1

A varredura NAO define automaticamente um "melhor" B0 ou B1.

O campo candidato_B0 apenas indica se uma amplitude satisfez criterios
minimos objetivos:

    - fase mista com coexistencia de liquido e gelo;
    - graupel em -15 C;
    - corrente ascendente em -15 C;
    - F3 > 0;
    - LPI* > 0;
    - CFL dentro dos limites.

A escolha final deve considerar tambem se a tempestade e suficientemente
robusta e nao apenas marginal.

Depois escolhe-se B1 > B0 como uma amplitude claramente mais intensa que B0,
mas ainda numericamente estavel.

ETAPA 3 - matriz cientifica final

Exemplo, caso a varredura leve o grupo a escolher B0 = 10 K e B1 = 18 K:

    python experiments/group2_warming_lightning/experimento_grupo2.py final --b0 10 --b1 18

A matriz final sera:

    CTRL              : Delta T = 0 K,  bolha = B0
    WARM              : Delta T = +4 K, bolha = B0
    BUBBLE_PLUS       : Delta T = 0 K,  bolha = B1
    WARM_BUBBLE_PLUS  : Delta T = +4 K, bolha = B1

Em WARM e WARM_BUBBLE_PLUS, qv ambiental e recalculado para preservar a
umidade relativa inicial.

A configuracao comum permanece fixa entre os quatro casos finais.
"""

# ============================================================================
# 1. IMPORTACOES DA BIBLIOTECA PADRAO
# ============================================================================

# argparse permite controlar o experimento pela linha de comando.
import argparse

# csv sera usado para salvar tabelas de resumo.
import csv

# json sera usado para registrar configuracoes e metadados.
import json

# subprocess permite registrar o commit Git utilizado.
import subprocess

# sys permite inserir a raiz do repositorio no caminho de importacao.
import sys

# warnings sera usado apenas para suprimir avisos esperados de fatias all-NaN.
import warnings

# asdict converte a dataclass do nucleo em um dicionario serializavel.
from dataclasses import asdict

# Path fornece manipulacao portavel de caminhos.
from pathlib import Path


# ============================================================================
# 2. LOCALIZACAO DA RAIZ DO REPOSITORIO
# ============================================================================

# O arquivo deve ficar em:
#
# experiments/group2_warming_lightning/experimento_grupo2.py
#
# parents[0] -> group2_warming_lightning
# parents[1] -> experiments
# parents[2] -> raiz do repositorio
ROOT = Path(__file__).resolve().parents[2]

# Garante que os pacotes locais possam ser importados.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# 3. IMPORTACOES CIENTIFICAS E DA API DO REPOSITORIO
# ============================================================================

# NumPy e usado para manipulacao dos campos e diagnosticos.
import numpy as np

# API comum do nucleo dinamico 2D.
from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d

# QMIN e o proprio limiar do esquema microfisico para considerar que existe
# uma quantidade de hidrometeoro numericamente relevante.
from microfisica.constantes import QMIN

# API oficial dos diagnosticos eletricos 2D.
from lightning import diagnosticar_relampagos_2d, resumir_diagnosticos_2d


# ============================================================================
# 4. CONFIGURACAO COMUM DO GRUPO 2
# ============================================================================

# Aquecimento aplicado aos experimentos WARM.
DELTA_T_WARM_K = 4.0

# Concentracao de ativacao de goticulas mantida fixa no Grupo 2.
NC_CONTROLE_KG1 = 2.0e8

# Grade horizontal e vertical.
NX = 90
NZ = 110

# Resolucao espacial.
DX_M = 100.0
DZ_M = 100.0

# Passo de tempo.
DT_S = 1

# Intervalo entre frames armazenados.
SALVAR_A_CADA_S = 300.0

# Duracao padrao.
#
# A varredura usa por padrao o mesmo tempo da matriz final. Isso evita
# classificar como "fraca" uma simulacao que simplesmente ainda nao teve tempo
# para desenvolver gelo/graupel e os proxies eletricos.
TEMPO_PADRAO_MIN = 40.0

# Limiar de w usado pelo proprio LPI* do repositorio.
W_LPI_THRESHOLD_M_S = 0.5

# Diretorio geral de saida do Grupo 2.
OUTPUT_BASE = ROOT / "outputs" / "group2"


# ============================================================================
# 5. FUNCOES AUXILIARES DE RASTREABILIDADE
# ============================================================================

def obter_commit_git():
    """
    Retorna o hash do commit Git atual.

    Se o comando Git nao estiver disponivel, a simulacao continua, mas o
    arquivo de metadados registra que o commit nao pode ser obtido.
    """

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
    """
    Reconstrui a linha de comando usada para iniciar o experimento.
    """

    return " ".join([sys.executable, *sys.argv])


def escrever_texto(caminho, texto):
    """
    Salva uma informacao textual simples em UTF-8.
    """

    caminho.write_text(str(texto) + "\n", encoding="utf-8")


def salvar_json(caminho, objeto):
    """
    Salva um dicionario em formato JSON legivel.
    """

    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(
            objeto,
            arquivo,
            indent=2,
            ensure_ascii=False,
        )


def rotulo_amplitude(amplitude_k):
    """
    Converte uma amplitude em um nome seguro para diretorio.

    Exemplo:
        8.0  -> 8K
        7.5  -> 7p5K
    """

    texto = f"{float(amplitude_k):g}"
    texto = texto.replace(".", "p")

    return f"{texto}K"


# ============================================================================
# 6. CONFIGURACAO DE UM CASO
# ============================================================================

def criar_configuracao(
    caso,
    bolha_k,
    delta_t_ambiente_k,
    tempo_min,
):
    """
    Constroi a configuracao de um caso.

    Entre os quatro casos finais, apenas dois fatores cientificos mudam:

        bolha_k
        delta_t_ambiente_k

    Todo o restante deve permanecer fixo.
    """

    config = ConfiguracaoDinamica2D(

        # Numero de pontos horizontais.
        nx=NX,

        # Numero de pontos verticais.
        nz=NZ,

        # Resolucao horizontal [m].
        dx=DX_M,

        # Resolucao vertical [m].
        dz=DZ_M,

        # Passo de tempo [s].
        dt=DT_S,

        # Tempo total da simulacao [s].
        tempo_total_s=float(tempo_min) * 60.0,

        # Frequencia de salvamento [s].
        salvar_a_cada_s=SALVAR_A_CADA_S,

        # Amplitude da perturbacao termica inicial [K].
        bolha_k=float(bolha_k),

        # Aquecimento uniforme da temperatura real do ambiente [K].
        delta_t_ambiente_k=float(delta_t_ambiente_k),

        # Recalcula qv para manter RH nos ambientes aquecidos.
        preservar_rh=True,

        # Mantem Nc fixo no Grupo 2.
        nc_ativacao_kg1=NC_CONTROLE_KG1,

        # Microfisica completa de dois momentos.
        microfisica="thompson",

        # Evaporacao de chuva ligada.
        evap_chuva=True,

        # Radiacao desligada.
        radiacao=False,

        # Ciclo diurno desligado.
        ciclo_diurno=False,

        # Identificador do caso.
        cenario=caso,

        # Aviso operacional de CFL.
        cfl_aviso=0.80,

        # Limite advectivo/sedimentacao.
        cfl_limite=1.00,

        # Interrompe se o limite configurado for violado.
        abortar_se_cfl_violar=True,
    )

    return config


# ============================================================================
# 7. FUNCOES NUMERICAS SEGURAS
# ============================================================================

def maximo_seguro(campo):
    """
    Retorna o maximo ignorando NaN.

    Se todo o campo for NaN, retorna NaN.
    """

    campo = np.asarray(campo, dtype=float)

    if campo.size == 0 or np.all(np.isnan(campo)):
        return np.nan

    return float(np.nanmax(campo))


def media_segura(campo):
    """
    Retorna a media ignorando NaN.

    Se todo o campo for NaN, retorna NaN.
    """

    campo = np.asarray(campo, dtype=float)

    if campo.size == 0 or np.all(np.isnan(campo)):
        return np.nan

    return float(np.nanmean(campo))


def razao_segura(numerador, denominador):
    """
    Calcula uma razao somente quando os dois valores permitem a operacao.
    """

    if not np.isfinite(numerador):
        return np.nan

    if not np.isfinite(denominador):
        return np.nan

    if denominador == 0.0:
        return np.nan

    return float(numerador / denominador)


# ============================================================================
# 8. SALVAMENTO COMPLETO DOS CAMPOS
# ============================================================================

def salvar_npz(
    resultado,
    diagnosticos,
    caminho,
):
    """
    Salva dinamica, microfisica e diagnosticos eletricos em um NPZ comprimido.
    """

    # Atalho para os frames temporais.
    frames = resultado["frames"]

    # Metadados e perfis ambientais.
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

    # Preserva todos os campos produzidos pelo nucleo.
    for nome, valores in frames.items():

        # O tempo ja foi salvo como t_s.
        if nome == "t":
            continue

        dados[nome] = np.asarray(valores)

    # Acrescenta os diagnosticos eletricos.
    for nome, valores in diagnosticos.items():

        # t_s e x_m ja existem no arquivo.
        if nome in {"t_s", "x_m"}:
            continue

        dados[f"lightning_{nome}"] = np.asarray(valores)

    # Salva o arquivo comprimido.
    np.savez_compressed(
        caminho,
        **dados,
    )


# ============================================================================
# 9. DIAGNOSTICOS DINAMICOS E MICROFISICOS
# ============================================================================

def diagnosticos_dinamicos_microfisicos(
    resultado,
    diagnosticos_eletricos,
):
    """
    Constroi metricas compactas para avaliar cada amplitude de bolha.

    Alem dos maximos tradicionais, esta funcao verifica explicitamente se
    existe coexistencia de fase liquida e congelada na camada de 0 a -20 C.

    Essa verificacao e importante para B0: nao basta a atmosfera conter as
    isotermas de 0 e -20 C; a nuvem precisa efetivamente ocupar essa camada
    com hidrometeoros relevantes.
    """

    # Atalho para os frames.
    frames = resultado["frames"]

    # Coordenada vertical.
    z_m = np.asarray(
        resultado["z_m"],
        dtype=float,
    )

    # Tempo.
    t_s = np.asarray(
        frames["t"],
        dtype=float,
    )

    # Velocidade vertical.
    w = np.asarray(
        frames["w"],
        dtype=float,
    )

    # Temperatura real.
    T = np.asarray(
        frames["T"],
        dtype=float,
    )

    # Hidrometeoros liquidos.
    qc = np.asarray(
        frames["qc"],
        dtype=float,
    )

    qr = np.asarray(
        frames["qr"],
        dtype=float,
    )

    # Hidrometeoros congelados.
    qi = np.asarray(
        frames["qi"],
        dtype=float,
    )

    qs = np.asarray(
        frames["qs"],
        dtype=float,
    )

    qg = np.asarray(
        frames["qg"],
        dtype=float,
    )

    # ------------------------------------------------------------------------
    # 9.1. Movimento ascendente maximo
    # ------------------------------------------------------------------------

    w_max = maximo_seguro(w)

    # w nao deve ser all-NaN em uma simulacao valida.
    indice_w = np.unravel_index(
        np.nanargmax(w),
        w.shape,
    )

    it_w, ix_w, iz_w = indice_w

    # Altura do maior movimento ascendente.
    z_w_max_m = float(
        z_m[iz_w]
    )

    # Tempo do maior movimento ascendente.
    tempo_w_max_s = float(
        t_s[it_w]
    )

    # ------------------------------------------------------------------------
    # 9.2. Conteudo total e topo da nuvem
    # ------------------------------------------------------------------------

    # Soma de todos os hidrometeoros.
    q_total = (
        qc
        + qr
        + qi
        + qs
        + qg
    )

    # Usa QMIN do proprio esquema microfisico.
    mascara_nuvem = q_total > QMIN

    if np.any(mascara_nuvem):

        # Identifica os niveis ocupados por nuvem em qualquer tempo/x.
        indices_z_nuvem = np.where(
            np.any(
                mascara_nuvem,
                axis=(0, 1),
            )
        )[0]

        # Maior nivel ocupado.
        topo_nuvem_m = float(
            z_m[indices_z_nuvem[-1]]
        )

    else:
        topo_nuvem_m = np.nan

    # ------------------------------------------------------------------------
    # 9.3. Camada termodinamica de fase mista
    # ------------------------------------------------------------------------

    # 273.15 K = 0 C
    # 253.15 K = -20 C
    mascara_termica_fase_mista = (
        (T <= 273.15)
        & (T >= 253.15)
    )

    # Agua liquida.
    q_liquido = qc + qr

    # Fase congelada.
    q_congelado = qi + qs + qg

    # Guarda somente valores localizados na camada de 0 a -20 C.
    q_liquido_fase_mista = np.where(
        mascara_termica_fase_mista,
        q_liquido,
        np.nan,
    )

    q_congelado_fase_mista = np.where(
        mascara_termica_fase_mista,
        q_congelado,
        np.nan,
    )

    qg_fase_mista = np.where(
        mascara_termica_fase_mista,
        qg,
        np.nan,
    )

    # Maximos dentro da camada.
    q_liquido_fase_mista_max = maximo_seguro(
        q_liquido_fase_mista
    )

    q_congelado_fase_mista_max = maximo_seguro(
        q_congelado_fase_mista
    )

    qg_fase_mista_max = maximo_seguro(
        qg_fase_mista
    )

    # Existe coexistencia local de liquido e congelado?
    coexistencia_fase_mista = (
        mascara_termica_fase_mista
        & (q_liquido > QMIN)
        & (q_congelado > QMIN)
    )

    fase_mista_ativa = bool(
        np.any(coexistencia_fase_mista)
    )

    # ------------------------------------------------------------------------
    # 9.4. Diagnosticos especificamente em -15 C
    # ------------------------------------------------------------------------

    # O modulo lightning ja interpola w e qg exatamente na isoterma de -15 C.
    w_minus15 = np.asarray(
        diagnosticos_eletricos["w_minus15_m_s"],
        dtype=float,
    )

    qg_minus15 = np.asarray(
        diagnosticos_eletricos["qg_minus15_kgkg"],
        dtype=float,
    )

    w_minus15_max = maximo_seguro(
        w_minus15
    )

    qg_minus15_max = maximo_seguro(
        qg_minus15
    )

    # Graupel numericamente presente em -15 C.
    graupel_minus15_ativo = bool(
        np.isfinite(qg_minus15_max)
        and qg_minus15_max > QMIN
    )

    # Para manter coerencia com LPI*, consideramos um updraft eletricamente
    # relevante quando ultrapassa 0.5 m/s.
    updraft_minus15_ativo = bool(
        np.isfinite(w_minus15_max)
        and w_minus15_max > W_LPI_THRESHOLD_M_S
    )

    # ------------------------------------------------------------------------
    # 9.5. Retorno dos diagnosticos
    # ------------------------------------------------------------------------

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

        "q_liquido_fase_mista_max_kgkg": q_liquido_fase_mista_max,
        "q_congelado_fase_mista_max_kgkg": q_congelado_fase_mista_max,
        "qg_fase_mista_max_kgkg": qg_fase_mista_max,

        "fase_mista_ativa": fase_mista_ativa,

        "w_minus15_max_m_s": w_minus15_max,
        "qg_minus15_max_kgkg": qg_minus15_max,

        "graupel_minus15_ativo": graupel_minus15_ativo,
        "updraft_minus15_ativo": updraft_minus15_ativo,
    }


# ============================================================================
# 10. CRITERIOS DE ADEQUACAO PARA B0
# ============================================================================

def avaliar_criterios_b0(
    resumo,
):
    """
    Avalia se um caso pode ser considerado candidato a B0.

    O objetivo NAO e escolher automaticamente o "melhor" B0.

    A funcao apenas verifica criterios minimos transparentes.
    """

    # CFL advectivo/sedimentacao.
    cfl_adv_ok = (
        np.isfinite(resumo["cfl_max_adv"])
        and resumo["cfl_max_adv"] <= 1.0
    )

    # CFL difusivo.
    cfl_diff_ok = (
        np.isfinite(resumo["cfl_max_diff"])
        and resumo["cfl_max_diff"] <= 0.5
    )

    # Ambos devem ser satisfeitos.
    cfl_ok = bool(
        cfl_adv_ok
        and cfl_diff_ok
    )

    # McCaul F3 deve ser mensuravel.
    f3_ativo = bool(
        np.isfinite(resumo["f3_max"])
        and resumo["f3_max"] > 0.0
    )

    # LPI* deve ser mensuravel.
    lpi_ativo = bool(
        np.isfinite(resumo["lpi_star_max"])
        and resumo["lpi_star_max"] > 0.0
    )

    # Conjunto de criterios para um B0 cientificamente utilizavel.
    candidato_b0 = bool(
        resumo["fase_mista_ativa"]
        and resumo["graupel_minus15_ativo"]
        and resumo["updraft_minus15_ativo"]
        and f3_ativo
        and lpi_ativo
        and cfl_ok
    )

    return {
        "cfl_adv_ok": cfl_adv_ok,
        "cfl_diff_ok": cfl_diff_ok,
        "cfl_ok": cfl_ok,
        "f3_ativo": f3_ativo,
        "lpi_ativo": lpi_ativo,
        "candidato_B0": candidato_b0,
    }


# ============================================================================
# 11. EXECUCAO DE UM UNICO CASO
# ============================================================================

def executar_caso(
    caso,
    bolha_k,
    delta_t_ambiente_k,
    tempo_min,
    output_base,
):
    """
    Executa um caso completo.

    Sequencia:
        configuracao
        -> dinamica + microfisica
        -> McCaul/LPI*
        -> resumo
        -> avaliacao dos criterios
        -> salvamento
    """

    print()
    print("=" * 78)
    print(f"INICIANDO CASO: {caso}")
    print(f"bolha_k = {bolha_k:.3f} K")
    print(
        "delta_t_ambiente_k = "
        f"{delta_t_ambiente_k:.3f} K"
    )
    print("=" * 78)

    # Cria a configuracao.
    config = criar_configuracao(
        caso=caso,
        bolha_k=bolha_k,
        delta_t_ambiente_k=delta_t_ambiente_k,
        tempo_min=tempo_min,
    )

    # Cria a pasta do caso.
    pasta_caso = output_base / caso

    pasta_caso.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Executa o modelo 2D acoplado.
    resultado = rodar_thompson_2d(
        config,
        verbose=True,
    )

    # Calcula os diagnosticos eletricos.
    diagnosticos = diagnosticar_relampagos_2d(
        resultado
    )

    # Resumo oficial do modulo lightning.
    resumo_lightning = resumir_diagnosticos_2d(
        diagnosticos
    )

    # Resumo dinamico/microfisico.
    resumo_dm = diagnosticos_dinamicos_microfisicos(
        resultado=resultado,
        diagnosticos_eletricos=diagnosticos,
    )

    # Monta o resumo inicial.
    resumo = {
        "caso": caso,
        "bolha_k": float(bolha_k),
        "delta_t_ambiente_k": float(delta_t_ambiente_k),
        "preservar_rh": True,
        "tempo_total_min": float(tempo_min),

        "cfl_max_adv": float(
            resultado["cfl_max_adv"]
        ),

        "cfl_max_diff": float(
            resultado["cfl_max_diff"]
        ),

        **resumo_dm,
        **resumo_lightning,
    }

    # Acrescenta os criterios de adequacao.
    resumo.update(
        avaliar_criterios_b0(
            resumo
        )
    )

    # ------------------------------------------------------------------------
    # Salvamento
    # ------------------------------------------------------------------------

    # Campos completos.
    salvar_npz(
        resultado=resultado,
        diagnosticos=diagnosticos,
        caminho=(
            pasta_caso
            / f"resultados_{caso}.npz"
        ),
    )

    # Configuracao completa.
    salvar_json(
        pasta_caso / "configuracao.json",
        asdict(config),
    )

    # Resumo.
    salvar_json(
        pasta_caso / "resumo.json",
        resumo,
    )

    # Comando executado.
    escrever_texto(
        pasta_caso / "comando.txt",
        comando_executado(),
    )

    # Commit.
    escrever_texto(
        pasta_caso / "commit.txt",
        obter_commit_git(),
    )

    # ------------------------------------------------------------------------
    # Mensagem de terminal
    # ------------------------------------------------------------------------

    print(f"Caso {caso} concluido.")

    print(
        "CFL max adv/sed = "
        f"{resumo['cfl_max_adv']:.4f}"
    )

    print(
        "CFL max diff    = "
        f"{resumo['cfl_max_diff']:.6f}"
    )

    print(
        "w max           = "
        f"{resumo['w_max_m_s']:.3f} m/s"
    )

    print(
        "w max em -15 C  = "
        f"{resumo['w_minus15_max_m_s']:.3f} m/s"
    )

    print(
        "qg max em -15 C = "
        f"{resumo['qg_minus15_max_kgkg']:.6e} kg/kg"
    )

    print(
        "F3 max          = "
        f"{resumo['f3_max']:.6g}"
    )

    print(
        "LPI* max        = "
        f"{resumo['lpi_star_max']:.6g}"
    )

    print(
        "fase mista      = "
        f"{resumo['fase_mista_ativa']}"
    )

    print(
        "candidato B0    = "
        f"{resumo['candidato_B0']}"
    )

    print(
        f"Saida           = {pasta_caso}"
    )

    return (
        resultado,
        diagnosticos,
        resumo,
    )


# ============================================================================
# 12. SALVAMENTO DE TABELAS
# ============================================================================

def salvar_tabela_resumo(
    resumos,
    caminho,
):
    """
    Salva uma lista de dicionarios em CSV.
    """

    if not resumos:
        return

    # Reune todas as colunas encontradas.
    colunas = []

    for resumo in resumos:

        for chave in resumo:

            if chave not in colunas:
                colunas.append(chave)

    # Escreve o arquivo.
    with caminho.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=colunas,
        )

        escritor.writeheader()

        escritor.writerows(
            resumos
        )


# ============================================================================
# 13. FIGURA DA VARREDURA DE BOLHA
# ============================================================================

def salvar_figura_varredura(
    resumos,
    caminho,
):
    """
    Gera uma figura diagnostica da varredura preliminar.

    A figura nao decide B0/B1. Ela apenas mostra como a resposta do modelo
    muda com a amplitude da bolha.
    """

    import matplotlib

    # Permite uso sem interface grafica.
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Ordena pela amplitude.
    resumos_ordenados = sorted(
        resumos,
        key=lambda r: r["bolha_k"],
    )

    # Eixo x.
    bolhas = np.asarray(
        [
            r["bolha_k"]
            for r in resumos_ordenados
        ],
        dtype=float,
    )

    # Series.
    wmax = np.asarray(
        [
            r["w_max_m_s"]
            for r in resumos_ordenados
        ],
        dtype=float,
    )

    qg15 = np.asarray(
        [
            r["qg_minus15_max_kgkg"] * 1000.0
            for r in resumos_ordenados
        ],
        dtype=float,
    )

    f3 = np.asarray(
        [
            r["f3_max"]
            for r in resumos_ordenados
        ],
        dtype=float,
    )

    lpi = np.asarray(
        [
            r["lpi_star_max"]
            for r in resumos_ordenados
        ],
        dtype=float,
    )

    qmix = np.asarray(
        [
            r["q_congelado_fase_mista_max_kgkg"] * 1000.0
            for r in resumos_ordenados
        ],
        dtype=float,
    )

    cfl = np.asarray(
        [
            r["cfl_max_adv"]
            for r in resumos_ordenados
        ],
        dtype=float,
    )

    # Cria 6 paineis.
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11, 11),
        sharex=True,
    )

    # w max.
    axes[0, 0].plot(
        bolhas,
        wmax,
        marker="o",
    )

    axes[0, 0].set_title(
        "(a) Movimento ascendente maximo"
    )

    axes[0, 0].set_ylabel(
        "w max [m s$^{-1}$]"
    )

    # Graupel em -15 C.
    axes[0, 1].plot(
        bolhas,
        qg15,
        marker="o",
    )

    axes[0, 1].set_title(
        "(b) Graupel em -15 °C"
    )

    axes[0, 1].set_ylabel(
        "qg max [g kg$^{-1}$]"
    )

    # Fase congelada na camada mista.
    axes[1, 0].plot(
        bolhas,
        qmix,
        marker="o",
    )

    axes[1, 0].set_title(
        "(c) Fase congelada em 0 a -20 °C"
    )

    axes[1, 0].set_ylabel(
        "q congelado max [g kg$^{-1}$]"
    )

    # F3.
    axes[1, 1].plot(
        bolhas,
        f3,
        marker="o",
    )

    axes[1, 1].set_title(
        "(d) McCaul F3"
    )

    axes[1, 1].set_ylabel(
        "F3 max"
    )

    # LPI*.
    axes[2, 0].plot(
        bolhas,
        lpi,
        marker="o",
    )

    axes[2, 0].set_title(
        "(e) Lightning Potential Index"
    )

    axes[2, 0].set_ylabel(
        "LPI* max"
    )

    axes[2, 0].set_xlabel(
        "amplitude da bolha [K]"
    )

    # CFL.
    axes[2, 1].plot(
        bolhas,
        cfl,
        marker="o",
    )

    # Linha do limite matematico.
    axes[2, 1].axhline(
        1.0,
        linestyle="--",
    )

    # Linha da margem operacional.
    axes[2, 1].axhline(
        0.8,
        linestyle=":",
    )

    axes[2, 1].set_title(
        "(f) CFL advectivo/sedimentacao"
    )

    axes[2, 1].set_ylabel(
        "CFL max"
    )

    axes[2, 1].set_xlabel(
        "amplitude da bolha [K]"
    )

    # Ajuste final.
    fig.tight_layout()

    # Salva a figura.
    fig.savefig(
        caminho,
        dpi=180,
    )

    plt.close(fig)


# ============================================================================
# 14. FIGURA COMPARATIVA DA MATRIZ FINAL
# ============================================================================

def nanmax_por_tempo(
    campo,
):
    """
    Calcula nanmax sobre todas as dimensoes exceto tempo.

    Evita transformar fatias completamente NaN em excecoes.
    """

    campo = np.asarray(
        campo,
        dtype=float,
    )

    # Numero de tempos.
    nt = campo.shape[0]

    # Vetor de saida.
    saida = np.full(
        nt,
        np.nan,
        dtype=float,
    )

    # Percorre cada tempo.
    for it in range(nt):

        fatia = campo[it]

        if not np.all(
            np.isnan(fatia)
        ):
            saida[it] = np.nanmax(
                fatia
            )

    return saida


def salvar_figura_comparativa_final(
    resultados_por_caso,
    diagnosticos_por_caso,
    caminho,
):
    """
    Compara a evolucao temporal dos quatro experimentos finais.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Grade 2x2.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
        sharex=True,
    )

    # Percorre os quatro casos.
    for caso, resultado in resultados_por_caso.items():

        # Frames.
        frames = resultado["frames"]

        # Tempo [min].
        t_min = (
            np.asarray(
                frames["t"],
                dtype=float,
            )
            / 60.0
        )

        # w max por tempo.
        w_t = np.nanmax(
            np.asarray(
                frames["w"],
                dtype=float,
            ),
            axis=(1, 2),
        )

        # qg max por tempo.
        qg_t = (
            np.nanmax(
                np.asarray(
                    frames["qg"],
                    dtype=float,
                ),
                axis=(1, 2),
            )
            * 1000.0
        )

        # Diagnosticos eletricos.
        diag = diagnosticos_por_caso[
            caso
        ]

        # F3 max por tempo.
        f3_t = nanmax_por_tempo(
            diag["f3"]
        )

        # LPI max por tempo.
        lpi_t = nanmax_por_tempo(
            diag["lpi_star"]
        )

        # Plota.
        axes[0, 0].plot(
            t_min,
            w_t,
            label=caso,
        )

        axes[0, 1].plot(
            t_min,
            qg_t,
            label=caso,
        )

        axes[1, 0].plot(
            t_min,
            f3_t,
            label=caso,
        )

        axes[1, 1].plot(
            t_min,
            lpi_t,
            label=caso,
        )

    # Rotulos.
    axes[0, 0].set_title(
        "(a) Movimento ascendente maximo"
    )

    axes[0, 0].set_ylabel(
        "w max [m s$^{-1}$]"
    )

    axes[0, 1].set_title(
        "(b) Graupel maximo"
    )

    axes[0, 1].set_ylabel(
        "qg max [g kg$^{-1}$]"
    )

    axes[1, 0].set_title(
        "(c) McCaul F3"
    )

    axes[1, 0].set_ylabel(
        "F3"
    )

    axes[1, 0].set_xlabel(
        "tempo [min]"
    )

    axes[1, 1].set_title(
        "(d) LPI*"
    )

    axes[1, 1].set_ylabel(
        "LPI*"
    )

    axes[1, 1].set_xlabel(
        "tempo [min]"
    )

    # Legenda.
    axes[0, 0].legend()

    # Ajuste.
    fig.tight_layout()

    # Salva.
    fig.savefig(
        caminho,
        dpi=180,
    )

    plt.close(fig)


# ============================================================================
# 15. MODO VARREDURA PRELIMINAR
# ============================================================================

def modo_varredura(
    args,
):
    """
    Executa uma unica varredura para investigar candidatos a B0 e B1.

    Todos os casos sao executados no ambiente CTRL:
        Delta T = 0 K

    Assim, a unica variavel alterada e bolha_k.
    """

    # Pasta especifica.
    pasta = (
        OUTPUT_BASE
        / "varredura_bolha"
    )

    pasta.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove duplicatas e ordena.
    amplitudes = sorted(
        set(
            float(v)
            for v in args.amplitudes
        )
    )

    # Exige amplitudes positivas.
    if any(
        valor <= 0.0
        for valor in amplitudes
    ):
        raise ValueError(
            "Todas as amplitudes da bolha devem ser positivas."
        )

    # Registra o commit antes da bateria.
    commit_inicial = obter_commit_git()

    # Lista de resumos.
    resumos = []

    # Executa todas as amplitudes.
    for amplitude in amplitudes:

        # Nome seguro.
        caso = (
            "SCAN_B_"
            + rotulo_amplitude(
                amplitude
            )
        )

        # Executa sempre sem aquecimento.
        _, _, resumo = executar_caso(
            caso=caso,
            bolha_k=amplitude,
            delta_t_ambiente_k=0.0,
            tempo_min=args.tempo_varredura,
            output_base=pasta,
        )

        resumos.append(
            resumo
        )

        # O codigo nao pode mudar no meio da varredura.
        commit_agora = obter_commit_git()

        if commit_agora != commit_inicial:
            raise RuntimeError(
                "O commit Git mudou durante a varredura. "
                "Repita a bateria usando um unico commit."
            )

    # ------------------------------------------------------------------------
    # Salva tabela.
    # ------------------------------------------------------------------------

    tabela = (
        pasta
        / "resumo_varredura_bolha.csv"
    )

    salvar_tabela_resumo(
        resumos,
        tabela,
    )

    # ------------------------------------------------------------------------
    # Salva figura.
    # ------------------------------------------------------------------------

    figura = (
        pasta
        / "comparacao_varredura_bolha.png"
    )

    salvar_figura_varredura(
        resumos=resumos,
        caminho=figura,
    )

    # ------------------------------------------------------------------------
    # Lista candidatos que passaram os criterios minimos.
    # ------------------------------------------------------------------------

    candidatos_b0 = [
        r["bolha_k"]
        for r in resumos
        if r["candidato_B0"]
    ]

    # ------------------------------------------------------------------------
    # Salva metadados da varredura.
    # ------------------------------------------------------------------------

    salvar_json(
        pasta / "metadados_varredura.json",
        {
            "commit": commit_inicial,
            "amplitudes_testadas_K": amplitudes,
            "tempo_varredura_min": float(
                args.tempo_varredura
            ),
            "QMIN_kgkg": float(QMIN),
            "w_lpi_threshold_m_s": float(
                W_LPI_THRESHOLD_M_S
            ),
            "criterios_candidato_B0": {
                "fase_mista_ativa": True,
                "graupel_minus15_ativo": True,
                "updraft_minus15_ativo": True,
                "F3_max_maior_que_zero": True,
                "LPI_star_max_maior_que_zero": True,
                "CFL_adv_menor_igual_1": True,
                "CFL_diff_menor_igual_0p5": True,
            },
            "candidatos_B0_K": candidatos_b0,
        },
    )

    # ------------------------------------------------------------------------
    # Mensagem final.
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("VARREDURA PRELIMINAR CONCLUIDA")
    print("=" * 78)

    print(
        f"Tabela: {tabela}"
    )

    print(
        f"Figura: {figura}"
    )

    if candidatos_b0:

        print()
        print(
            "Amplitudes que satisfizeram os criterios "
            "minimos de candidato a B0:"
        )

        print(
            ", ".join(
                f"{valor:g} K"
                for valor in candidatos_b0
            )
        )

        print()
        print(
            "Isso NAO significa que o menor valor seja "
            "automaticamente o melhor B0."
        )

        print(
            "Escolha um B0 robusto, nao apenas marginal, "
            "e depois escolha B1 > B0 com conveccao "
            "claramente mais intensa."
        )

    else:

        print()
        print(
            "Nenhuma amplitude testada satisfez todos os "
            "criterios minimos para B0."
        )

        print(
            "Amplie ou refine a faixa de bolha_k e repita "
            "a varredura."
        )

    print()
    print(
        "Depois execute a matriz final com:"
    )

    print(
        "python experiments/group2_warming_lightning/"
        "experimento_grupo2.py final --b0 <B0> --b1 <B1>"
    )

    print("=" * 78)


# ============================================================================
# 16. COMPARACOES RELATIVAS DA MATRIZ FINAL
# ============================================================================

def adicionar_razoes_relativas(
    resumos,
):
    """
    Acrescenta razoes relativas ao CTRL.

    Isso facilita a interpretacao dos proxies, que devem ser usados
    prioritariamente de forma relativa.
    """

    # Localiza CTRL.
    ctrl = next(
        r
        for r in resumos
        if r["caso"] == "CTRL"
    )

    # Cria copia enriquecida.
    saida = []

    for resumo in resumos:

        novo = dict(
            resumo
        )

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

        saida.append(
            novo
        )

    return saida


# ============================================================================
# 17. MODO FINAL - MATRIZ 2x2
# ============================================================================

def modo_final(
    args,
):
    """
    Executa a matriz cientifica completa com B0 e B1 escolhidos previamente.
    """

    # B0 deve ser positivo.
    if args.b0 <= 0.0:
        raise ValueError(
            "B0 deve ser positivo."
        )

    # B1 deve ser estritamente maior que B0.
    if args.b1 <= args.b0:
        raise ValueError(
            "B1 deve ser maior que B0. "
            f"Recebido: B0={args.b0} K, B1={args.b1} K."
        )

    # Matriz 2x2.
    matriz = {

        # Referencia.
        "CTRL": {
            "bolha_k": float(args.b0),
            "delta_t_ambiente_k": 0.0,
        },

        # Aquecimento com a mesma bolha de referencia.
        "WARM": {
            "bolha_k": float(args.b0),
            "delta_t_ambiente_k": DELTA_T_WARM_K,
        },

        # Intensificacao convectiva sem aquecimento.
        "BUBBLE_PLUS": {
            "bolha_k": float(args.b1),
            "delta_t_ambiente_k": 0.0,
        },

        # Intensificacao convectiva no ambiente aquecido.
        "WARM_BUBBLE_PLUS": {
            "bolha_k": float(args.b1),
            "delta_t_ambiente_k": DELTA_T_WARM_K,
        },
    }

    # Commit inicial.
    commit_inicial = obter_commit_git()

    # Objetos usados na figura.
    resultados_por_caso = {}

    diagnosticos_por_caso = {}

    # Resumos.
    resumos = []

    # Executa os quatro casos.
    for caso, fatores in matriz.items():

        resultado, diagnosticos, resumo = executar_caso(
            caso=caso,
            bolha_k=fatores["bolha_k"],
            delta_t_ambiente_k=fatores[
                "delta_t_ambiente_k"
            ],
            tempo_min=args.tempo_final,
            output_base=OUTPUT_BASE,
        )

        resultados_por_caso[
            caso
        ] = resultado

        diagnosticos_por_caso[
            caso
        ] = diagnosticos

        resumos.append(
            resumo
        )

        # Verifica se houve mudanca de commit.
        commit_agora = obter_commit_git()

        if commit_agora != commit_inicial:
            raise RuntimeError(
                "O commit Git mudou durante a bateria final. "
                "Os quatro casos devem usar o mesmo commit."
            )

    # ------------------------------------------------------------------------
    # Confirma se CTRL realmente satisfaz os criterios de B0.
    # ------------------------------------------------------------------------

    resumo_ctrl = next(
        r
        for r in resumos
        if r["caso"] == "CTRL"
    )

    if not resumo_ctrl["candidato_B0"]:

        warnings.warn(
            "O CTRL final nao satisfez todos os criterios minimos "
            "definidos para B0. Revise a escolha antes de interpretar "
            "cientificamente a matriz.",
            RuntimeWarning,
        )

    # ------------------------------------------------------------------------
    # Acrescenta normalizacoes relativas ao CTRL.
    # ------------------------------------------------------------------------

    resumos_relativos = adicionar_razoes_relativas(
        resumos
    )

    # ------------------------------------------------------------------------
    # Salva tabela final.
    # ------------------------------------------------------------------------

    tabela = (
        OUTPUT_BASE
        / "resumo_grupo2.csv"
    )

    salvar_tabela_resumo(
        resumos_relativos,
        tabela,
    )

    # ------------------------------------------------------------------------
    # Salva matriz experimental.
    # ------------------------------------------------------------------------

    salvar_json(
        OUTPUT_BASE
        / "matriz_experimental.json",
        {
            "B0_K": float(args.b0),
            "B1_K": float(args.b1),
            "delta_T_WARM_K": float(
                DELTA_T_WARM_K
            ),
            "preservar_rh": True,
            "commit": commit_inicial,
            "casos": matriz,
        },
    )

    # ------------------------------------------------------------------------
    # Figura temporal final.
    # ------------------------------------------------------------------------

    figura = (
        OUTPUT_BASE
        / "comparacao_grupo2.png"
    )

    salvar_figura_comparativa_final(
        resultados_por_caso=(
            resultados_por_caso
        ),
        diagnosticos_por_caso=(
            diagnosticos_por_caso
        ),
        caminho=figura,
    )

    # ------------------------------------------------------------------------
    # Mensagem final.
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("MATRIZ FINAL DO GRUPO 2 CONCLUIDA")
    print("=" * 78)

    print(
        f"B0 = {args.b0:.3f} K"
    )

    print(
        f"B1 = {args.b1:.3f} K"
    )

    print(
        f"Resumo CSV: {tabela}"
    )

    print(
        f"Figura: {figura}"
    )

    print(
        f"Commit: {commit_inicial}"
    )

    print("=" * 78)


# ============================================================================
# 18. INTERFACE DE LINHA DE COMANDO
# ============================================================================

def construir_parser():
    """
    Define os subcomandos e argumentos aceitos.
    """

    # Parser principal.
    parser = argparse.ArgumentParser(
        description=(
            "Grupo 2: varredura da intensidade da bolha, "
            "aquecimento e diagnosticos eletricos."
        )
    )

    # Subcomandos.
    subparsers = parser.add_subparsers(
        dest="modo",
        required=True,
    )

    # ------------------------------------------------------------------------
    # 18.1. Subcomando VARREDURA
    # ------------------------------------------------------------------------

    p_scan = subparsers.add_parser(
        "varredura",
        help=(
            "Testa varias amplitudes de bolha no ambiente CTRL "
            "para orientar a escolha de B0 e B1."
        ),
    )

    # Amplitudes iniciais de teste.
    p_scan.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            14.0,
        ],
        help=(
            "Lista de amplitudes da bolha em K. "
            "Padrao: 4 6 8 10 12 14."
        ),
    )

    # Tempo da varredura.
    p_scan.add_argument(
        "--tempo-varredura",
        type=float,
        default=TEMPO_PADRAO_MIN,
        help=(
            "Duracao de cada simulacao da varredura [min]. "
            f"Padrao: {TEMPO_PADRAO_MIN:g}."
        ),
    )

    # ------------------------------------------------------------------------
    # 18.2. Subcomando FINAL
    # ------------------------------------------------------------------------

    p_final = subparsers.add_parser(
        "final",
        help=(
            "Executa CTRL, WARM, BUBBLE_PLUS e "
            "WARM_BUBBLE_PLUS com B0/B1 ja escolhidos."
        ),
    )

    # B0 agora e obrigatorio.
    p_final.add_argument(
        "--b0",
        type=float,
        required=True,
        help=(
            "Amplitude B0 [K] escolhida a partir da varredura."
        ),
    )

    # B1 agora tambem e obrigatorio.
    p_final.add_argument(
        "--b1",
        type=float,
        required=True,
        help=(
            "Amplitude B1 [K] escolhida a partir da varredura. "
            "Deve ser maior que B0."
        ),
    )

    # Tempo final.
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
# 19. FUNCAO PRINCIPAL
# ============================================================================

def main():
    """
    Ponto de entrada do programa.
    """

    # Constroi o parser.
    parser = construir_parser()

    # Le a linha de comando.
    args = parser.parse_args()

    # Garante a existencia do diretorio geral.
    OUTPUT_BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Informacoes iniciais.
    print(
        f"Raiz do repositorio: {ROOT}"
    )

    print(
        f"Saidas do Grupo 2:   {OUTPUT_BASE}"
    )

    print(
        f"Commit atual:        {obter_commit_git()}"
    )

    # Executa o modo escolhido.
    if args.modo == "varredura":

        modo_varredura(
            args
        )

    elif args.modo == "final":

        modo_final(
            args
        )

    else:

        # Nao deve ocorrer porque argparse restringe os modos.
        raise ValueError(
            f"Modo desconhecido: {args.modo}"
        )


# ============================================================================
# 20. EXECUCAO DIRETA
# ============================================================================

# So executa main() quando este arquivo for chamado diretamente.
if __name__ == "__main__":
    main()
