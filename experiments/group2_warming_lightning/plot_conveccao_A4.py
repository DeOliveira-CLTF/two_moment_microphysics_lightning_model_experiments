# -*- coding: utf-8 -*-
"""
Figura de artigo: evolucao da conveccao e diagnosticos eletricos
================================================================

Este script le o arquivo .npz gerado pelo experimento do Grupo 2 e produz
uma figura composta para uso em artigo cientifico.

ESTRUTURA DA FIGURA
-------------------

(a-e) Snapshots verticais x-z em tempos selecionados:
    - sombreado:
        qc + qi + qs + qg [g/kg]
    - contornos:
        velocidade vertical positiva w [m/s]
    - isotermas:
        0 C
        -15 C
        -20 C
    - marcadores sobre a isoterma de -15 C:
        colunas onde McCaul F3 >= limiar de referencia
        tamanho do marcador proporcional a F3

(f) Evolucao do topo da nuvem

(g) Evolucao do McCaul F3 maximo
    - inclui linha horizontal em F3 = 0.02 por padrao

(h) Evolucao do LPI* maximo

IMPORTANTE
----------
O valor F3 = 0.02 e usado apenas como REFERENCIA baseada na aplicacao
original de McCaul et al. (2009).

No presente modelo idealizado 2D, F3 nao deve ser interpretado diretamente
como numero observado de flashes sem recalibracao.

Da mesma forma, LPI* e um indice relativo de potencial eletrico e nao uma
contagem de flashes.

O topo da nuvem usado no painel temporal e definido por:

    qc + qi > 1e-5 kg/kg

por padrao, consistente com o criterio usado no diagnostico verbose do
nucleo dinamico.

USO
---

Executando a partir de:

    experiments/group2_warming_lightning/

pode-se usar:

python plot_conveccao.py --input "outputs/group2/varredura_bolha/SCAN_B_10K/resultados_SCAN_B_10K.npz" --output-prefix "outputs/group2/varredura_bolha/SCAN_B_10K/figura_artigo_SCAN_B_10K" --times 10 15 20 30 40

Os caminhos relativos sao automaticamente interpretados a partir da
RAIZ DO REPOSITORIO.

SAIDAS
------
<output-prefix>.png
<output-prefix>.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

# Permite salvar figuras sem abrir janela.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


# ============================================================================
# CONFIGURACAO GRAFICA PARA ARTIGO EM A4
# ============================================================================

# A4 em orientacao paisagem, em polegadas.
# 297 x 210 mm = 11.69 x 8.27 in.
A4_LANDSCAPE_IN = (11.69, 8.27)

# Tamanhos pensados para permanecerem legiveis quando a figura for impressa
# em uma folha A4 inteira.
plt.rcParams.update(
    {
        "font.size": 8.0,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 8.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================================
# 1. RAIZ DO REPOSITORIO
# ============================================================================

# Este arquivo deve ficar em:
#
# experiments/group2_warming_lightning/plot_conveccao.py
#
# parents[0] -> group2_warming_lightning
# parents[1] -> experiments
# parents[2] -> raiz do repositorio
ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# 2. CONSTANTES GRAFICAS E DIAGNOSTICAS
# ============================================================================

# Limiar usado por padrao para destacar F3.
#
# IMPORTANTE:
# aqui ele e apenas uma referencia diagnostica baseada na aplicacao
# original de McCaul et al. (2009).
DEFAULT_F3_THRESHOLD = 0.02

# Criterio padrao para topo de nuvem.
#
# Usa qc + qi, como no diagnostico verbose do nucleo.
DEFAULT_CLOUD_TOP_THRESHOLD = 1.0e-5

# Niveis de movimento ascendente mostrados nos snapshots.
W_CONTOUR_LEVELS = (
    2.0,
    5.0,
    10.0,
)

# Isotermas relevantes para fase mista.
ISOTHERMS_C = (
    0.0,
    -15.0,
    -20.0,
)


# ============================================================================
# 3. CAMINHOS
# ============================================================================

def resolver_caminho(caminho: str | Path) -> Path:
    """
    Resolve um caminho.

    Se for absoluto:
        usa diretamente.

    Se for relativo:
        interpreta a partir da raiz do repositorio.

    Isso permite executar este script a partir de
    experiments/group2_warming_lightning sem precisar escrever ../../.
    """

    caminho = Path(
        caminho
    ).expanduser()

    if caminho.is_absolute():
        return caminho.resolve()

    return (
        ROOT
        / caminho
    ).resolve()


# ============================================================================
# 4. LEITURA DO NPZ
# ============================================================================

def carregar_npz(
    caminho_npz: Path,
) -> dict:
    """
    Carrega o arquivo NPZ e transforma seu conteudo em dicionario.
    """

    if not caminho_npz.exists():

        raise FileNotFoundError(
            "\nArquivo NPZ nao encontrado.\n\n"
            f"Caminho procurado:\n{caminho_npz}\n\n"
            f"Raiz do repositorio detectada:\n{ROOT}\n"
        )

    with np.load(
        caminho_npz,
        allow_pickle=True,
    ) as dados:

        return {
            nome: dados[nome]
            for nome in dados.files
        }


def obter_campo(
    dados: dict,
    nome: str,
    obrigatorio: bool = True,
):
    """
    Recupera um campo do arquivo NPZ.
    """

    if nome in dados:
        return np.asarray(
            dados[nome]
        )

    if obrigatorio:

        campos = "\n".join(
            sorted(
                dados.keys()
            )
        )

        raise KeyError(
            f"\nCampo '{nome}' nao encontrado no NPZ.\n\n"
            f"Campos disponiveis:\n{campos}"
        )

    return None


# ============================================================================
# 5. FUNCOES NUMERICAS SEGURAS
# ============================================================================

def maximo_seguro(
    campo,
):
    """
    Retorna nanmax sem emitir erro para campos completamente NaN.
    """

    campo = np.asarray(
        campo,
        dtype=float,
    )

    if (
        campo.size == 0
        or np.all(
            np.isnan(campo)
        )
    ):
        return np.nan

    return float(
        np.nanmax(
            campo
        )
    )


def nanmax_por_tempo(
    campo,
):
    """
    Calcula o maximo espacial para cada tempo.

    Espera um campo cujo primeiro eixo seja tempo.
    """

    campo = np.asarray(
        campo,
        dtype=float,
    )

    nt = campo.shape[0]

    saida = np.full(
        nt,
        np.nan,
        dtype=float,
    )

    for it in range(nt):

        fatia = campo[it]

        if not np.all(
            np.isnan(
                fatia
            )
        ):

            saida[it] = np.nanmax(
                fatia
            )

    return saida


# ============================================================================
# 6. SELECAO DOS TEMPOS
# ============================================================================

def tempo_para_indices(
    t_min,
    tempos_desejados_min,
):
    """
    Encontra o frame salvo mais proximo de cada tempo solicitado.
    """

    t_min = np.asarray(
        t_min,
        dtype=float,
    )

    indices = []

    for tempo_alvo in tempos_desejados_min:

        idx = int(
            np.argmin(
                np.abs(
                    t_min
                    - float(tempo_alvo)
                )
            )
        )

        indices.append(
            idx
        )

    return indices


# ============================================================================
# 7. ALTURA DAS ISOTERMAS
# ============================================================================

def altura_isoterma_coluna(
    T_col_K,
    z_m,
    nivel_c,
):
    """
    Interpola a altura de uma isoterma em uma coluna vertical.

    Retorna NaN se a isoterma estiver fora do dominio.
    """

    T_col_K = np.asarray(
        T_col_K,
        dtype=float,
    )

    z_m = np.asarray(
        z_m,
        dtype=float,
    )

    T_col_C = (
        T_col_K
        - 273.15
    )

    alvo = float(
        nivel_c
    )

    diferenca = (
        T_col_C
        - alvo
    )

    # Percorre pares de niveis adjacentes.
    for iz in range(
        len(z_m) - 1
    ):

        d1 = diferenca[iz]
        d2 = diferenca[iz + 1]

        if (
            not np.isfinite(d1)
            or not np.isfinite(d2)
        ):
            continue

        # Exatamente na isoterma.
        if d1 == 0.0:

            return float(
                z_m[iz]
            )

        # Cruzamento.
        if d1 * d2 < 0.0:

            T1 = T_col_C[iz]
            T2 = T_col_C[iz + 1]

            z1 = z_m[iz]
            z2 = z_m[iz + 1]

            if T2 == T1:
                return float(
                    0.5
                    * (z1 + z2)
                )

            fracao = (
                (alvo - T1)
                / (T2 - T1)
            )

            return float(
                z1
                + fracao
                * (z2 - z1)
            )

    return np.nan


def campo_altura_isoterma(
    T_K,
    z_m,
    nivel_c,
):
    """
    Calcula a altura da isoterma para cada tempo e coluna x.

    Entrada:
        T_K -> (nt, nx, nz)

    Saida:
        z_iso -> (nt, nx)
    """

    T_K = np.asarray(
        T_K,
        dtype=float,
    )

    nt, nx, _ = T_K.shape

    saida = np.full(
        (nt, nx),
        np.nan,
        dtype=float,
    )

    for it in range(nt):

        for ix in range(nx):

            saida[it, ix] = altura_isoterma_coluna(
                T_col_K=T_K[it, ix, :],
                z_m=z_m,
                nivel_c=nivel_c,
            )

    return saida


# ============================================================================
# 8. TOPO DA NUVEM
# ============================================================================

def calcular_topo_nuvem(
    qc,
    qi,
    z_m,
    threshold_kgkg=DEFAULT_CLOUD_TOP_THRESHOLD,
):
    """
    Calcula topo da nuvem em cada tempo.

    Criterio:

        qc + qi > threshold

    Isso e deliberadamente diferente de usar QMIN.

    QMIN e muito pequeno para representar um topo de nuvem fisicamente
    significativo e pode detectar apenas tracos numericos de hidrometeoros.
    """

    qc = np.asarray(
        qc,
        dtype=float,
    )

    qi = np.asarray(
        qi,
        dtype=float,
    )

    z_m = np.asarray(
        z_m,
        dtype=float,
    )

    condensado_nuvem = (
        qc
        + qi
    )

    nt = condensado_nuvem.shape[0]

    topo = np.zeros(
        nt,
        dtype=float,
    )

    for it in range(nt):

        # Maior qc+qi em cada nivel vertical.
        maximo_horizontal = np.nanmax(
            condensado_nuvem[it],
            axis=0,
        )

        niveis = np.where(
            maximo_horizontal
            > threshold_kgkg
        )[0]

        if niveis.size > 0:

            topo[it] = z_m[
                niveis[-1]
            ]

        else:

            topo[it] = 0.0

    return topo


# ============================================================================
# 9. ESCALA DOS MARCADORES DE F3
# ============================================================================

def tamanhos_marcadores_f3(
    valores_f3,
    f3_max_global,
    threshold,
    tamanho_min=20.0,
    tamanho_max=125.0,
):
    """
    Converte F3 em tamanho de marcador.

    Usa a mesma escala em TODOS os snapshots.

    Isso e importante:
    um F3 = 0.03 deve aparecer com o mesmo tamanho em qualquer painel.
    """

    valores_f3 = np.asarray(
        valores_f3,
        dtype=float,
    )

    if valores_f3.size == 0:

        return np.asarray(
            [],
            dtype=float,
        )

    if (
        not np.isfinite(f3_max_global)
        or f3_max_global <= threshold
    ):

        return np.full(
            valores_f3.shape,
            tamanho_min,
            dtype=float,
        )

    # Normalizacao considerando apenas a faixa acima do limiar.
    numerador = np.maximum(
        valores_f3
        - threshold,
        0.0,
    )

    denominador = (
        f3_max_global
        - threshold
    )

    fracao = (
        numerador
        / denominador
    )

    # Raiz quadrada melhora a legibilidade dos valores menores.
    fracao = np.sqrt(
        np.clip(
            fracao,
            0.0,
            1.0,
        )
    )

    return (
        tamanho_min
        + (
            tamanho_max
            - tamanho_min
        )
        * fracao
    )


# ============================================================================
# 10. ESCALA ROBUSTA DO SOMBREADO
# ============================================================================

def calcular_vmax_condensado(
    qcond,
    indices_tempos,
    percentil=99.5,
):
    """
    Determina vmax do sombreado usando percentil robusto.

    Evita que um unico pixel extremo deixe o restante da nuvem praticamente
    branco.

    Valores acima do vmax continuam existindo; a colorbar usa extend='max'.
    """

    valores = []

    for idx in indices_tempos:

        fatia = np.asarray(
            qcond[idx],
            dtype=float,
        ).ravel()

        fatia = fatia[
            np.isfinite(
                fatia
            )
        ]

        fatia = fatia[
            fatia > 0.0
        ]

        if fatia.size > 0:

            valores.append(
                fatia
            )

    if not valores:

        return 1.0

    valores = np.concatenate(
        valores
    )

    vmax = np.percentile(
        valores,
        percentil,
    )

    return max(
        float(vmax),
        0.05,
    )


# ============================================================================
# 11. PLOT DOS SNAPSHOTS
# ============================================================================

def plot_snapshot(
    ax,
    indice,
    letra,
    t_min,
    x_km,
    z_km,
    qcond,
    w,
    T,
    f3,
    z_minus15_m,
    norm_condensado,
    f3_threshold,
    f3_max_global,
    zmax_km,
):
    """
    Desenha um snapshot vertical x-z.
    """

    tempo = float(
        t_min[indice]
    )

    qcond_t = np.asarray(
        qcond[indice],
        dtype=float,
    )

    w_t = np.asarray(
        w[indice],
        dtype=float,
    )

    T_t_C = (
        np.asarray(
            T[indice],
            dtype=float,
        )
        - 273.15
    )

    f3_t = np.asarray(
        f3[indice],
        dtype=float,
    )

    z15_t_km = (
        np.asarray(
            z_minus15_m[indice],
            dtype=float,
        )
        / 1000.0
    )

    # ------------------------------------------------------------------------
    # Sombreado
    # ------------------------------------------------------------------------

    pcm = ax.pcolormesh(
        x_km,
        z_km,
        qcond_t.T,
        shading="auto",
        cmap="Blues",
        norm=norm_condensado,
        rasterized=True,
    )

    # ------------------------------------------------------------------------
    # Contornos de w positivo
    # ------------------------------------------------------------------------

    w_max_instante = maximo_seguro(
        w_t
    )

    niveis_w = [
        nivel
        for nivel in W_CONTOUR_LEVELS
        if (
            np.isfinite(
                w_max_instante
            )
            and w_max_instante >= nivel
        )
    ]

    if niveis_w:

        contornos_w = ax.contour(
            x_km,
            z_km,
            w_t.T,
            levels=niveis_w,
            colors="black",
            linewidths=0.8,
        )

        ax.clabel(
            contornos_w,
            inline=True,
            fontsize=6.2,
            fmt=lambda valor: f"{valor:g}",
        )

    # ------------------------------------------------------------------------
    # Isotermas
    # ------------------------------------------------------------------------

    Tmin = maximo_seguro(
        -T_t_C
    )

    Tmin = (
        -Tmin
        if np.isfinite(Tmin)
        else np.nan
    )

    Tmax = maximo_seguro(
        T_t_C
    )

    estilos_isotermas = {
        0.0: {
            "color": "forestgreen",
            "linestyle": "-",
            "linewidth": 1.3,
        },

        -15.0: {
            "color": "firebrick",
            "linestyle": "--",
            "linewidth": 1.4,
        },

        -20.0: {
            "color": "royalblue",
            "linestyle": ":",
            "linewidth": 1.3,
        },
    }

    for nivel_c in ISOTHERMS_C:

        if (
            np.isfinite(Tmin)
            and np.isfinite(Tmax)
            and Tmin <= nivel_c <= Tmax
        ):

            estilo = estilos_isotermas[
                nivel_c
            ]

            ax.contour(
                x_km,
                z_km,
                T_t_C.T,
                levels=[
                    nivel_c
                ],
                colors=[
                    estilo["color"]
                ],
                linestyles=[
                    estilo["linestyle"]
                ],
                linewidths=[
                    estilo["linewidth"]
                ],
            )

    # ------------------------------------------------------------------------
    # F3 sobre -15 C
    # ------------------------------------------------------------------------

    mascara_f3 = (
        np.isfinite(
            f3_t
        )
        & np.isfinite(
            z15_t_km
        )
        & (
            f3_t
            >= f3_threshold
        )
    )

    if np.any(
        mascara_f3
    ):

        tamanhos = tamanhos_marcadores_f3(
            valores_f3=f3_t[
                mascara_f3
            ],
            f3_max_global=f3_max_global,
            threshold=f3_threshold,
        )

        ax.scatter(
            x_km[
                mascara_f3
            ],
            z15_t_km[
                mascara_f3
            ],
            s=tamanhos,
            facecolor="gold",
            edgecolor="black",
            linewidth=0.45,
            alpha=0.90,
            zorder=8,
        )

    # ------------------------------------------------------------------------
    # Titulos
    # ------------------------------------------------------------------------

    ax.set_title(
        f"({letra})  {tempo:.0f} min",
        loc="left",
        fontsize=8.7,
        fontweight="bold",
        pad=4.0,
    )

    # ------------------------------------------------------------------------
    # Anotacao
    # ------------------------------------------------------------------------

    f3_max_instante = maximo_seguro(
        f3_t
    )

    if np.isfinite(
        f3_max_instante
    ):

        texto_f3 = (
            f"{f3_max_instante:.3f}"
        )

    else:

        texto_f3 = "NaN"

    texto = (
        rf"$w_{{max}}$ = {w_max_instante:.1f} m s$^{{-1}}$"
        "\n"
        rf"$F_{{3,max}}$ = {texto_f3}"
    )

    ax.text(
        0.97,
        0.97,
        texto,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=6.6,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.65",
            "alpha": 0.83,
        },
    )

    # ------------------------------------------------------------------------
    # Eixos
    # ------------------------------------------------------------------------

    ax.set_xlim(
        x_km[0],
        x_km[-1],
    )

    ax.set_ylim(
        0.0,
        zmax_km,
    )

    ax.set_xlabel(
        r"$x$ (km)",
        fontsize=8.0,
        labelpad=2.0,
    )

    ax.tick_params(
        axis="both",
        labelsize=7.2,
        pad=2.0,
    )

    return pcm


# ============================================================================
# 12. FIGURA COMPLETA
# ============================================================================

def gerar_figura_artigo(
    caminho_npz: Path,
    output_prefix: Path,
    tempos_desejados_min,
    f3_threshold=DEFAULT_F3_THRESHOLD,
    cloud_top_threshold=DEFAULT_CLOUD_TOP_THRESHOLD,
    zmax_km=None,
):
    """
    Gera a figura final para artigo.
    """

    # ------------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------------

    dados = carregar_npz(
        caminho_npz
    )

    t_s = obter_campo(
        dados,
        "t_s",
    )

    x_m = obter_campo(
        dados,
        "x_m",
    )

    z_m = obter_campo(
        dados,
        "z_m",
    )

    T = obter_campo(
        dados,
        "T",
    )

    w = obter_campo(
        dados,
        "w",
    )

    qc = obter_campo(
        dados,
        "qc",
    )

    qi = obter_campo(
        dados,
        "qi",
    )

    qs = obter_campo(
        dados,
        "qs",
    )

    qg = obter_campo(
        dados,
        "qg",
    )

    f3 = obter_campo(
        dados,
        "lightning_f3",
    )

    lpi = obter_campo(
        dados,
        "lightning_lpi_star",
    )

    z_minus15_m = obter_campo(
        dados,
        "lightning_z_minus15_m",
        obrigatorio=False,
    )

    # Se o modulo lightning nao tiver salvo z(-15 C),
    # calcula diretamente a partir de T.
    if z_minus15_m is None:

        print(
            "lightning_z_minus15_m nao encontrado. "
            "Calculando a isoterma de -15 C a partir de T..."
        )

        z_minus15_m = campo_altura_isoterma(
            T_K=T,
            z_m=z_m,
            nivel_c=-15.0,
        )

    # ------------------------------------------------------------------------
    # Coordenadas
    # ------------------------------------------------------------------------

    t_min = (
        np.asarray(
            t_s,
            dtype=float,
        )
        / 60.0
    )

    x_km = (
        np.asarray(
            x_m,
            dtype=float,
        )
        / 1000.0
    )

    z_km = (
        np.asarray(
            z_m,
            dtype=float,
        )
        / 1000.0
    )

    # ------------------------------------------------------------------------
    # Campo usado no sombreado
    # ------------------------------------------------------------------------

    # Exclui chuva para que o sombreado represente melhor a estrutura
    # da nuvem e da fase congelada, e nao o eixo de precipitacao.
    #
    # qc = agua de nuvem
    # qi = gelo de nuvem
    # qs = neve
    # qg = graupel
    qcond = (
        qc
        + qi
        + qs
        + qg
    ) * 1000.0

    # Unidade final: g/kg.

    # ------------------------------------------------------------------------
    # Topo da nuvem
    # ------------------------------------------------------------------------

    topo_m = calcular_topo_nuvem(
        qc=qc,
        qi=qi,
        z_m=z_m,
        threshold_kgkg=cloud_top_threshold,
    )

    topo_km = (
        topo_m
        / 1000.0
    )

    # ------------------------------------------------------------------------
    # Series eletricas
    # ------------------------------------------------------------------------

    f3_max_t = nanmax_por_tempo(
        f3
    )

    lpi_max_t = nanmax_por_tempo(
        lpi
    )

    f3_max_global = maximo_seguro(
        f3
    )

    # ------------------------------------------------------------------------
    # Tempos dos snapshots
    # ------------------------------------------------------------------------

    indices_snapshots = tempo_para_indices(
        t_min=t_min,
        tempos_desejados_min=(
            tempos_desejados_min
        ),
    )

    # ------------------------------------------------------------------------
    # Limite vertical
    # ------------------------------------------------------------------------

    if zmax_km is None:

        zmax_km = float(
            z_km[-1]
        )

    else:

        zmax_km = min(
            float(zmax_km),
            float(z_km[-1]),
        )

    # ------------------------------------------------------------------------
    # Escala de condensado
    # ------------------------------------------------------------------------

    vmax_condensado = calcular_vmax_condensado(
        qcond=qcond,
        indices_tempos=(
            indices_snapshots
        ),
        percentil=99.5,
    )

    norm_condensado = Normalize(
        vmin=0.0,
        vmax=vmax_condensado,
    )

    # ------------------------------------------------------------------------
    # Numero de snapshots
    # ------------------------------------------------------------------------

    n_snapshots = len(
        indices_snapshots
    )

    # ------------------------------------------------------------------------
    # Figura em A4 paisagem
    # ------------------------------------------------------------------------

    # A figura final possui exatamente o tamanho fisico de uma folha A4 em
    # orientacao paisagem. Nao usamos bbox_inches="tight" no salvamento para
    # nao alterar esse tamanho fisico.
    fig = plt.figure(
        figsize=A4_LANDSCAPE_IN,
    )

    # Estrutura vertical:
    #
    #   linha 0 -> legenda dos snapshots
    #   linha 1 -> snapshots + colorbar vertical a direita
    #   linha 2 -> serie temporal F3
    #   linha 3 -> serie temporal LPI*
    #
    # A legenda fica em uma linha propria para nunca sobrepor os titulos.
    outer = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[
            0.34,
            4.25,
            0.90,
            0.90,
        ],
        left=0.060,
        right=0.985,
        bottom=0.070,
        top=0.985,
        hspace=0.30,
    )

    # ------------------------------------------------------------------------
    # Linha exclusiva da legenda
    # ------------------------------------------------------------------------

    ax_legenda = fig.add_subplot(
        outer[0, 0]
    )

    ax_legenda.axis(
        "off"
    )

    handles = [

        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.1,
            label=r"$w$ = 2, 5, 10 m s$^{-1}$",
        ),

        Line2D(
            [0],
            [0],
            color="forestgreen",
            linestyle="-",
            linewidth=1.5,
            label="0 °C",
        ),

        Line2D(
            [0],
            [0],
            color="firebrick",
            linestyle="--",
            linewidth=1.5,
            label="-15 °C",
        ),

        Line2D(
            [0],
            [0],
            color="royalblue",
            linestyle=":",
            linewidth=1.5,
            label="-20 °C",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            markersize=6.5,
            markerfacecolor="gold",
            markeredgecolor="black",
            linestyle="None",
            label=(
                rf"$F_3 \geq {f3_threshold:g}$ "
                "na isoterma de -15 °C"
            ),
        ),
    ]

    ax_legenda.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=False,
        fontsize=8.1,
        handlelength=2.3,
        handletextpad=0.55,
        columnspacing=1.25,
        borderaxespad=0.0,
    )

    # ------------------------------------------------------------------------
    # Snapshots + colorbar lateral
    # ------------------------------------------------------------------------

    # A ultima coluna e reservada exclusivamente para a colorbar.
    snapshots_grid = gridspec.GridSpecFromSubplotSpec(
        nrows=1,
        ncols=n_snapshots + 1,
        subplot_spec=outer[1, 0],
        width_ratios=(
            [1.0] * n_snapshots
            + [0.070]
        ),
        wspace=0.065,
    )

    axes_snapshots = []

    letras = "abcdefghijklmnopqrstuvwxyz"

    pcm = None

    for i, idx in enumerate(
        indices_snapshots
    ):

        if i == 0:

            ax = fig.add_subplot(
                snapshots_grid[0, i]
            )

        else:

            ax = fig.add_subplot(
                snapshots_grid[0, i],
                sharey=axes_snapshots[0],
            )

        axes_snapshots.append(
            ax
        )

        pcm = plot_snapshot(
            ax=ax,
            indice=idx,
            letra=letras[i],
            t_min=t_min,
            x_km=x_km,
            z_km=z_km,
            qcond=qcond,
            w=w,
            T=T,
            f3=f3,
            z_minus15_m=z_minus15_m,
            norm_condensado=norm_condensado,
            f3_threshold=f3_threshold,
            f3_max_global=f3_max_global,
            zmax_km=zmax_km,
        )

        if i == 0:

            ax.set_ylabel(
                "Altura (km)",
                fontsize=8.2,
                labelpad=3.0,
            )

        else:

            ax.tick_params(
                labelleft=False
            )

    # Colorbar vertical na lateral direita.
    cax = fig.add_subplot(
        snapshots_grid[0, -1]
    )

    cbar = fig.colorbar(
        pcm,
        cax=cax,
        orientation="vertical",
        extend="max",
    )

    # Rotulo curto para permanecer legivel em A4.
    cbar.set_label(
        r"$q_c+q_i+q_s+q_g$ (g kg$^{-1}$)",
        fontsize=8.1,
        labelpad=6.0,
    )

    cbar.ax.tick_params(
        labelsize=7.2,
        pad=2.0,
    )

    # ------------------------------------------------------------------------
    # Letras dos dois paineis temporais
    # ------------------------------------------------------------------------

    letra_f3 = letras[
        n_snapshots
    ]

    letra_lpi = letras[
        n_snapshots + 1
    ]

    # ------------------------------------------------------------------------
    # Painel temporal: F3
    # ------------------------------------------------------------------------

    ax_f3 = fig.add_subplot(
        outer[2, 0],
    )

    ax_f3.plot(
        t_min,
        f3_max_t,
        linewidth=1.8,
        color="darkorange",
        label=r"$F_{3,\max}$",
    )

    ax_f3.axhline(
        f3_threshold,
        color="0.25",
        linestyle="--",
        linewidth=1.0,
        label=(
            rf"Referência = {f3_threshold:g}"
        ),
    )

    ax_f3.set_ylabel(
        r"$F_{3,\max}$",
        fontsize=8.2,
    )

    ax_f3.set_ylim(
        bottom=0.0
    )

    ax_f3.text(
        0.006,
        0.78,
        f"({letra_f3})",
        transform=ax_f3.transAxes,
        fontweight="bold",
        fontsize=8.7,
    )

    ax_f3.legend(
        loc="upper right",
        fontsize=7.5,
        frameon=False,
        ncol=2,
        handlelength=2.0,
        columnspacing=1.0,
    )

    ax_f3.grid(
        linestyle=":",
        alpha=0.45,
    )

    ax_f3.tick_params(
        axis="both",
        labelsize=7.2,
        pad=2.0,
        labelbottom=False,
    )

    # ------------------------------------------------------------------------
    # Painel temporal: LPI*
    # ------------------------------------------------------------------------

    ax_lpi = fig.add_subplot(
        outer[3, 0],
        sharex=ax_f3,
    )

    ax_lpi.plot(
        t_min,
        lpi_max_t,
        linewidth=1.8,
        color="mediumpurple",
    )

    ax_lpi.set_ylabel(
        r"$LPI^*_{\max}$",
        fontsize=8.2,
    )

    ax_lpi.set_xlabel(
        "Tempo (min)",
        fontsize=8.2,
    )

    ax_lpi.set_ylim(
        bottom=0.0
    )

    ax_lpi.text(
        0.006,
        0.78,
        f"({letra_lpi})",
        transform=ax_lpi.transAxes,
        fontweight="bold",
        fontsize=8.7,
    )

    ax_lpi.grid(
        linestyle=":",
        alpha=0.45,
    )

    ax_lpi.tick_params(
        axis="both",
        labelsize=7.2,
        pad=2.0,
    )

    # ------------------------------------------------------------------------
    # Linhas verticais que marcam os tempos dos snapshots
    # ------------------------------------------------------------------------

    for idx in indices_snapshots:

        tempo_snapshot = t_min[
            idx
        ]

        for ax in (
            ax_f3,
            ax_lpi,
        ):

            ax.axvline(
                tempo_snapshot,
                color="0.80",
                linestyle=":",
                linewidth=0.75,
                zorder=0,
            )

    ax_lpi.set_xlim(
        t_min[0],
        t_min[-1],
    )

    # ------------------------------------------------------------------------
    # Salvamento
    # ------------------------------------------------------------------------

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    caminho_png = output_prefix.with_suffix(
        ".png"
    )

    caminho_pdf = output_prefix.with_suffix(
        ".pdf"
    )

    # Nao usar bbox_inches="tight": assim o arquivo mantem o tamanho A4
    # definido em figsize.
    fig.savefig(
        caminho_png,
        dpi=300,
    )

    fig.savefig(
        caminho_pdf,
    )

    plt.close(
        fig
    )

    # ------------------------------------------------------------------------
    # Relatorio
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("FIGURA CONCLUIDA")
    print("=" * 78)

    print(
        f"Entrada:           {caminho_npz}"
    )

    print(
        f"PNG:               {caminho_png}"
    )

    print(
        f"PDF:               {caminho_pdf}"
    )

    print(
        f"F3 referencia:     {f3_threshold:g}"
    )

    print(
        "Topo da nuvem:    "
        f"qc + qi > {cloud_top_threshold:.2e} kg/kg"
    )

    print(
        "vmax condensado:  "
        f"{vmax_condensado:.3f} g/kg "
        "(percentil 99.5 dos snapshots)"
    )

    print("=" * 78)


# ============================================================================
# 13. ARGUMENTOS
# ============================================================================

def construir_parser():
    """
    Constroi interface de linha de comando.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Gera figura para artigo mostrando evolucao vertical "
            "da conveccao, McCaul F3 e LPI*."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Arquivo .npz do experimento. "
            "Caminhos relativos sao interpretados a partir "
            "da raiz do repositorio."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default=(
            "outputs/group2/"
            "figura_artigo_conveccao"
        ),
        help=(
            "Prefixo da saida, sem extensao. "
            "Serao criados PNG e PDF."
        ),
    )

    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[
            5.0,
            10.0,
            15.0,
            20.0,
            30.0,
        ],
        help=(
            "Tempos dos snapshots em minutos. "
            "Padrao: 5 10 15 20 30."
        ),
    )

    parser.add_argument(
        "--f3-threshold",
        type=float,
        default=DEFAULT_F3_THRESHOLD,
        help=(
            "Limiar de referencia de F3 para mostrar marcadores. "
            f"Padrao: {DEFAULT_F3_THRESHOLD:g}."
        ),
    )

    parser.add_argument(
        "--cloud-top-threshold",
        type=float,
        default=DEFAULT_CLOUD_TOP_THRESHOLD,
        help=(
            "Limiar de qc+qi para definir o topo da nuvem [kg/kg]. "
            f"Padrao: {DEFAULT_CLOUD_TOP_THRESHOLD:.1e}."
        ),
    )

    parser.add_argument(
        "--zmax-km",
        type=float,
        default=None,
        help=(
            "Limite superior opcional do eixo vertical [km]. "
            "Se omitido, usa o topo do dominio."
        ),
    )

    return parser


# ============================================================================
# 14. MAIN
# ============================================================================

def main():
    """
    Ponto de entrada.
    """

    parser = construir_parser()

    args = parser.parse_args()

    # Resolve caminhos em relacao a raiz do repositorio.
    caminho_npz = resolver_caminho(
        args.input
    )

    output_prefix = resolver_caminho(
        args.output_prefix
    )

    # Informacoes iniciais.
    print()
    print("=" * 78)
    print("FIGURA DE EVOLUCAO CONVECTIVA")
    print("=" * 78)

    print(
        f"Raiz do repositorio: {ROOT}"
    )

    print(
        f"Arquivo de entrada:  {caminho_npz}"
    )

    print(
        f"Prefixo de saida:     {output_prefix}"
    )

    print(
        "Snapshots [min]:     "
        + ", ".join(
            f"{valor:g}"
            for valor in args.times
        )
    )

    print("=" * 78)

    gerar_figura_artigo(
        caminho_npz=caminho_npz,
        output_prefix=output_prefix,
        tempos_desejados_min=args.times,
        f3_threshold=float(
            args.f3_threshold
        ),
        cloud_top_threshold=float(
            args.cloud_top_threshold
        ),
        zmax_km=args.zmax_km,
    )


# ============================================================================
# 15. EXECUCAO
# ============================================================================

if __name__ == "__main__":
    main()