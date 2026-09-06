"""
Figura principal do artigo - comparacao 2D entre experimentos do Grupo 2
========================================================================

Gera uma figura comparativa 2x2 para o corpo principal do artigo,
mostrando um instante-chave (por padrao, 20 min) para os casos:

(a) CTRL
(b) DYN_PLUS
(c) WARM com qv fixo / RH variavel + DYN_PLUS
(d) WARM com RH fixa + DYN_PLUS

Cada painel mostra:
    - sombreado de qc + qi + qs + qg [g kg^-1]
    - contornos de w > 0 (solidos) e w < 0 (tracejados)
    - isotermas 0 C, -15 C, -20 C
    - marcadores onde F3 >= limiar na isoterma de -15 C

Uso tipico
----------
Executar a partir de:
    experiments/group2_warming_lightning/

Exemplo:
python plot_figura_comparativa.py
    --ctrl "outputs/group2/final_decomposto/CTRL/resultados_CTRL.npz"
    --dyn "outputs/group2/final_decomposto/DYN_PLUS/resultados_DYN_PLUS.npz"
    --warm-qv "outputs/group2/final_decomposto/WARM_QV_DYN_PLUS/resultados_WARM_QV_DYN_PLUS.npz"
    --warm-rh "outputs/group2/final_decomposto/WARM_RH_DYN_PLUS/resultados_WARM_RH_DYN_PLUS.npz"
    --output-prefix "outputs/group2/final_decomposto/figura_principal_grupo2_20min"
    --time 20

Se o caminho do caso RH fixa for diferente no seu repositorio,
basta ajustar --warm-rh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


# =============================================================================
# CONFIGURACAO GRAFICA
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

FIGSIZE = (7.4, 6.3)  # boa para figura de artigo em largura dupla
CMAP_COND = "jet"

W_POS_LEVELS = [1, 5, 10, 20, 30, 40]
W_NEG_LEVELS = [-30, -20, -10, -5, -1]

DEFAULT_F3_THRESHOLD = 0.02

ISOTHERM_STYLES = {
    0.0:   dict(color="forestgreen", linestyle="-",  linewidth=1.3),
    -15.0: dict(color="firebrick",   linestyle="--", linewidth=1.3),
    -20.0: dict(color="royalblue",   linestyle=":",  linewidth=1.5),
}


# =============================================================================
# CAMINHOS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]


def resolver_caminho(caminho: str | Path) -> Path:
    caminho = Path(caminho).expanduser()
    if caminho.is_absolute():
        return caminho.resolve()
    return (ROOT / caminho).resolve()


# =============================================================================
# LEITURA
# =============================================================================

def carregar_npz(caminho_npz: Path) -> dict:
    if not caminho_npz.exists():
        raise FileNotFoundError(
            f"\nArquivo nao encontrado:\n{caminho_npz}\n"
        )

    with np.load(caminho_npz, allow_pickle=True) as dados:
        return {k: dados[k] for k in dados.files}


def obter_campo(dados: dict, nome: str, obrigatorio: bool = True):
    if nome in dados:
        return np.asarray(dados[nome])

    if obrigatorio:
        campos = "\n".join(sorted(dados.keys()))
        raise KeyError(
            f"\nCampo '{nome}' nao encontrado.\n\n"
            f"Campos disponiveis:\n{campos}\n"
        )
    return None


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def maximo_seguro(campo) -> float:
    campo = np.asarray(campo, dtype=float)
    if campo.size == 0 or np.all(np.isnan(campo)):
        return np.nan
    return float(np.nanmax(campo))


def tempo_para_indice(t_min: np.ndarray, tempo_alvo_min: float) -> int:
    return int(np.argmin(np.abs(np.asarray(t_min, dtype=float) - float(tempo_alvo_min))))


def altura_isoterma_coluna(T_col_K: np.ndarray, z_m: np.ndarray, nivel_c: float) -> float:
    T_col_C = np.asarray(T_col_K, dtype=float) - 273.15
    z_m = np.asarray(z_m, dtype=float)
    alvo = float(nivel_c)

    diferenca = T_col_C - alvo

    for iz in range(len(z_m) - 1):
        d1 = diferenca[iz]
        d2 = diferenca[iz + 1]

        if not (np.isfinite(d1) and np.isfinite(d2)):
            continue

        if d1 == 0.0:
            return float(z_m[iz])

        if d1 * d2 < 0.0:
            T1, T2 = T_col_C[iz], T_col_C[iz + 1]
            z1, z2 = z_m[iz], z_m[iz + 1]

            if T2 == T1:
                return float(0.5 * (z1 + z2))

            frac = (alvo - T1) / (T2 - T1)
            return float(z1 + frac * (z2 - z1))

    return np.nan


def campo_altura_isoterma(T_K: np.ndarray, z_m: np.ndarray, nivel_c: float) -> np.ndarray:
    T_K = np.asarray(T_K, dtype=float)
    nt, nx, _ = T_K.shape

    saida = np.full((nt, nx), np.nan, dtype=float)

    for it in range(nt):
        for ix in range(nx):
            saida[it, ix] = altura_isoterma_coluna(
                T_col_K=T_K[it, ix, :],
                z_m=z_m,
                nivel_c=nivel_c,
            )

    return saida


def tamanhos_marcadores_f3(
    valores_f3: np.ndarray,
    f3_max_global: float,
    threshold: float,
    smin: float = 18.0,
    smax: float = 90.0,
) -> np.ndarray:
    valores_f3 = np.asarray(valores_f3, dtype=float)

    if valores_f3.size == 0:
        return np.asarray([], dtype=float)

    if not np.isfinite(f3_max_global) or f3_max_global <= threshold:
        return np.full(valores_f3.shape, smin, dtype=float)

    frac = (np.maximum(valores_f3 - threshold, 0.0)) / (f3_max_global - threshold)
    frac = np.sqrt(np.clip(frac, 0.0, 1.0))

    return smin + (smax - smin) * frac


def calcular_vmax_global(lista_dados: list[dict], lista_indices: list[int], percentil: float = 99.5) -> float:
    valores = []

    for dados, idx in zip(lista_dados, lista_indices):
        qc = obter_campo(dados, "qc")
        qi = obter_campo(dados, "qi")
        qs = obter_campo(dados, "qs")
        qg = obter_campo(dados, "qg")

        qcond = (qc + qi + qs + qg) * 1000.0  # g/kg
        fatia = np.asarray(qcond[idx], dtype=float).ravel()
        fatia = fatia[np.isfinite(fatia)]
        fatia = fatia[fatia > 0.0]

        if fatia.size > 0:
            valores.append(fatia)

    if not valores:
        return 1.0

    valores = np.concatenate(valores)
    vmax = np.percentile(valores, percentil)

    return max(float(vmax), 0.05)


# =============================================================================
# PLOT DE UM PAINEL
# =============================================================================

def plot_painel(
    ax,
    dados: dict,
    idx: int,
    titulo: str,
    letra: str,
    norm_cond: Normalize,
    f3_threshold: float,
    f3_max_global: float,
    zmax_km: float | None,
):
    t_s = obter_campo(dados, "t_s")
    x_m = obter_campo(dados, "x_m")
    z_m = obter_campo(dados, "z_m")
    T = obter_campo(dados, "T")
    w = obter_campo(dados, "w")
    qc = obter_campo(dados, "qc")
    qi = obter_campo(dados, "qi")
    qs = obter_campo(dados, "qs")
    qg = obter_campo(dados, "qg")
    f3 = obter_campo(dados, "lightning_f3")
    z_minus15 = obter_campo(dados, "lightning_z_minus15_m", obrigatorio=False)

    if z_minus15 is None:
        z_minus15 = campo_altura_isoterma(T, z_m, -15.0)

    t_min = np.asarray(t_s, dtype=float) / 60.0
    x_km = np.asarray(x_m, dtype=float) / 1000.0
    z_km = np.asarray(z_m, dtype=float) / 1000.0

    qcond = (qc + qi + qs + qg) * 1000.0  # g/kg
    qcond_t = np.asarray(qcond[idx], dtype=float)
    w_t = np.asarray(w[idx], dtype=float)
    T_t_C = np.asarray(T[idx], dtype=float) - 273.15
    f3_t = np.asarray(f3[idx], dtype=float)
    z15_t_km = np.asarray(z_minus15[idx], dtype=float) / 1000.0

    # sombreado
    pcm = ax.pcolormesh(
        x_km,
        z_km,
        qcond_t.T,
        shading="auto",
        cmap=CMAP_COND,
        norm=norm_cond,
        rasterized=True,
    )

    # contornos de w positivo
    niveis_pos = [lv for lv in W_POS_LEVELS if np.nanmax(w_t) >= lv] if np.any(np.isfinite(w_t)) else []
    if niveis_pos:
        cpos = ax.contour(
            x_km,
            z_km,
            w_t.T,
            levels=niveis_pos,
            colors="magenta",
            linewidths=1.0,
            linestyles="-",
        )
        ax.clabel(cpos, inline=True, fontsize=6.0, fmt=lambda v: f"{v:g}")

    # contornos de w negativo
    if np.any(np.isfinite(w_t)) and np.nanmin(w_t) < 0.0:
        niveis_neg = [lv for lv in W_NEG_LEVELS if np.nanmin(w_t) <= lv]
        if niveis_neg:
            cneg = ax.contour(
                x_km,
                z_km,
                w_t.T,
                levels=niveis_neg,
                colors="magenta",
                linewidths=1.0,
                linestyles="--",
            )
            ax.clabel(cneg, inline=True, fontsize=6.0, fmt=lambda v: f"{v:g}")

    # isotermas
    Tmin = np.nanmin(T_t_C) if np.any(np.isfinite(T_t_C)) else np.nan
    Tmax = np.nanmax(T_t_C) if np.any(np.isfinite(T_t_C)) else np.nan

    for nivel, estilo in ISOTHERM_STYLES.items():
        if np.isfinite(Tmin) and np.isfinite(Tmax) and Tmin <= nivel <= Tmax:
            ax.contour(
                x_km,
                z_km,
                T_t_C.T,
                levels=[nivel],
                colors=[estilo["color"]],
                linestyles=[estilo["linestyle"]],
                linewidths=[estilo["linewidth"]],
            )

    # marcadores F3
    mascara = (
        np.isfinite(f3_t)
        & np.isfinite(z15_t_km)
        & (f3_t >= f3_threshold)
    )

    if np.any(mascara):
        tamanhos = tamanhos_marcadores_f3(
            f3_t[mascara],
            f3_max_global=f3_max_global,
            threshold=f3_threshold,
        )

        ax.scatter(
            x_km[mascara],
            z15_t_km[mascara],
            s=tamanhos,
            facecolor="gold",
            edgecolor="black",
            linewidth=0.45,
            alpha=0.92,
            zorder=8,
        )

    tempo_real = float(t_min[idx])
    wmax = maximo_seguro(w_t)
    f3max = maximo_seguro(f3_t)

    ax.set_title(
        f"({letra}) {titulo}\n$t \\approx$ {tempo_real:.0f} min",
        loc="left",
        pad=4,
        fontweight="bold",
    )

    texto = (
        rf"$w_{{max}}$ = {wmax:.1f} m s$^{{-1}}$" "\n"
        rf"$F_{{3,max}}$ = {f3max:.3f}"
    )

    ax.text(
        0.97,
        0.97,
        texto,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.65", alpha=0.85),
    )

    ax.set_xlim(x_km[0], x_km[-1])

    if zmax_km is None:
        ax.set_ylim(0.0, z_km[-1])
    else:
        ax.set_ylim(0.0, min(float(zmax_km), float(z_km[-1])))

    ax.tick_params(axis="both", pad=2.0)

    return pcm


# =============================================================================
# FIGURA COMPLETA
# =============================================================================

def gerar_figura_principal(
    caminho_ctrl: Path,
    caminho_dyn: Path,
    caminho_warm_qv: Path,
    caminho_warm_rh: Path,
    output_prefix: Path,
    tempo_alvo_min: float,
    f3_threshold: float = DEFAULT_F3_THRESHOLD,
    zmax_km: float | None = None,
):
    casos = [
        ("CTRL", caminho_ctrl),
        ("DYN_PLUS", caminho_dyn),
        ("WARM ($q_v$ fixo; RH variavel) + DYN_PLUS", caminho_warm_qv),
        ("WARM (RH fixa) + DYN_PLUS", caminho_warm_rh),
    ]

    lista_dados = []
    lista_indices = []

    for nome, caminho in casos:
        dados = carregar_npz(caminho)
        t_s = obter_campo(dados, "t_s")
        t_min = np.asarray(t_s, dtype=float) / 60.0
        idx = tempo_para_indice(t_min, tempo_alvo_min)

        lista_dados.append(dados)
        lista_indices.append(idx)

    # escala compartilhada do condensado
    vmax_global = calcular_vmax_global(lista_dados, lista_indices, percentil=99.5)
    norm_cond = Normalize(vmin=0.0, vmax=vmax_global)

    # F3 max global para escalar os marcadores igualmente em todos os paineis
    f3_max_global = np.nanmax([
        np.nanmax(obter_campo(d, "lightning_f3"))
        for d in lista_dados
    ])

    fig = plt.figure(figsize=FIGSIZE)

    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[0.18, 1.0, 1.0],
        left=0.08,
        right=0.94,
        bottom=0.08,
        top=0.96,
        wspace=0.12,
        hspace=0.38,
    )

    # legenda superior
    ax_leg = fig.add_subplot(gs[0, :2])
    ax_leg.axis("off")

    handles = [
        Line2D([0], [0], color="magenta", lw=1.1, linestyle="-",  label=r"$w>0$ (m s$^{-1}$)"),
        Line2D([0], [0], color="magenta", lw=1.1, linestyle="--", label=r"$w<0$ (m s$^{-1}$)"),
        Line2D([0], [0], color="forestgreen", lw=1.4, linestyle="-",  label="0 °C"),
        Line2D([0], [0], color="firebrick",   lw=1.4, linestyle="--", label="-15 °C"),
        Line2D([0], [0], color="royalblue",   lw=1.5, linestyle=":",  label="-20 °C"),
        Line2D([0], [0], marker="o", markersize=6.5, markerfacecolor="gold",
               markeredgecolor="black", linestyle="None",
               label=rf"$F_3 \geq {f3_threshold:g}$ na isoterma de -15 °C"),
    ]

    ax_leg.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.4,
        handletextpad=0.6,
    )

    # eixos
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[2, 1], sharex=ax1, sharey=ax1)
    cax = fig.add_subplot(gs[1:, 2])

    axes = [ax1, ax2, ax3, ax4]
    letras = ["a", "b", "c", "d"]

    pcm = None
    for ax, letra, (titulo, _), dados, idx in zip(
        axes, letras, casos, lista_dados, lista_indices
    ):
        pcm = plot_painel(
            ax=ax,
            dados=dados,
            idx=idx,
            titulo=titulo,
            letra=letra,
            norm_cond=norm_cond,
            f3_threshold=f3_threshold,
            f3_max_global=f3_max_global,
            zmax_km=zmax_km,
        )

    # labels compartilhados
    ax1.set_ylabel("Altura (km)")
    ax3.set_ylabel("Altura (km)")

    # O eixo x e rotulado apenas na linha inferior.
    # Isso evita conflito visual com os titulos dos paineis (c) e (d).
    ax3.set_xlabel(r"$x$ (km)")
    ax4.set_xlabel(r"$x$ (km)")

    # A linha superior compartilha o eixo x, mas nao exibe os rotulos.
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    ax2.tick_params(labelleft=False)
    ax4.tick_params(labelleft=False)

    # colorbar
    cbar = fig.colorbar(pcm, cax=cax, orientation="vertical", extend="max")
    cbar.set_label(r"$q_c + q_i + q_s + q_g$ (g kg$^{-1}$)", rotation=270, labelpad=15)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.tick_right()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    caminho_png = output_prefix.with_suffix(".png")
    caminho_pdf = output_prefix.with_suffix(".pdf")

    fig.savefig(caminho_png)
    fig.savefig(caminho_pdf)
    plt.close(fig)

    print()
    print("=" * 80)
    print("FIGURA PRINCIPAL CONCLUIDA")
    print("=" * 80)
    print(f"CTRL:         {caminho_ctrl}")
    print(f"DYN_PLUS:     {caminho_dyn}")
    print(f"WARM qv:      {caminho_warm_qv}")
    print(f"WARM RH fixa: {caminho_warm_rh}")
    print(f"Tempo alvo:   {tempo_alvo_min:g} min")
    print(f"PNG:          {caminho_png}")
    print(f"PDF:          {caminho_pdf}")
    print(f"vmax comum:   {vmax_global:.3f} g/kg")
    print("=" * 80)


# =============================================================================
# ARGUMENTOS
# =============================================================================

def construir_parser():
    parser = argparse.ArgumentParser(
        description="Gera a figura comparativa principal do Grupo 2 para o artigo."
    )

    parser.add_argument(
        "--ctrl",
        type=str,
        default="outputs/group2/CTRL/resultados_CTRL.npz",
        help="Arquivo NPZ do caso CTRL."
    )

    parser.add_argument(
        "--dyn",
        type=str,
        default="outputs/group2/DYN_PLUS/resultados_DYN_PLUS.npz",
        help="Arquivo NPZ do caso DYN_PLUS."
    )

    parser.add_argument(
        "--warm-qv",
        type=str,
        default="outputs/group2/WARM_DYN_PLUS/resultados_WARM_DYN_PLUS.npz",
        help="Arquivo NPZ do caso WARM com qv fixo / RH variavel."
    )

    parser.add_argument(
        "--warm-rh",
        type=str,
        default="outputs/group2/rh_fixa/WARM_DYN_PLUS/resultados_WARM_DYN_PLUS.npz",
        help="Arquivo NPZ do caso WARM com RH fixa."
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/group2/figura_principal_grupo2_20min",
        help="Prefixo da saida (sem extensao)."
    )

    parser.add_argument(
        "--time",
        type=float,
        default=20.0,
        help="Tempo alvo em minutos. Padrao: 20."
    )

    parser.add_argument(
        "--f3-threshold",
        type=float,
        default=DEFAULT_F3_THRESHOLD,
        help=f"Limiar de referencia de F3. Padrao: {DEFAULT_F3_THRESHOLD:g}."
    )

    parser.add_argument(
        "--zmax-km",
        type=float,
        default=None,
        help="Limite superior opcional do eixo vertical em km."
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = construir_parser()
    args = parser.parse_args()

    caminho_ctrl = resolver_caminho(args.ctrl)
    caminho_dyn = resolver_caminho(args.dyn)
    caminho_warm_qv = resolver_caminho(args.warm_qv)
    caminho_warm_rh = resolver_caminho(args.warm_rh)
    output_prefix = resolver_caminho(args.output_prefix)

    print()
    print("=" * 80)
    print("GERANDO FIGURA PRINCIPAL DO GRUPO 2")
    print("=" * 80)
    print(f"Raiz do repositorio: {ROOT}")
    print(f"CTRL:                {caminho_ctrl}")
    print(f"DYN_PLUS:            {caminho_dyn}")
    print(f"WARM qv:             {caminho_warm_qv}")
    print(f"WARM RH fixa:        {caminho_warm_rh}")
    print(f"Tempo alvo:          {args.time:g} min")
    print(f"Saida:               {output_prefix}")
    print("=" * 80)

    gerar_figura_principal(
        caminho_ctrl=caminho_ctrl,
        caminho_dyn=caminho_dyn,
        caminho_warm_qv=caminho_warm_qv,
        caminho_warm_rh=caminho_warm_rh,
        output_prefix=output_prefix,
        tempo_alvo_min=float(args.time),
        f3_threshold=float(args.f3_threshold),
        zmax_km=args.zmax_km,
    )


if __name__ == "__main__":
    main()