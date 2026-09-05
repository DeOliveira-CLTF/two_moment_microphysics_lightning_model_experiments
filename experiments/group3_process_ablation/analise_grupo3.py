# -*- coding: utf-8 -*-
"""
analise_grupo3.py
=====================

Le as saidas de `executar_grupo3.py` (outputs/group3/<CASO>/resultados_<CASO>.npz)
e monta a Tabela 2 do plano (secao 4.1): metricas de gelo, neve, graupel,
precipitacao, w_max, McCaul e LPI*, normalizadas por CASO/CTRL.

DEFINICAO DE METRICA ADOTADA (unica e consistente entre todos os casos,
conforme exigido na secao 4.1: "sem misturar definicoes entre casos"):

    - Gelo, neve, graupel : MAXIMO TEMPORAL da massa integrada no dominio
      2D (kg/m, por unidade de profundidade em y), isto e,
      max_t [ sum_x sum_z rho(z) * q_x(t,x,z) * dx * dz ].
      (Alternativa documentada no plano: integral no periodo convectivo;
      trocar por `metodo="integral_tempo"` abaixo se preferirem essa
      definicao -- mas ai TODAS as linhas da Tabela 2 devem usar a mesma
      troca.)

    - Precipitacao : PROXY DIAGNOSTICO, nao a precipitacao final validada.
      O plano (secoes 1.2, 2.1 e 6.1) e explicito: o acumulador de fluxo
      de precipitacao que cruza a base ainda nao esta consolidado no
      orcamento de agua do modelo 2D. Aqui, "Precip." e estimada como o
      fluxo de massa de qr+qs+qg cruzando o nivel mais baixo do dominio
      (rho*q*Vt, Vt via campo_Vt_* do nucleo comum), integrado no tempo e
      em x. Use esta coluna apenas para RANKING relativo entre casos, e
      documente a ressalva do plano ao reportar a Tabela 2 final.

    - w_max : maximo, em todo o dominio (x,z) e todo o tempo salvo, de w.

    - McCaul (F3) e LPI* : maximo, em todo o dominio (x) e todo o tempo
      salvo, da serie F3(t,x) / LPI*(t,x) calculada por
      `diagnosticos_relampago_grupo3.py` (ver ressalva de integridade
      cientifica nesse modulo sobre a fonte de verdade ser
      `lightning/diagnosticos_2d.py` do repositorio, quando disponivel).

Uso:
    python experiments/group3_process_ablation/analise_grupo3.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from dinamica_2d import campo_Vt_chuva, campo_Vt_neve, campo_Vt_graupel
from configuracao_casos_grupo3 import ORDEM_CASOS
from diagnosticos_relampago_grupo3 import diagnosticos_2d_para_caso

# numpy >= 2.0 renomeou trapz -> trapezoid.
_trapz = getattr(np, "trapezoid", None) or np.trapz

OUTPUT_ROOT = ROOT / "outputs" / "group3"


def _massa_integrada_max(frames, campo, rho0_1d, dx, dz):
    """max_t [ sum_x sum_z rho(z) * campo(t,x,z) * dx * dz ]  (kg/m)."""
    q = frames[campo]  # (nt, nx, nz)
    massa_t = np.einsum("z,txz->t", rho0_1d, q) * dx * dz
    return float(np.max(massa_t))


def _precip_proxy(frames, rho0_1d, dx, dt_salvo, k_baixo=1):
    """Fluxo diagnostico de qr+qs+qg cruzando o nivel k_baixo, integrado
    no tempo (usando o intervalo entre saidas salvas) e em x. Retorna em
    kg (massa total que cruzou a base durante a simulacao, por unidade de
    profundidade em y) -- NAO calibrado como lamina de precipitacao em
    mm; ver ressalva no docstring do modulo."""
    rho_k = rho0_1d[k_baixo]
    qr, Nr = frames["qr"][:, :, k_baixo], frames["Nr"][:, :, k_baixo]
    qs, Ns = frames["qs"][:, :, k_baixo], frames["Ns"][:, :, k_baixo]
    qg, Ng = frames["qg"][:, :, k_baixo], frames["Ng"][:, :, k_baixo]

    Vtq_r, _ = campo_Vt_chuva(qr, Nr, rho_k)
    Vtq_s, _ = campo_Vt_neve(qs, Ns, rho_k)
    Vtq_g, _ = campo_Vt_graupel(qg, Ng, rho_k)

    fluxo_txx = rho_k * (qr * Vtq_r + qs * Vtq_s + qg * Vtq_g)  # kg m^-2 s^-1, (nt, nx)
    fluxo_t = np.sum(fluxo_txx, axis=1) * dx  # kg/m/s, por instante salvo
    return float(_trapz(fluxo_t, dx=dt_salvo))


def carregar_caso(caso):
    caminho = OUTPUT_ROOT / caso / f"resultados_{caso}.npz"
    if not caminho.exists():
        return None
    dados = np.load(caminho)

    frames = {
        nome: dados[nome]
        for nome in ("T", "qv", "qc", "Nc", "qr", "Nr", "qi", "Ni",
                     "qs", "Ns", "qg", "Ng", "w", "u")
    }
    z = dados["z_m"]
    x = dados["x_m"]
    rho0_1d = dados["rho0_1d"]
    t_s = dados["t_s"]

    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    dt_salvo = float(t_s[1] - t_s[0]) if len(t_s) > 1 else 1.0

    metricas = {
        "gelo_kgm": _massa_integrada_max(frames, "qi", rho0_1d, dx, dz),
        "neve_kgm": _massa_integrada_max(frames, "qs", rho0_1d, dx, dz),
        "graupel_kgm": _massa_integrada_max(frames, "qg", rho0_1d, dx, dz),
        "precip_proxy_kgm": _precip_proxy(frames, rho0_1d, dx, dt_salvo),
        "w_max_ms": float(np.max(frames["w"])),
    }

    F3_txx, LPI_txx = diagnosticos_2d_para_caso(frames, z, rho0_1d)
    metricas["mccaul_F3_max"] = float(np.nanmax(F3_txx)) if np.any(~np.isnan(F3_txx)) else np.nan
    metricas["lpi_estrela_max"] = float(np.nanmax(LPI_txx)) if np.any(~np.isnan(LPI_txx)) else np.nan

    return metricas


def montar_tabela2():
    resultados = {}
    for caso in ORDEM_CASOS:
        m = carregar_caso(caso)
        if m is None:
            print(f"[aviso] sem resultados salvos para {caso} "
                  f"(rode executar_grupo3.py primeiro)")
            continue
        resultados[caso] = m

    if "CTRL" not in resultados:
        raise RuntimeError(
            "CTRL nao encontrado em outputs/group3/ -- necessario para "
            "normalizar a Tabela 2."
        )
    ctrl = resultados["CTRL"]

    colunas = [
        ("gelo_kgm", "Gelo"),
        ("neve_kgm", "Neve"),
        ("graupel_kgm", "Graupel"),
        ("precip_proxy_kgm", "Precip.*"),
        ("w_max_ms", "w_max"),
        ("mccaul_F3_max", "McCaul"),
        ("lpi_estrela_max", "LPI*"),
    ]

    linhas = []
    for caso in ORDEM_CASOS:
        if caso not in resultados:
            continue
        m = resultados[caso]
        linha = {"Caso": caso}
        for chave, rotulo in colunas:
            base = ctrl[chave]
            valor = m[chave]
            if base == 0 or np.isnan(base) or np.isnan(valor):
                linha[rotulo] = np.nan
            else:
                linha[rotulo] = valor / base
        linhas.append(linha)

    return linhas, colunas, resultados


def imprimir_tabela(linhas, colunas):
    rotulos = ["Caso"] + [r for _, r in colunas]
    larguras = {r: max(len(r), 10) for r in rotulos}

    header = " | ".join(r.ljust(larguras[r]) for r in rotulos)
    print(header)
    print("-" * len(header))
    for linha in linhas:
        celulas = [linha["Caso"].ljust(larguras["Caso"])]
        for _, rotulo in colunas:
            v = linha[rotulo]
            txt = "nan" if (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"
            celulas.append(txt.rjust(larguras[rotulo]))
        print(" | ".join(celulas))
    print("\n* Precip. e um proxy diagnostico (fluxo na base), nao a "
          "precipitacao final validada -- ver secao 6.1 do plano.")


def figura3_opcional(linhas, saida_png=None):
    """Grafico de barras com a variacao percentual de w_max, McCaul e
    LPI* para cada ablacao em relacao ao CTRL (Figura opcional do
    Grupo 3, secao 4.1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    casos_ablacao = [l for l in linhas if l["Caso"] != "CTRL"]
    nomes = [l["Caso"] for l in casos_ablacao]
    metricas = ["w_max", "McCaul", "LPI*"]

    x = np.arange(len(nomes))
    largura = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, met in enumerate(metricas):
        valores = [(l[met] - 1.0) * 100.0 if not np.isnan(l[met]) else 0.0
                   for l in casos_ablacao]
        ax.bar(x + (i - 1) * largura, valores, width=largura, label=met)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, rotation=20, ha="right")
    ax.set_ylabel("Variacao percentual em relacao ao CTRL (%)")
    ax.set_title("Grupo 3 - Ranking de sensibilidade por processo removido")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    saida_png = saida_png or (OUTPUT_ROOT / "fig_grupo3_ranking.png")
    plt.savefig(saida_png, dpi=140)
    plt.close()
    print(f"\nFigura opcional salva em: {saida_png}")


if __name__ == "__main__":
    linhas, colunas, _ = montar_tabela2()
    imprimir_tabela(linhas, colunas)

    if len(linhas) > 1:
        figura3_opcional(linhas)

    with open(OUTPUT_ROOT / "tabela2_grupo3.csv", "w") as f:
        rotulos = ["Caso"] + [r for _, r in colunas]
        f.write(",".join(rotulos) + "\n")
        for linha in linhas:
            f.write(",".join(str(linha[r]) for r in rotulos) + "\n")
    print(f"Tabela 2 salva em: {OUTPUT_ROOT / 'tabela2_grupo3.csv'}")
