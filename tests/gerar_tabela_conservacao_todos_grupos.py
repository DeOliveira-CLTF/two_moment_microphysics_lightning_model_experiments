# -*- coding: utf-8 -*-
"""
gerar_tabela_conservacao_todos_grupos.py
========================================

Gera uma tabela unica de diagnostico de conservacao de agua para os
experimentos finais dos Grupos 1, 2 e 3.

O script le diretamente os arquivos NPZ ja gerados pelas simulacoes.
Nao e necessario rerodar os experimentos.

Dois orcamentos sao calculados:

1) Orcamento Boussinesq
   --------------------
   E o diagnostico matematicamente mais coerente com o nucleo atual,
   que resolve um campo de velocidade incompressivel (div v = 0):

       I_B(t) = integral_A qt dA

   onde

       qt = qv + qc + qr + qi + qs + qg.

   A agua que sai pela base por sedimentacao e adicionada de volta ao
   inventario por um fluxo diagnostico:

       P_B(t) = integral_0^t integral_x
                (qr Vtr + qi Vti + qs Vts + qg Vtg) dx dt.

   Assim:

       B_B(t) = I_B(t) + P_B(t).

2) Orcamento ponderado por rho0
   ----------------------------
   E um diagnostico fisico aproximado em kg/m, por unidade de profundidade
   transversal do modelo 2D:

       M(t) = integral_A rho0(z) qt dA

       P_rho(t) = integral_0^t integral_x rho0
                  (qr Vtr + qi Vti + qs Vts + qg Vtg) dx dt

       B_rho(t) = M(t) + P_rho(t).

IMPORTANTE
----------
A precipitacao e reconstruida offline a partir dos tempos SALVOS no NPZ.
Como as simulacoes normalmente armazenam saidas a cada 300 s, P_B e P_rho
sao proxies de fechamento do orcamento, obtidos por integracao trapezoidal
dos fluxos salvos. Eles nao substituem um acumulador de precipitacao
calculado a cada passo de tempo dentro do nucleo.

O script inclui as quatro categorias que sedimentam no nucleo atual:
chuva (qr), gelo de nuvem (qi), neve (qs) e graupel (qg).

Por padrao usa k_base=1, isto e, o primeiro nivel interno acima da fronteira,
evitando utilizar diretamente a celula de contorno k=0.

Saidas
------
Por padrao, em:

    outputs/conservacao_massa/

sao gerados:

    tabela_conservacao_massa_todos_grupos.csv
    tabela_conservacao_massa_todos_grupos.tex
    nota_metodologica_conservacao.txt

Uso
---
A partir da raiz do repositorio:

    python tests/gerar_tabela_conservacao_todos_grupos.py

ou, se salvar o arquivo na raiz:

    python gerar_tabela_conservacao_todos_grupos.py

Somente alguns grupos:

    python tests/gerar_tabela_conservacao_todos_grupos.py --grupos G1 G2

Alterar o nivel usado para o fluxo na base:

    python tests/gerar_tabela_conservacao_todos_grupos.py --k-base 1

Adicionar um limiar apenas para classificacao diagnostica:

    python tests/gerar_tabela_conservacao_todos_grupos.py \
        --tolerancia-pct 10

O limiar NAO e definido por padrao, para evitar classificar arbitrariamente
os experimentos como "conservados" ou "nao conservados".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# 1. RAIZ DO REPOSITORIO
# ============================================================================


def encontrar_raiz_repositorio() -> Path:
    """Procura a raiz do repositorio a partir da localizacao deste script."""

    arquivo = Path(__file__).resolve()
    candidatos = [arquivo.parent, *arquivo.parents]

    for pasta in candidatos:
        if (
            (pasta / "dinamica_2d").is_dir()
            and (pasta / "experiments").is_dir()
        ):
            return pasta

    raise RuntimeError(
        "Nao foi possivel localizar a raiz do repositorio. "
        "Coloque este script dentro do repositorio "
        "two_moment_microphysics_lightning_model_experiments."
    )


ROOT = encontrar_raiz_repositorio()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# 2. API DO NUCLEO
# ============================================================================

from dinamica_2d import (
    campo_Vt_chuva,
    campo_Vt_gelo,
    campo_Vt_neve,
    campo_Vt_graupel,
)


# ============================================================================
# 3. ORDEM PREFERENCIAL DOS CASOS
# ============================================================================

ORDEM_GRUPOS = ("G1", "G2", "G3")

ORDEM_G1 = (
    "N_LOW",
    "CTRL",
    "N_HIGH",
)

ORDEM_G2 = (
    "CTRL",
    "DYN_PLUS",
    "WARM_QV",
    "WARM_QV_DYN_PLUS",
    "WARM_RH",
    "WARM_RH_DYN_PLUS",
)

ORDEM_G3 = (
    "CTRL",
    "SEM-NUC",
    "SEM-DEP",
    "SEM-CONG-NUV",
    "SEM-CONG-CHUVA",
    "SEM-RIMING",
    "SEM-HM",
    "SEM-GELO-NEVE",
    "SEM-RIMING_HM",
    "SEM-CONG-CHUVA_RIMING",
    "SEM-RIMING-HM",
    "SEM-CONG-CHUVA-RIMING",
)


# ============================================================================
# 4. UTILITARIOS
# ============================================================================


def ler_commit(pasta_caso: Path) -> str:
    """Le commit.txt quando existir."""

    candidatos = [
        pasta_caso / "commit.txt",
        pasta_caso.parent / "commit.txt",
    ]

    for caminho in candidatos:
        if caminho.exists():
            texto = caminho.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()

            if texto:
                return texto

    return ""


def inferir_caso(caminho: Path) -> str:
    """Usa preferencialmente o nome da pasta do caso."""

    return caminho.parent.name


def ordenar_casos(grupo: str, caminhos: list[Path]) -> list[Path]:
    """Ordena os casos de forma consistente com o desenho experimental."""

    if grupo == "G1":
        ordem = ORDEM_G1
    elif grupo == "G2":
        ordem = ORDEM_G2
    elif grupo == "G3":
        ordem = ORDEM_G3
    else:
        ordem = ()

    indice = {nome: i for i, nome in enumerate(ordem)}

    return sorted(
        caminhos,
        key=lambda p: (
            indice.get(inferir_caso(p), 10_000),
            inferir_caso(p),
            str(p),
        ),
    )


def remover_duplicatas(caminhos: list[Path]) -> list[Path]:
    vistos = set()
    saida = []

    for caminho in caminhos:
        chave = caminho.resolve()

        if chave in vistos:
            continue

        vistos.add(chave)
        saida.append(caminho)

    return saida


def acumulada_trapezoidal(
    fluxo_t: np.ndarray,
    t_s: np.ndarray,
) -> np.ndarray:
    """Integral acumulada trapezoidal para tempos possivelmente nao uniformes."""

    fluxo_t = np.asarray(fluxo_t, dtype=float)
    t_s = np.asarray(t_s, dtype=float)

    if fluxo_t.ndim != 1:
        raise ValueError("fluxo_t deve ser unidimensional")

    if t_s.ndim != 1:
        raise ValueError("t_s deve ser unidimensional")

    if len(fluxo_t) != len(t_s):
        raise ValueError("fluxo_t e t_s devem ter o mesmo tamanho")

    acumulada = np.zeros_like(fluxo_t, dtype=float)

    if len(t_s) <= 1:
        return acumulada

    dt = np.diff(t_s)

    if np.any(dt <= 0.0):
        raise ValueError("Os tempos salvos devem ser estritamente crescentes.")

    incremento = 0.5 * (fluxo_t[1:] + fluxo_t[:-1]) * dt
    acumulada[1:] = np.cumsum(incremento)

    return acumulada


def erro_relativo_percentual(
    serie: np.ndarray,
    referencia: float,
) -> np.ndarray:
    referencia = float(referencia)
    denominador = max(abs(referencia), 1.0e-30)

    return (
        (np.asarray(serie, dtype=float) - referencia)
        / denominador
        * 100.0
    )


def formatar_numero_csv(valor):
    if valor is None:
        return ""

    if isinstance(valor, (float, np.floating)):
        if not np.isfinite(valor):
            return ""
        return float(valor)

    return valor


def latex_escape(texto: str) -> str:
    return (
        str(texto)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


# ============================================================================
# 5. DESCOBERTA DOS EXPERIMENTOS FINAIS
# ============================================================================


def descobrir_grupo1() -> list[Path]:
    """
    Saidas finais do Grupo 1.

    Estrutura esperada:
        outputs/group1/experimentos_B10/<CASO>/resultados_<CASO>.npz
    """

    base = ROOT / "outputs" / "group1" / "experimentos_B10"

    if not base.exists():
        return []

    caminhos = []

    for pasta in base.iterdir():
        if not pasta.is_dir():
            continue

        if pasta.name.lower() == "testes":
            continue

        caminhos.extend(sorted(pasta.glob("resultados_*.npz")))

    return ordenar_casos("G1", remover_duplicatas(caminhos))


def descobrir_grupo2() -> list[Path]:
    """
    Prioriza a matriz final decomposta de seis casos.

    Se ela nao existir, procura tambem nas pastas finais separadas por
    tratamento de umidade.
    """

    candidatos_base = [
        ROOT / "outputs" / "group2" / "final_decomposto",
        ROOT / "outputs" / "group2" / "final_qv_fixo_rh_variavel",
        ROOT / "outputs" / "group2" / "final_rh_fixa",
    ]

    encontrados_por_caso: dict[str, Path] = {}

    for base in candidatos_base:
        if not base.exists():
            continue

        for pasta in sorted(base.iterdir()):
            if not pasta.is_dir():
                continue

            arquivos = sorted(pasta.glob("resultados_*.npz"))

            if not arquivos:
                continue

            caso = pasta.name
            encontrados_por_caso.setdefault(caso, arquivos[0])

    caminhos = list(encontrados_por_caso.values())
    return ordenar_casos("G2", caminhos)


def descobrir_grupo3() -> list[Path]:
    """
    Saidas finais do Grupo 3.

    Estrutura esperada:
        outputs/group3/<CASO>/resultados_<CASO>.npz
    """

    base = ROOT / "outputs" / "group3"

    if not base.exists():
        return []

    caminhos = []

    for pasta in sorted(base.iterdir()):
        if not pasta.is_dir():
            continue

        arquivos = sorted(pasta.glob("resultados_*.npz"))

        if arquivos:
            caminhos.append(arquivos[0])

    return ordenar_casos("G3", remover_duplicatas(caminhos))


def descobrir_experimentos(
    grupos: tuple[str, ...],
) -> list[tuple[str, Path]]:
    saida = []

    for grupo in grupos:
        if grupo == "G1":
            caminhos = descobrir_grupo1()
        elif grupo == "G2":
            caminhos = descobrir_grupo2()
        elif grupo == "G3":
            caminhos = descobrir_grupo3()
        else:
            raise ValueError(f"Grupo desconhecido: {grupo}")

        for caminho in caminhos:
            saida.append((grupo, caminho))

    return saida


# ============================================================================
# 6. LEITURA DOS CAMPOS
# ============================================================================

CAMPOS_HIDROMETEOROS = (
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
)


def obter_qv_total(dados) -> np.ndarray:
    """
    Os NPZ atuais salvam qv total diretamente.

    Como fallback, aceita qvp + qv_env_1d.
    """

    if "qv" in dados.files:
        return np.asarray(dados["qv"], dtype=float)

    if "qvp" in dados.files and "qv_env_1d" in dados.files:
        qvp = np.asarray(dados["qvp"], dtype=float)
        qv_env = np.asarray(dados["qv_env_1d"], dtype=float)
        return qv_env[None, None, :] + qvp

    raise KeyError(
        "Nao foi encontrado qv nem a combinacao qvp + qv_env_1d."
    )


def validar_dimensoes(
    nome: str,
    campo: np.ndarray,
    nt: int,
    nx: int,
    nz: int,
):
    esperado = (nt, nx, nz)

    if campo.shape != esperado:
        raise ValueError(
            f"Campo {nome!r} possui shape={campo.shape}, "
            f"mas era esperado {esperado}."
        )


# ============================================================================
# 7. ORCAMENTO DE AGUA
# ============================================================================


def calcular_fluxo_sedimentacao_base(
    dados,
    rho0_1d: np.ndarray,
    dx: float,
    k_base: int,
):
    """
    Calcula os fluxos sedimentantes na base usando as mesmas funcoes
    de velocidade terminal expostas pelo nucleo.

    Retorna:
        fluxo_boussinesq_t : integral_x sum(q_j Vt_j) dx
        fluxo_rho_t        : integral_x rho0 sum(q_j Vt_j) dx
        componentes        : fluxos rho0 integrados por especie
    """

    rho_k = float(rho0_1d[k_base])

    qr = np.asarray(dados["qr"][:, :, k_base], dtype=float)
    Nr = np.asarray(dados["Nr"][:, :, k_base], dtype=float)

    qi = np.asarray(dados["qi"][:, :, k_base], dtype=float)
    Ni = np.asarray(dados["Ni"][:, :, k_base], dtype=float)

    qs = np.asarray(dados["qs"][:, :, k_base], dtype=float)
    Ns = np.asarray(dados["Ns"][:, :, k_base], dtype=float)

    qg = np.asarray(dados["qg"][:, :, k_base], dtype=float)
    Ng = np.asarray(dados["Ng"][:, :, k_base], dtype=float)

    Vt_r, _ = campo_Vt_chuva(qr, Nr, rho_k)
    Vt_i, _ = campo_Vt_gelo(qi, Ni, rho_k)
    Vt_s, _ = campo_Vt_neve(qs, Ns, rho_k)
    Vt_g, _ = campo_Vt_graupel(qg, Ng, rho_k)

    fluxo_r_x = qr * Vt_r
    fluxo_i_x = qi * Vt_i
    fluxo_s_x = qs * Vt_s
    fluxo_g_x = qg * Vt_g

    fluxo_total_x = fluxo_r_x + fluxo_i_x + fluxo_s_x + fluxo_g_x

    fluxo_boussinesq_t = np.sum(fluxo_total_x, axis=1) * dx
    fluxo_rho_t = rho_k * fluxo_boussinesq_t

    componentes_rho = {
        "chuva": rho_k * np.sum(fluxo_r_x, axis=1) * dx,
        "gelo": rho_k * np.sum(fluxo_i_x, axis=1) * dx,
        "neve": rho_k * np.sum(fluxo_s_x, axis=1) * dx,
        "graupel": rho_k * np.sum(fluxo_g_x, axis=1) * dx,
    }

    return fluxo_boussinesq_t, fluxo_rho_t, componentes_rho


def calcular_conservacao_arquivo(
    grupo: str,
    caminho: Path,
    k_base: int,
    tolerancia_pct: float | None,
) -> dict:
    caso = inferir_caso(caminho)

    with np.load(caminho, allow_pickle=False) as dados:
        obrigatorios = {
            "t_s",
            "x_m",
            "z_m",
            "rho0_1d",
            "qc",
            "qr",
            "Nr",
            "qi",
            "Ni",
            "qs",
            "Ns",
            "qg",
            "Ng",
        }

        faltantes = sorted(obrigatorios.difference(dados.files))

        if faltantes:
            raise KeyError(
                "Campos ausentes no NPZ: " + ", ".join(faltantes)
            )

        t_s = np.asarray(dados["t_s"], dtype=float)
        x_m = np.asarray(dados["x_m"], dtype=float)
        z_m = np.asarray(dados["z_m"], dtype=float)
        rho0_1d = np.asarray(dados["rho0_1d"], dtype=float)

        if len(t_s) < 1:
            raise ValueError("O arquivo nao possui tempos salvos.")

        if len(x_m) < 2 or len(z_m) < 2:
            raise ValueError("A grade precisa ter pelo menos dois pontos.")

        nt = len(t_s)
        nx = len(x_m)
        nz = len(z_m)

        if k_base < 0 or k_base >= nz:
            raise ValueError(
                f"k_base={k_base} fora do intervalo [0, {nz - 1}]."
            )

        dx = float(np.median(np.diff(x_m)))
        dz = float(np.median(np.diff(z_m)))

        qv = obter_qv_total(dados)

        campos = {"qv": qv}

        for nome in CAMPOS_HIDROMETEOROS:
            campos[nome] = np.asarray(dados[nome], dtype=float)

        for nome, campo in campos.items():
            validar_dimensoes(nome, campo, nt, nx, nz)

        qt = (
            campos["qv"]
            + campos["qc"]
            + campos["qr"]
            + campos["qi"]
            + campos["qs"]
            + campos["qg"]
        )

        inventario_b_t = np.sum(qt, axis=(1, 2)) * dx * dz

        massa_rho_t = (
            np.sum(
                qt * rho0_1d[None, None, :],
                axis=(1, 2),
            )
            * dx
            * dz
        )

        (
            fluxo_b_t,
            fluxo_rho_t,
            componentes_rho_t,
        ) = calcular_fluxo_sedimentacao_base(
            dados=dados,
            rho0_1d=rho0_1d,
            dx=dx,
            k_base=k_base,
        )

        precip_b_t = acumulada_trapezoidal(fluxo_b_t, t_s)
        precip_rho_t = acumulada_trapezoidal(fluxo_rho_t, t_s)

        precip_componentes = {
            nome: acumulada_trapezoidal(fluxo, t_s)
            for nome, fluxo in componentes_rho_t.items()
        }

        budget_b_t = inventario_b_t + precip_b_t
        budget_rho_t = massa_rho_t + precip_rho_t

        variacao_b_pct = erro_relativo_percentual(
            budget_b_t,
            budget_b_t[0],
        )

        variacao_rho_pct = erro_relativo_percentual(
            budget_rho_t,
            budget_rho_t[0],
        )

        i_max_b = int(np.argmax(np.abs(variacao_b_pct)))
        i_max_rho = int(np.argmax(np.abs(variacao_rho_pct)))

        mudanca_massa_dominio_final_pct = float(
            (massa_rho_t[-1] - massa_rho_t[0])
            / max(abs(massa_rho_t[0]), 1.0e-30)
            * 100.0
        )

        precip_final_pct_m0 = float(
            precip_rho_t[-1]
            / max(abs(massa_rho_t[0]), 1.0e-30)
            * 100.0
        )

        if tolerancia_pct is None:
            status = ""
        else:
            status = (
                "DENTRO"
                if abs(float(variacao_b_pct[i_max_b])) <= tolerancia_pct
                else "FORA"
            )

        dt_saida_s = (
            float(np.median(np.diff(t_s)))
            if nt > 1
            else np.nan
        )

        return {
            "grupo": grupo,
            "caso": caso,
            "arquivo": str(caminho.relative_to(ROOT)),
            "commit": ler_commit(caminho.parent),
            "tempo_total_min": float(t_s[-1] / 60.0),
            "n_frames": int(nt),
            "dt_saida_s": dt_saida_s,
            "nx": int(nx),
            "nz": int(nz),
            "dx_m": float(dx),
            "dz_m": float(dz),
            "k_base": int(k_base),
            "z_base_fluxo_m": float(z_m[k_base]),
            "budget_boussinesq_inicial": float(budget_b_t[0]),
            "budget_boussinesq_final": float(budget_b_t[-1]),
            "variacao_boussinesq_final_pct": float(variacao_b_pct[-1]),
            "variacao_boussinesq_max_abs_pct": float(
                abs(variacao_b_pct[i_max_b])
            ),
            "tempo_variacao_boussinesq_max_min": float(
                t_s[i_max_b] / 60.0
            ),
            "variacao_boussinesq_min_pct": float(np.min(variacao_b_pct)),
            "variacao_boussinesq_max_pct": float(np.max(variacao_b_pct)),
            "massa_rho0_inicial_kg_m": float(massa_rho_t[0]),
            "massa_rho0_final_dominio_kg_m": float(massa_rho_t[-1]),
            "precip_proxy_final_kg_m": float(precip_rho_t[-1]),
            "budget_rho0_final_kg_m": float(budget_rho_t[-1]),
            "variacao_rho0_final_pct": float(variacao_rho_pct[-1]),
            "variacao_rho0_max_abs_pct": float(
                abs(variacao_rho_pct[i_max_rho])
            ),
            "tempo_variacao_rho0_max_min": float(
                t_s[i_max_rho] / 60.0
            ),
            "variacao_rho0_min_pct": float(np.min(variacao_rho_pct)),
            "variacao_rho0_max_pct": float(np.max(variacao_rho_pct)),
            "mudanca_massa_dominio_final_pct": (
                mudanca_massa_dominio_final_pct
            ),
            "precip_proxy_final_pct_massa_inicial": precip_final_pct_m0,
            "precip_chuva_proxy_kg_m": float(
                precip_componentes["chuva"][-1]
            ),
            "precip_gelo_proxy_kg_m": float(
                precip_componentes["gelo"][-1]
            ),
            "precip_neve_proxy_kg_m": float(
                precip_componentes["neve"][-1]
            ),
            "precip_graupel_proxy_kg_m": float(
                precip_componentes["graupel"][-1]
            ),
            "tolerancia_pct": (
                "" if tolerancia_pct is None else float(tolerancia_pct)
            ),
            "status_boussinesq": status,
        }


# ============================================================================
# 8. SAIDAS
# ============================================================================


def salvar_csv(
    resultados: list[dict],
    caminho: Path,
):
    if not resultados:
        return

    caminho.parent.mkdir(parents=True, exist_ok=True)
    colunas = list(resultados[0].keys())

    with caminho.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()

        for linha in resultados:
            escritor.writerow(
                {
                    chave: formatar_numero_csv(linha.get(chave))
                    for chave in colunas
                }
            )


def salvar_latex(
    resultados: list[dict],
    caminho: Path,
):
    """Tabela resumida pronta para o Material Suplementar."""

    linhas = []

    linhas.append(r"\begin{table}[htbp]")
    linhas.append(r"\centering")
    linhas.append(
        r"\caption{Diagnóstico de conservação de água dos experimentos "
        r"finais. $\Delta B_{\mathrm{B,max}}$ e "
        r"$\Delta B_{\rho,\mathrm{max}}$ representam a maior variação "
        r"absoluta relativa dos orçamentos Boussinesq e ponderado por "
        r"$\rho_0$, respectivamente. A precipitação é um proxy reconstruído "
        r"offline a partir dos campos salvos.}"
    )
    linhas.append(r"\label{tab:conservacao_massa_todos_grupos}")
    linhas.append(r"\resizebox{\textwidth}{!}{%")
    linhas.append(r"\begin{tabular}{llrrrrrr}")
    linhas.append(r"\toprule")
    linhas.append(
        r"Grupo & Caso & "
        r"$T$ (min) & "
        r"$\Delta B_{\mathrm{B,max}}$ (\%) & "
        r"$\Delta B_{\mathrm{B,final}}$ (\%) & "
        r"$\Delta B_{\rho,\mathrm{max}}$ (\%) & "
        r"$\Delta B_{\rho,\mathrm{final}}$ (\%) & "
        r"$P^{*}/M_0$ (\%) \\"
    )
    linhas.append(r"\midrule")

    ultimo_grupo = None

    for r in resultados:
        grupo = str(r["grupo"])

        if ultimo_grupo is not None and grupo != ultimo_grupo:
            linhas.append(r"\midrule")

        linhas.append(
            "{} & {} & {:.0f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                latex_escape(grupo),
                latex_escape(r["caso"]),
                r["tempo_total_min"],
                r["variacao_boussinesq_max_abs_pct"],
                r["variacao_boussinesq_final_pct"],
                r["variacao_rho0_max_abs_pct"],
                r["variacao_rho0_final_pct"],
                r["precip_proxy_final_pct_massa_inicial"],
            )
        )

        ultimo_grupo = grupo

    linhas.append(r"\bottomrule")
    linhas.append(r"\end{tabular}%")
    linhas.append(r"}")
    linhas.append(
        r"\begin{flushleft}\footnotesize "
        r"$P^{*}$ representa a precipitação acumulada reconstruída "
        r"diagnosticamente a partir dos fluxos sedimentantes de chuva, "
        r"gelo de nuvem, neve e graupel no primeiro nível interno da grade. "
        r"Como o fluxo é integrado apenas nos tempos salvos, esse termo "
        r"deve ser interpretado como proxy de fechamento do orçamento. "
        r"O diagnóstico Boussinesq é o critério primário para avaliar a "
        r"deriva do transporte no núcleo incompressível."
        r"\end{flushleft}"
    )
    linhas.append(r"\end{table}")

    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def salvar_nota_metodologica(
    caminho: Path,
    k_base: int,
    tolerancia_pct: float | None,
):
    texto = f"""DIAGNOSTICO DE CONSERVACAO DE AGUA
===================================

Este arquivo acompanha a tabela gerada por:
    {Path(__file__).name}

Diagnosticos
------------
1. Orcamento Boussinesq:
   B_B(t) = integral qt dA + precipitacao_sedimentante_proxy

2. Diagnostico ponderado por densidade:
   B_rho(t) = integral rho0 qt dA + precipitacao_sedimentante_proxy

Categorias de agua:
    qt = qv + qc + qr + qi + qs + qg

Categorias sedimentantes incluidas no fluxo:
    qr, qi, qs, qg

Nivel usado para reconstruir o fluxo:
    k_base = {k_base}

A precipitacao e reconstruida offline a partir dos tempos salvos no NPZ
por integracao trapezoidal. Portanto, nao e um acumulador exato calculado
a cada passo de tempo do modelo.

O orcamento Boussinesq deve ser considerado o diagnostico primario da deriva
do transporte, pois o nucleo atual resolve div(v)=0. O diagnostico ponderado
por rho0 e mantido como informacao fisica complementar.

Tolerancia para classificacao:
    {"nao definida" if tolerancia_pct is None else f"{tolerancia_pct:g} %"}

Nao se recomenda definir uma tolerancia a posteriori apenas para classificar
os experimentos como aprovados. O objetivo principal desta tabela e medir e
comparar quantitativamente a deriva entre as configuracoes experimentais.
"""

    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(texto, encoding="utf-8")


# ============================================================================
# 9. RELATORIO NO TERMINAL
# ============================================================================


def imprimir_resumo(
    resultados: list[dict],
):
    print()
    print("=" * 100)
    print("TABELA DE CONSERVACAO DE AGUA")
    print("=" * 100)

    cabecalho = (
        f"{'Grupo':<6}"
        f"{'Caso':<28}"
        f"{'Bmax[%]':>12}"
        f"{'Bfinal[%]':>12}"
        f"{'rhoMax[%]':>12}"
        f"{'rhoFin[%]':>12}"
        f"{'P*/M0[%]':>12}"
    )

    print(cabecalho)
    print("-" * 100)

    for r in resultados:
        print(
            f"{r['grupo']:<6}"
            f"{r['caso']:<28}"
            f"{r['variacao_boussinesq_max_abs_pct']:>12.4f}"
            f"{r['variacao_boussinesq_final_pct']:>12.4f}"
            f"{r['variacao_rho0_max_abs_pct']:>12.4f}"
            f"{r['variacao_rho0_final_pct']:>12.4f}"
            f"{r['precip_proxy_final_pct_massa_inicial']:>12.4f}"
        )

    print("=" * 100)


# ============================================================================
# 10. CLI
# ============================================================================


def construir_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Gera tabela de conservacao de agua para os "
            "experimentos finais dos Grupos 1, 2 e 3."
        )
    )

    parser.add_argument(
        "--grupos",
        nargs="+",
        choices=ORDEM_GRUPOS,
        default=list(ORDEM_GRUPOS),
        help="Grupos a processar. Padrao: G1 G2 G3.",
    )

    parser.add_argument(
        "--k-base",
        type=int,
        default=1,
        help=(
            "Indice vertical usado para reconstruir "
            "o fluxo sedimentante na base. Padrao: 1."
        ),
    )

    parser.add_argument(
        "--tolerancia-pct",
        type=float,
        default=None,
        help=(
            "Limiar opcional para classificar o maior "
            "erro do orcamento Boussinesq como DENTRO/FORA. "
            "Por padrao nao ha classificacao."
        ),
    )

    parser.add_argument(
        "--saida",
        type=str,
        default="outputs/conservacao_massa",
        help=(
            "Diretorio de saida relativo a raiz do repositorio "
            "ou caminho absoluto."
        ),
    )

    return parser


def resolver_saida(texto: str) -> Path:
    caminho = Path(texto).expanduser()

    if caminho.is_absolute():
        return caminho.resolve()

    return (ROOT / caminho).resolve()


# ============================================================================
# 11. MAIN
# ============================================================================


def main():
    parser = construir_parser()
    args = parser.parse_args()

    grupos = tuple(args.grupos)

    if args.tolerancia_pct is not None and args.tolerancia_pct < 0.0:
        raise ValueError("--tolerancia-pct deve ser >= 0.")

    experimentos = descobrir_experimentos(grupos)

    if not experimentos:
        raise FileNotFoundError(
            "Nenhum arquivo de experimento final foi encontrado "
            "nos diretorios esperados."
        )

    print()
    print(f"Raiz do repositorio: {ROOT}")
    print(f"Grupos:              {', '.join(grupos)}")
    print(f"k_base:              {args.k_base}")
    print(
        "Tolerancia:          "
        + (
            "nao definida"
            if args.tolerancia_pct is None
            else f"{args.tolerancia_pct:g} %"
        )
    )
    print()

    resultados = []
    erros = []

    for grupo, caminho in experimentos:
        caso = inferir_caso(caminho)

        print(
            f"Processando {grupo}/{caso}: "
            f"{caminho.relative_to(ROOT)}"
        )

        try:
            resultado = calcular_conservacao_arquivo(
                grupo=grupo,
                caminho=caminho,
                k_base=args.k_base,
                tolerancia_pct=args.tolerancia_pct,
            )
            resultados.append(resultado)

        except Exception as exc:
            erros.append(
                {
                    "grupo": grupo,
                    "caso": caso,
                    "arquivo": str(caminho.relative_to(ROOT)),
                    "erro": f"{type(exc).__name__}: {exc}",
                }
            )

            print(f"  [ERRO] {type(exc).__name__}: {exc}")

    if not resultados:
        raise RuntimeError("Nenhum caso foi processado com sucesso.")

    ordem_grupo = {g: i for i, g in enumerate(ORDEM_GRUPOS)}

    def chave_resultado(r):
        grupo = r["grupo"]

        if grupo == "G1":
            ordem = ORDEM_G1
        elif grupo == "G2":
            ordem = ORDEM_G2
        else:
            ordem = ORDEM_G3

        idx = {nome: i for i, nome in enumerate(ordem)}

        return (
            ordem_grupo.get(grupo, 999),
            idx.get(r["caso"], 9999),
            r["caso"],
        )

    resultados.sort(key=chave_resultado)

    saida = resolver_saida(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    caminho_csv = saida / "tabela_conservacao_massa_todos_grupos.csv"
    caminho_tex = saida / "tabela_conservacao_massa_todos_grupos.tex"
    caminho_nota = saida / "nota_metodologica_conservacao.txt"

    salvar_csv(resultados, caminho_csv)
    salvar_latex(resultados, caminho_tex)
    salvar_nota_metodologica(
        caminho=caminho_nota,
        k_base=args.k_base,
        tolerancia_pct=args.tolerancia_pct,
    )

    if erros:
        caminho_erros = saida / "casos_nao_processados.json"
        caminho_erros.write_text(
            json.dumps(erros, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    imprimir_resumo(resultados)

    print()
    print("Arquivos gerados:")
    print(f"  CSV   : {caminho_csv}")
    print(f"  LaTeX : {caminho_tex}")
    print(f"  Nota  : {caminho_nota}")

    if erros:
        print(f"  Erros : {caminho_erros}")
        print(f"  {len(erros)} caso(s) nao puderam ser processados.")

    print()
    print(
        "Observacao: a precipitacao acumulada e um proxy offline "
        "integrado nos tempos salvos; o diagnostico Boussinesq "
        "deve ser usado como referencia primaria da deriva."
    )


if __name__ == "__main__":
    main()
