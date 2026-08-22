# -*- coding: utf-8 -*-
"""Sensibilidade vertical do CTRL para McCaul e LPI*.

O seed de chuva super-resfriada é uma função contínua de temperatura. Assim,
mudar DZ_M apenas altera sua amostragem vertical, nunca sua amplitude local.

Executar da raiz:
    python experiments/lightning_parameterization_consistency/teste_series_temporais_mccaul_lpi.py
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "lightning_parameterization_consistency"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lightning import compute_epsilon, compute_lpi_star, compute_mccaul
from microfisica.coluna_step3 import ColunaFaseMista

TEMPO_TOTAL_S = 1800.0
DT_S = 2.0
INTERVALO_SAIDA_S = 60.0
DIAGNOSTIC_SPINUP_MIN = 6.0
DZ_M = 50.0
DZ_VALUES_M = (100.0, 50.0, 25.0)
DOMAIN_TOP_M = 7900.0
W_PICO_M_S = 8.0
CLOUD_BASE_M = 1500.0
CLOUD_TOP_MARGIN_M = 300.0
QC_INICIAL_KGKG = 1.0e-3
NC_INICIAL_KG_INV = 2.0e8
RAIN_SEED_CENTER_C = -15.0
RAIN_SEED_SIGMA_C = 3.0
RAIN_SEED_MIN_C = -20.0
RAIN_SEED_MAX_C = -8.0
# Amplitudes do máximo do seed do CTRL ajustado de referência (dz=100 m).
# Agora são parâmetros locais da condição inicial, sem normalização por dz.
QR_SEED_MAX_KGKG = 4.955558888078556e-05
NR_SEED_MAX_KG_INV = 49555.58888078555


def _suffix(dz_m):
    return f"dz_{float(dz_m):g}m"


def _nz_para_dominio(dz_m):
    razao = DOMAIN_TOP_M / float(dz_m)
    if not np.isclose(razao, round(razao), rtol=0.0, atol=1.0e-10):
        raise ValueError("DOMAIN_TOP_M deve ser múltiplo inteiro de dz_m")
    return int(round(razao)) + 1


def _integral_trapezoidal(valores, z_m):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(valores, x=z_m))
    return float(np.trapz(valores, x=z_m))


def _altura_isoterma(z_m, temperature_k, alvo_c):
    temperature_c = np.asarray(temperature_k, dtype=np.float64) - 273.15
    delta = temperature_c - alvo_c
    exato = np.flatnonzero(np.isclose(delta, 0.0, rtol=0.0, atol=1.0e-10))
    if exato.size:
        return float(z_m[exato[0]])
    for superior in range(1, len(z_m)):
        inferior = superior - 1
        if delta[inferior] * delta[superior] < 0.0:
            fracao = -delta[inferior] / (delta[superior] - delta[inferior])
            return float(z_m[inferior] + fracao * (z_m[superior] - z_m[inferior]))
    raise ValueError(f"Isoterma de {alvo_c:g} degC ausente na coluna")


def _extrair_camada(z_m, z_base_m, z_topo_m, campos):
    internos = (z_m > z_base_m) & (z_m < z_topo_m)
    z_camada = np.concatenate(([z_base_m], z_m[internos], [z_topo_m]))
    camada = {}
    for nome, valores in campos.items():
        valores = np.asarray(valores, dtype=np.float64)
        camada[nome] = np.concatenate(([np.interp(z_base_m, z_m, valores)], valores[internos], [np.interp(z_topo_m, z_m, valores)]))
    return z_camada, camada


def _qf_microfisico(qi, qs, qg):
    term_i = np.divide(np.sqrt(qi * qg), qi + qg, out=np.zeros_like(qg), where=(qi + qg) > 0.0)
    term_s = np.divide(np.sqrt(qs * qg), qs + qg, out=np.zeros_like(qg), where=(qs + qg) > 0.0)
    return qg * (term_i + term_s)


def _configurar_caso(dz_m=DZ_M):
    """Construa o CTRL em um domínio físico fixo e resolução configurável."""
    nz = _nz_para_dominio(dz_m)
    coluna = ColunaFaseMista(nz=nz, dz=float(dz_m), T_base=293.0, p_base=95000.0)
    if not np.isclose(coluna.z[-1], DOMAIN_TOP_M, rtol=0.0, atol=1.0e-9):
        raise RuntimeError("O topo físico da grade não corresponde ao domínio solicitado")
    isotermas = {alvo: _altura_isoterma(coluna.z, coluna.T, alvo) for alvo in (0.0, -10.0, -15.0, -20.0)}

    topo_nuvem_m = min(isotermas[-20.0] + CLOUD_TOP_MARGIN_M, coluna.z[-1])
    k_base = int(np.searchsorted(coluna.z, CLOUD_BASE_M, side="left"))
    k_topo = int(np.searchsorted(coluna.z, topo_nuvem_m, side="left"))
    coluna.inserir_nuvem(k_base, k_topo, qc_valor=QC_INICIAL_KGKG, Nc_valor=NC_INICIAL_KG_INV)

    temperature_c = coluna.T - 273.15
    peso = np.exp(-0.5 * ((temperature_c - RAIN_SEED_CENTER_C) / RAIN_SEED_SIGMA_C) ** 2)
    peso = np.where((temperature_c >= RAIN_SEED_MIN_C) & (temperature_c <= RAIN_SEED_MAX_C), peso, 0.0)
    coluna.qr[:] = QR_SEED_MAX_KGKG * peso
    coluna.Nr[:] = NR_SEED_MAX_KG_INV * peso
    if np.any(coluna.qg != 0.0) or np.any(coluna.qi != 0.0) or np.any(coluna.qs != 0.0):
        raise RuntimeError("O seed deve conter somente chuva super-resfriada")

    profundidade = isotermas[-20.0] - isotermas[0.0]
    z_centro_w = 0.5 * (isotermas[0.0] + isotermas[-20.0])
    w_prescrito = W_PICO_M_S * np.exp(-0.5 * ((coluna.z - z_centro_w) / (0.5 * profundidade)) ** 2)
    metadados = {
        "dz_m": float(dz_m), "nz": nz, "domain_top_m": float(coluna.z[-1]), "isotermas_m": isotermas,
        "z_m": coluna.z.copy(), "temperature_c_inicial": temperature_c.copy(), "w_prescrito": w_prescrito.copy(),
        "qr_seed_inicial": coluna.qr.copy(), "Nr_seed_inicial": coluna.Nr.copy(),
        "massa_chuva_inicial_kg_m2": _integral_trapezoidal(coluna.rho * coluna.qr, coluna.z),
        "numero_chuva_inicial_m2": _integral_trapezoidal(coluna.rho * coluna.Nr, coluna.z),
    }
    return coluna, w_prescrito, metadados


def _perfil_lpi(z_m, campos, w_m_s, lpi):
    z, camada = _extrair_camada(z_m, lpi.h_0c_m, lpi.h_minus20c_m, {"w": w_m_s, **{n: campos[n] for n in ("qc", "qr", "qi", "qs", "qg")}})
    ql = camada["qc"] + camada["qr"]
    qf = _qf_microfisico(camada["qi"], camada["qs"], camada["qg"])
    epsilon = compute_epsilon(camada["qc"], camada["qr"], camada["qi"], camada["qs"], camada["qg"])
    integrando = camada["w"] ** 2 * (camada["w"] > 0.5) * epsilon
    internos = (z_m > lpi.h_0c_m) & (z_m < lpi.h_minus20c_m)
    eps_nativo = compute_epsilon(*[campos[n][internos] for n in ("qc", "qr", "qi", "qs", "qg")])
    return {"z_m": z, "qL": ql, "qF": qf, "epsilon": epsilon, "ilpi": integrando, "n_epsilon_positivo": int(np.count_nonzero(eps_nativo > 0.0)), "h0_m": lpi.h_0c_m, "h20_m": lpi.h_minus20c_m}


def executar_teste(dz_m=DZ_M):
    coluna, w_prescrito, metadados = _configurar_caso(dz_m)
    historico = coluna.integrar(TEMPO_TOTAL_S, dt=DT_S, salvar_a_cada=INTERVALO_SAIDA_S)
    nomes = ("f1", "f2", "f3", "f3_from_f1", "f3_from_f2", "qg_minus15_kgkg", "w_minus15_m_s", "graupel_flux_minus15", "lpi_star", "mean_epsilon", "max_epsilon", "charging_depth_m", "qc_integrated", "qr_integrated", "qi_integrated", "qs_integrated", "qg_integrated", "qL_integrated", "qF_integrated")
    series = {nome: [] for nome in nomes}
    for i in range(len(historico["t"])):
        campos = {nome: np.asarray(historico[nome][i], dtype=np.float64) for nome in ("T", "qc", "qr", "qi", "qs", "qg")}
        mccaul = compute_mccaul(coluna.z, campos["T"], coluna.rho, w_prescrito, campos["qi"], campos["qs"], campos["qg"])
        lpi = compute_lpi_star(coluna.z, campos["T"], w_prescrito, campos["qc"], campos["qr"], campos["qi"], campos["qs"], campos["qg"])
        if not mccaul.valid_f1 or not lpi.valid:
            raise RuntimeError("A região de carregamento saiu do domínio vertical")
        series["f1"].append(mccaul.f1); series["f2"].append(mccaul.f2); series["f3"].append(mccaul.f3)
        series["f3_from_f1"].append(0.95 * mccaul.f1); series["f3_from_f2"].append(0.05 * mccaul.f2)
        series["qg_minus15_kgkg"].append(mccaul.qg_minus15_kgkg); series["w_minus15_m_s"].append(mccaul.w_minus15_m_s); series["graupel_flux_minus15"].append(mccaul.graupel_flux_minus15)
        series["lpi_star"].append(lpi.lpi_star); series["mean_epsilon"].append(lpi.mean_epsilon); series["max_epsilon"].append(lpi.max_epsilon); series["charging_depth_m"].append(lpi.charging_depth_m)
        z, camada = _extrair_camada(coluna.z, lpi.h_0c_m, lpi.h_minus20c_m, {"rho": coluna.rho, **{n: campos[n] for n in ("qc", "qr", "qi", "qs", "qg")}})
        for nome in ("qc", "qr", "qi", "qs", "qg"):
            series[f"{nome}_integrated"].append(_integral_trapezoidal(camada["rho"] * camada[nome], z))
        series["qL_integrated"].append(_integral_trapezoidal(camada["rho"] * (camada["qc"] + camada["qr"]), z))
        series["qF_integrated"].append(_integral_trapezoidal(camada["rho"] * _qf_microfisico(camada["qi"], camada["qs"], camada["qg"]), z))
    resultado = {"tempo_min": np.asarray(historico["t"], dtype=float) / 60.0, "metadados": metadados, "historico": historico}
    resultado.update({nome: np.asarray(valores) for nome, valores in series.items()})
    return resultado


def _plotar_series_individuais(series):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t, sufixo = series["tempo_min"], _suffix(series["metadados"]["dz_m"])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(t, series["f1"], lw=2, color="tab:blue"); axes[0].set(ylabel="F1 (diagnóstico relativo)", title="McCaul: fluxo ascendente de graupel em −15 °C")
    axes[1].plot(t, series["f2"], label="F2", color="tab:orange"); axes[1].plot(t, series["f3"], label="F3", color="tab:green"); axes[1].set(xlabel="Tempo (min)", ylabel="Diagnóstico relativo", title="McCaul: conteúdo sólido e diagnóstico combinado") ; axes[1].legend()
    for ax in axes: ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"fig_teste_mccaul_serie_temporal_{sufixo}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(t, series["lpi_star"], lw=2.2, color="tab:purple"); ax.set(xlabel="Tempo (min)", ylabel="LPI* (m² s⁻²)", title="Série temporal do LPI*"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"fig_teste_lpi_star_serie_temporal_{sufixo}.png", dpi=150); plt.close(fig)


def _plotar_diagnosticos_adicionais(series):
    """Preserve as figuras auxiliares dos diagnósticos já calculados."""
    t, meta = series["tempo_min"], series["metadados"]
    sufixo = _suffix(meta["dz_m"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes[0, 0].plot(t, series["qc_integrated"], label="Água de nuvem"); axes[0, 0].plot(t, series["qr_integrated"], label="Chuva")
    axes[0, 0].set(ylabel="Massa (kg m⁻²)", title="Líquido integrado na camada"); axes[0, 0].legend()
    axes[0, 1].plot(t, series["qi_integrated"], label="Gelo"); axes[0, 1].plot(t, series["qs_integrated"], label="Neve"); axes[0, 1].plot(t, series["qg_integrated"], label="Graupel")
    axes[0, 1].set(ylabel="Massa (kg m⁻²)", title="Sólidos integrados na camada"); axes[0, 1].legend()
    axes[1, 0].plot(t, series["qL_integrated"], color="tab:blue", label="qL integrado"); axes[1, 0].plot(t, series["qF_integrated"], color="tab:orange", label="qF integrado")
    axes[1, 0].set(xlabel="Tempo (min)", ylabel="Massa (kg m⁻²)", title="Componentes microfísicos de epsilon"); axes[1, 0].legend()
    axes[1, 1].plot(t, series["lpi_star"], color="tab:purple", label="LPI*"); ax_eps = axes[1, 1].twinx(); ax_eps.plot(t, series["mean_epsilon"], color="tab:green", ls="--", label="epsilon médio")
    axes[1, 1].set(xlabel="Tempo (min)", ylabel="LPI* (m² s⁻²)", title="LPI* e epsilon"); ax_eps.set_ylabel("epsilon médio")
    for ax in axes.flat: ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"fig_diagnostico_lpi_microfisica_{sufixo}.png", dpi=150); plt.close(fig)

    iso, z = meta["isotermas_m"], meta["z_m"] / 1000.
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    axes[0, 0].plot(meta["temperature_c_inicial"], z, color="black")
    for alvo in (0., -10., -15., -20.): axes[0, 0].axhline(iso[alvo] / 1000., ls="--", label=f"{alvo:g} °C")
    axes[0, 0].set(xlabel="Temperatura (°C)", ylabel="Altura (km)", title="Perfil térmico inicial"); axes[0, 0].legend()
    axes[0, 1].plot(meta["qr_seed_inicial"] * 1000., z, color="tab:blue"); axes[0, 1].set(xlabel="qr seed (g kg⁻¹)", ylabel="Altura (km)", title="Seed contínuo de chuva")
    axes[1, 0].plot(meta["w_prescrito"], z, color="tab:green"); axes[1, 0].set(xlabel="w prescrito (m s⁻¹)", ylabel="Altura (km)", title="Updraft diagnóstico")
    axes[1, 1].plot(t, series["qg_minus15_kgkg"], color="tab:gray"); axes[1, 1].set(xlabel="Tempo (min)", ylabel="qg(-15 °C) (kg kg⁻¹)", title="Graupel produzido em −15 °C")
    axes[2, 0].plot(t, series["f3_from_f1"], label="0.95 F1"); axes[2, 0].plot(t, series["f3_from_f2"], label="0.05 F2"); axes[2, 0].set(xlabel="Tempo (min)", ylabel="Diagnóstico", title="Componentes de F3"); axes[2, 0].legend()
    axes[2, 1].plot(t, series["lpi_star"], color="tab:purple", label="LPI*"); axes[2, 1].plot(t, series["mean_epsilon"], color="tab:green", ls="--", label="epsilon médio"); axes[2, 1].set(xlabel="Tempo (min)", ylabel="Diagnóstico", title="LPI* e coexistência"); axes[2, 1].legend()
    for ax in axes.flat: ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"fig_diagnostico_perfil_lightning_{sufixo}.png", dpi=150); plt.close(fig)

def _plotar_perfis_queda(series):
    """Mostre a maior queda após o spin-up, sem alterar a série completa."""
    t, lpi = series["tempo_min"], series["lpi_star"]
    deltas = np.diff(lpi)
    candidatos = np.flatnonzero(t[:-1] >= DIAGNOSTIC_SPINUP_MIN)
    if candidatos.size == 0:
        raise RuntimeError("Não há intervalo diagnóstico após o spin-up configurado")
    indice = int(candidatos[np.argmin(deltas[candidatos])])
    selecionados = sorted(set(np.clip([indice - 1, indice, indice + 1, indice + 2], 0, len(t) - 1)))
    z, w, historico = series["metadados"]["z_m"], series["metadados"]["w_prescrito"], series["historico"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 7), sharey=True)
    paineis = (("qL", "qL (g kg⁻¹)", 1000.), ("qF", "qF (g kg⁻¹)", 1000.), ("epsilon", "epsilon (−)", 1.), ("ilpi", "w² g(w) epsilon (m² s⁻²)", 1.))
    for indice_t in selecionados:
        campos = {n: np.asarray(historico[n][indice_t]) for n in ("T", "qc", "qr", "qi", "qs", "qg")}
        lpi_t = compute_lpi_star(z, campos["T"], w, campos["qc"], campos["qr"], campos["qi"], campos["qs"], campos["qg"])
        perfil = _perfil_lpi(z, campos, w, lpi_t)
        for ax, (campo, label, escala) in zip(axes, paineis): ax.plot(perfil[campo] * escala, perfil["z_m"] / 1000., lw=2, label=f"{t[indice_t]:.0f} min")
    for ax, (_, label, _) in zip(axes, paineis): ax.set(xlabel=label); ax.grid(alpha=.3)
    axes[0].set_ylabel("Altura (km)"); axes[-1].legend(title="Instante")
    dz_m = series["metadados"]["dz_m"]; fig.suptitle(f"Perfis na maior queda pós-spin-up ({DIAGNOSTIC_SPINUP_MIN:g} min; {_suffix(dz_m)})"); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"fig_perfis_lpi_queda_{_suffix(dz_m)}.png", dpi=180); plt.close(fig)
    return t[indice], t[indice + 1], lpi[indice + 1] - lpi[indice]


def _metricas(series, queda):
    t = series["tempo_min"]; i_lpi = int(np.argmax(series["lpi_star"])); meta = series["metadados"]
    return {"dz_m": meta["dz_m"], "nz": meta["nz"], "domain_top_m": meta["domain_top_m"], "max_qr_seed": float(meta["qr_seed_inicial"].max()), "max_Nr_seed": float(meta["Nr_seed_inicial"].max()), "massa_chuva_inicial_kg_m2": meta["massa_chuva_inicial_kg_m2"], "numero_chuva_inicial_m2": meta["numero_chuva_inicial_m2"], "max_qg_minus15": float(series["qg_minus15_kgkg"].max()), "max_f1": float(series["f1"].max()), "max_f2": float(series["f2"].max()), "max_f3": float(series["f3"].max()), "max_lpi": float(series["lpi_star"].max()), "tempo_max_lpi_min": float(t[i_lpi]), "lpi_final": float(series["lpi_star"][-1]), "max_mean_epsilon": float(series["mean_epsilon"].max()), "queda_inicio_min": float(queda[0]), "queda_fim_min": float(queda[1]), "queda_lpi": float(queda[2])}


def _plotar_convergencias(resultados):
    cores = ("tab:blue", "tab:orange", "tab:green")
    fig, ax = plt.subplots(figsize=(9, 5))
    for cor, dz_m, serie in zip(cores, DZ_VALUES_M, resultados.values()): ax.plot(serie["tempo_min"], serie["lpi_star"], lw=2, color=cor, label=f"dz={dz_m:g} m")
    ax.set(xlabel="Tempo (min)", ylabel="LPI* (m² s⁻²)", title="Sensibilidade do LPI* à resolução vertical"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_convergencia_lpi_resolucao_vertical.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, nome in zip(axes, ("f1", "f2", "f3")):
        for cor, dz_m, serie in zip(cores, DZ_VALUES_M, resultados.values()): ax.plot(serie["tempo_min"], serie[nome], lw=1.8, color=cor, label=f"dz={dz_m:g} m")
        ax.set(ylabel=nome.upper()); ax.grid(alpha=.3)
    axes[0].legend(); axes[-1].set_xlabel("Tempo (min)"); fig.suptitle("Sensibilidade de McCaul à resolução vertical"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_convergencia_mccaul_resolucao_vertical.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for cor, dz_m, serie in zip(cores, DZ_VALUES_M, resultados.values()):
        meta = serie["metadados"]; axes[0].plot(meta["qr_seed_inicial"] * 1000., meta["z_m"] / 1000., color=cor, label=f"dz={dz_m:g} m"); axes[1].plot(meta["Nr_seed_inicial"], meta["z_m"] / 1000., color=cor, label=f"dz={dz_m:g} m")
    axes[0].set(xlabel="qr seed (g kg⁻¹)", ylabel="Altura (km)", title="Seed contínuo de chuva"); axes[1].set(xlabel="Nr seed (kg⁻¹)", title="Seed contínuo de número")
    for ax in axes: ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_seed_convergencia_vertical.png", dpi=180); plt.close(fig)


def _imprimir_relatorio(metricas):
    print("Seed físico contínuo: QR_SEED_MAX_KGKG={:.15e}; NR_SEED_MAX_KG_INV={:.15e}".format(QR_SEED_MAX_KGKG, NR_SEED_MAX_KG_INV))
    print("Essas amplitudes foram extraídas do máximo do CTRL ajustado de dz=100 m; qr e Nr não são normalizados por dz.")
    for m in metricas:
        print("dz={dz_m:g} m, nz={nz}, topo={domain_top_m:.0f} m, max(qr)={max_qr_seed:.6e}, max(Nr)={max_Nr_seed:.6e}, M={massa_chuva_inicial_kg_m2:.6e} kg m-2, N={numero_chuva_inicial_m2:.6e} m-2".format(**m))
        print("  max qg(-15)={max_qg_minus15:.6e}, F1/F2/F3={max_f1:.6e}/{max_f2:.6e}/{max_f3:.6e}, max LPI*={max_lpi:.6e} em {tempo_max_lpi_min:.1f} min, LPI final={lpi_final:.6e}, max mean_epsilon={max_mean_epsilon:.6e}".format(**m))
        print("  maior queda pós-spin-up: {queda_inicio_min:.1f}->{queda_fim_min:.1f} min, delta LPI*={queda_lpi:.6e}".format(**m))
    for anterior, atual in zip(metricas, metricas[1:]):
        print(f"Diferenças relativas {anterior['dz_m']:g}->{atual['dz_m']:g} m: M={(atual['massa_chuva_inicial_kg_m2']-anterior['massa_chuva_inicial_kg_m2'])/anterior['massa_chuva_inicial_kg_m2']:.3e}, N={(atual['numero_chuva_inicial_m2']-anterior['numero_chuva_inicial_m2'])/anterior['numero_chuva_inicial_m2']:.3e}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resultados, metricas = {}, []
    for dz_m in DZ_VALUES_M:
        serie = executar_teste(dz_m)
        nomes = (nome for nome in serie if nome not in ("tempo_min", "metadados", "historico"))
        if not all(np.all(np.isfinite(serie[nome])) for nome in nomes): raise RuntimeError("A série contém diagnóstico não finito")
        _plotar_series_individuais(serie)
        _plotar_diagnosticos_adicionais(serie)
        queda = _plotar_perfis_queda(serie)
        resultados[dz_m] = serie; metricas.append(_metricas(serie, queda))
    _plotar_convergencias(resultados)
    _imprimir_relatorio(metricas)


if __name__ == "__main__":
    main()
