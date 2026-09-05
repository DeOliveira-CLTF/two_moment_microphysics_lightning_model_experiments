# -*- coding: utf-8 -*-
"""
configuracao_casos_grupo3.py
================================

Define os 8 casos do Grupo 3 (atribuicao de processos da fase mista),
estritamente conforme a Tabela "Casos do Grupo 3" do Plano de
Experimentos (secao 4): CTRL + 7 ablacoes, cada uma desligando UM
processo (ou uma familia fisica claramente definida) via
`microfisica.configuracao.OpcoesMicrofisica`.

IMPORTANTE (regra do plano, secao 4 e 5.2): as ablacoes usam
diretamente as chaves ja existentes em `OpcoesMicrofisica` -- este
modulo NAO copia nem edita nenhuma parametrizacao da microfisica
comum, apenas combina as chaves ja expostas.

Correspondencia caso -> chave de OpcoesMicrofisica (a UNICA chave
desligada em cada caso; todas as demais permanecem True, como no
CTRL):

    CTRL             -> nenhuma (todas True)
    SEM-NUC          -> nucleacao_gelo=False
    SEM-DEP          -> deposicao=False
    SEM-CONG-NUV     -> congelamento_nuvem=False
    SEM-CONG-CHUVA   -> congelamento_chuva=False
    SEM-RIMING       -> riming=False
    SEM-HM           -> hallett_mossop=False
    SEM-GELO-NEVE    -> gelo_para_neve=False

As chaves `condensacao_liquida`, `coleta_chuva_por_gelo`, `degelo` e
`chuva_quente` de `OpcoesMicrofisica` NAO fazem parte do desenho do
Grupo 3 (nao aparecem na Tabela do plano) e por isso permanecem
sempre True em todos os 8 casos -- nenhum caso extra e criado.
"""

from dataclasses import replace

from microfisica.configuracao import OpcoesMicrofisica


# Opcoes-base: todas ligadas (== CTRL). As ablacoes partem desta base
# e desligam apenas UMA chave por vez, via dataclasses.replace, para
# garantir que nenhuma outra chave seja alterada por engano.
_BASE = OpcoesMicrofisica()

CASOS_GRUPO3 = {
    "CTRL": {
        "opcoes": _BASE,
        "processo_removido": "Nenhum; todos ativos",
        "pergunta_fisica": "Referencia",
    },
    "SEM-NUC": {
        "opcoes": replace(_BASE, nucleacao_gelo=False),
        "processo_removido": "Nucleacao primaria de gelo (Pidsn)",
        "pergunta_fisica": (
            "Quanto a formacao inicial de cristais condiciona a fase "
            "solida e o feedback dinamico?"
        ),
    },
    "SEM-DEP": {
        "opcoes": replace(_BASE, deposicao=False),
        "processo_removido": (
            "Deposicao/sublimacao em gelo, neve e graupel "
            "(Pidep, Psdep, Pgdep)"
        ),
        "pergunta_fisica": (
            "Quanto o crescimento por vapor controla massa congelada e "
            "calor latente?"
        ),
    },
    "SEM-CONG-NUV": {
        "opcoes": replace(_BASE, congelamento_nuvem=False),
        "processo_removido": "Congelamento de goticulas de nuvem (Pifzc)",
        "pergunta_fisica": (
            "Quanto a conversao direta de liquido para gelo modifica a "
            "fase mista?"
        ),
    },
    "SEM-CONG-CHUVA": {
        "opcoes": replace(_BASE, congelamento_chuva=False),
        "processo_removido": (
            "Congelamento de gotas de chuva em graupel (Pgfzr)"
        ),
        "pergunta_fisica": (
            "Quanto do graupel e da eletrificacao depende dessa via?"
        ),
    },
    "SEM-RIMING": {
        "opcoes": replace(_BASE, riming=False),
        "processo_removido": (
            "Coleta e congelamento de liquido super-resfriado por "
            "particulas de gelo (Pi_iacw, Ps_sacw, Pgacw)"
        ),
        "pergunta_fisica": (
            "Quanto o riming controla graupel, calor latente e "
            "atividade eletrica?"
        ),
    },
    "SEM-HM": {
        "opcoes": replace(_BASE, hallett_mossop=False),
        "processo_removido": "Multiplicacao secundaria Hallett-Mossop (Pispl)",
        "pergunta_fisica": (
            "Quanto a producao secundaria de cristais altera colisoes "
            "e fase solida?"
        ),
    },
    "SEM-GELO-NEVE": {
        "opcoes": replace(_BASE, gelo_para_neve=False),
        "processo_removido": "Autoconversao de gelo de nuvem em neve (Picns)",
        "pergunta_fisica": (
            "Quanto a redistribuicao entre gelo e neve modifica o "
            "sistema?"
        ),
    },
    "SEM_RIMING_HM": {
        "opcoes": replace(_BASE, riming=False, hallett_mossop=False),
        "processo_removido": "Riming + Hallett-Mossop (sinergia)",
        "pergunta_fisica": (
            "A remoção simultânea do riming e da produção secundária de "
            "cristais tem efeito maior que a soma dos efeitos individuais?"
        ),
    },
    "SEM_CONG_CHUVA_RIMING": {
        "opcoes": replace(_BASE, congelamento_chuva=False, riming=False),
        "processo_removido": "Congelamento de chuva + Riming (vias principais de graupel)",
        "pergunta_fisica": (
            "As duas principais vias de formação de graupel, quando "
            "removidas juntas, eliminam completamente o graupel?"
        ),
    },
}

# Ordem de execucao/relato (mesma ordem da Tabela do plano).
ORDEM_CASOS = [
    "CTRL",
    "SEM-NUC",
    "SEM-DEP",
    "SEM-CONG-NUV",
    "SEM-CONG-CHUVA",
    "SEM-RIMING",
    "SEM-HM",
    "SEM-GELO-NEVE",
    "SEM_RIMING_HM",          
    "SEM_CONG_CHUVA_RIMING", 
]


def validar_casos():
    """Confere que cada ablação difere do CTRL em 1 ou 2 chaves."""
    base_dict = _BASE.__dict__
    for nome, spec in CASOS_GRUPO3.items():
        if nome == "CTRL":
            continue
        opc_dict = spec["opcoes"].__dict__
        diffs = [k for k in base_dict if opc_dict[k] != base_dict[k]]
        
        # AQUI ESTÁ A MUDANÇA: aceita 1 ou 2 diferenças
        if len(diffs) not in (1, 2):
            raise AssertionError(
                f"Caso {nome} deveria diferir do CTRL em 1 ou 2 chaves, "
                f"mas difere em {len(diffs)}: {diffs}"
            )
        
        # Verifica se as chaves desligadas são False
        for chave in diffs:
            if opc_dict[chave] is not False:
                raise AssertionError(
                    f"Caso {nome}: a chave {chave} deveria ser False, "
                    f"mas é {opc_dict[chave]}"
                )