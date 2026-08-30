# Núcleo microfísico

Este pacote contém o núcleo físico reutilizável do modelo de coluna. Os módulos
`coluna_step1.py`, `coluna_step2.py` e `coluna_step3.py` representam etapas
incrementais da demonstração, apoiadas pelas constantes, distribuições e
processos microfísicos definidos nos demais módulos.

Os exemplos, testes e futuros experimentos importam este pacote; ele não deve
depender dos scripts demonstrativos nem dos diretórios de resultados.

## Microfisica desacoplada da coluna

O modulo `coluna_generica.py` expoe a funcao:

```python
from microfisica.coluna_generica import passo_microfisica_coluna
```

Use essa funcao quando o experimento tiver sua propria dinamica, parametrizacao de cumulus ou movimento vertical simulado fora das classes de coluna. Ela recebe `T`, `p`, `rho` e os campos microfisicos de uma coluna vertical e retorna esses campos apos um passo local do esquema completo de fase mista:

```python
T, qv, qc, Nc, qr, Nr, qi, Ni, qs, Ns, qg, Ng = passo_microfisica_coluna(
    dt, T, p, rho, qv, qc, Nc, qr, Nr, qi, Ni, qs, Ns, qg, Ng,
    evap_chuva=True,
)
```

Ela nao faz sedimentacao, adveccao nem resolve `w`. Esses termos devem ficar no script do experimento ou no nucleo dinamico que chama a microfisica. Isso mantem os modulos principais reutilizaveis e permite que grupos em `experiments/` troquem apenas o import e o ponto de acoplamento.
