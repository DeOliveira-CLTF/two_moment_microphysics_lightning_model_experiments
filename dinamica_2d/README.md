# Nucleo dinamico 2D

Esta pasta contem a versao modular do nucleo dinamico dos scripts do professor:

- `nuvem_2d.py`: modelo 2D original com microfisica simplificada.
- `nuvem_2d_thompson.py`: mesmo nucleo dinamico 2D acoplado ao esquema completo de microfisica de dois momentos.

No repositorio do grupo, o codigo reutilizavel fica em `dinamica_2d/nucleo.py`. O driver executavel esta em `examples/nuvem_2d_thompson.py`.

## Import principal

```python
from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d
```

## Uso minimo

```python
from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d

config = ConfiguracaoDinamica2D(
    nx=40,
    nz=60,
    tempo_total_s=10 * 60,
    microfisica="thompson",
    bolha_k=3.0,
    evap_chuva=True,
)
resultado = rodar_thompson_2d(config)
frames = resultado["frames"]
```

Os principais campos salvos em `frames` sao:

- `t`: tempos salvos, em segundos;
- `qc_qi`: agua de nuvem + gelo de nuvem;
- `qr_qs_qg`: chuva + neve + graupel;
- `qg`: graupel isolado;
- `w` e `u`: velocidades resolvidas pelo nucleo dinamico;
- `thp` e `qvp`: perturbacoes de temperatura potencial e vapor.

## Relacao com `microfisica`

`dinamica_2d` nao reimplementa os processos microfisicos. O acoplamento usa:

```python
from microfisica.coluna_generica import passo_microfisica_coluna
```

A dinamica calcula `T`, `p`, `rho`, `u`, `w`, adveccao, difusao, empuxo e sedimentacao efetiva por velocidade terminal. A microfisica calcula as fontes locais de vapor, calor latente e hidrometeoros.

## Como usar nos experimentos

Para experimentos de coluna offline, continue usando `microfisica.coluna_step3.ColunaFaseMista`.

Para experimentos com movimento vertical dinamico, importe `rodar_thompson_2d` ou os operadores de `dinamica_2d.nucleo`. Nesses casos, `w` deixa de ser apenas diagnostico e passa a alterar a evolucao por adveccao, empuxo e tempo de residencia dos hidrometeoros. Portanto, resultados dinamicos devem ser discutidos separadamente dos experimentos offline do plano original.

## Exemplo executavel

```bash
python examples/nuvem_2d_thompson.py --tempo 5 --nx 30 --nz 45 --cenario teste_rapido
```

As saidas vao para `outputs/dynamic_2d/` por padrao.
