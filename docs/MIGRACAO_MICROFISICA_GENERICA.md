# Migracao para microfisica com dinamica externa

Esta atualizacao incorpora o esquema completo de microfisica de fase mista em uma interface funcional, sem substituir as classes atuais de coluna. O objetivo e permitir que os grupos em `experiments/` acoplem a microfisica a scripts com parametrizacao de cumulus, movimento vertical resolvido ou outro nucleo dinamico sem editar os modulos principais.

## O que mudou

- Foi adicionado `microfisica/coluna_generica.py`.
- O pacote `microfisica` agora exporta `passo_microfisica_coluna`.
- `ColunaChuvaQuente`, `ColunaFaseGelo` e `ColunaFaseMista` continuam com a mesma interface. Experimentos antigos nao precisam mudar.
- A nova funcao resolve apenas processos microfisicos locais. A dinamica, adveccao, sedimentacao e calculo de `w` ficam no script do experimento.

## Import novo para experimentos

Use um destes imports:

```python
from microfisica.coluna_generica import passo_microfisica_coluna
```

ou:

```python
from microfisica import passo_microfisica_coluna
```

## Como chamar

Para uma coluna vertical:

```python
T, qv, qc, Nc, qr, Nr, qi, Ni, qs, Ns, qg, Ng = passo_microfisica_coluna(
    dt,
    T,
    p,
    rho,
    qv,
    qc,
    Nc,
    qr,
    Nr,
    qi,
    Ni,
    qs,
    Ns,
    qg,
    Ng,
    evap_chuva=True,
)
```

Todos os campos devem ser arrays 1D de mesmo tamanho `(nz,)`. Em um modelo 2D, chame a funcao coluna a coluna no eixo horizontal:

```python
for i in range(nx):
    (
        T_i,
        qv_i,
        qc[i, :],
        Nc[i, :],
        qr[i, :],
        Nr[i, :],
        qi[i, :],
        Ni[i, :],
        qs[i, :],
        Ns[i, :],
        qg[i, :],
        Ng[i, :],
    ) = passo_microfisica_coluna(
        dt,
        T[i, :],
        p[i, :],
        rho[i, :],
        qv[i, :],
        qc[i, :],
        Nc[i, :],
        qr[i, :],
        Nr[i, :],
        qi[i, :],
        Ni[i, :],
        qs[i, :],
        Ns[i, :],
        qg[i, :],
        Ng[i, :],
    )
    T[i, :] = T_i
    qv[i, :] = qv_i
```

Se o experimento trabalha com temperatura potencial ou perturbacoes, converta o retorno de `T_i` de volta para as variaveis do seu nucleo dinamico no proprio script de experimento.

## O que continua no experimento

O script em `experiments/` continua responsavel por:

- calcular ou parametrizar movimento vertical (`w`);
- advectar `qv`, `qc`, `qr`, `qi`, `qs`, `qg` e concentracoes numericas;
- aplicar sedimentacao quando ela for resolvida pelo nucleo dinamico;
- diagnosticar empuxo/cold pool a partir de `T`, `qv` e agua condensada;
- escolher se `evap_chuva=True` ou `False`.

## Migracao minima de scripts antigos

Se um experimento antigo fazia:

```python
from microfisica.coluna_step3 import ColunaFaseMista
```

ele pode continuar assim.

Se o experimento passou a ter dinamica propria, troque para:

```python
from microfisica.coluna_generica import passo_microfisica_coluna
```

e mova a chamada da microfisica para dentro do loop temporal, depois da atualizacao dinamica dos campos atmosfericos.
