# Grupo 1 — Sensibilidade à concentração numérica de gotículas

## 1. Objetivo

Avaliar como a concentração numérica de gotículas de nuvem (`Nc`) modifica a
evolução microfísica da nuvem convectiva e, indiretamente, as propriedades
relevantes aos diagnósticos elétricos.

Este grupo deve alterar somente a concentração utilizada na ativação das
gotículas. Dinâmica, ambiente, amplitude da bolha, duração, grade e demais
processos microfísicos devem permanecer iguais entre os casos.

## 2. Fator experimental

Parâmetro da API:

```python
nc_ativacao_kg1
```

Não utilizar o valor default interno. O valor deve ser declarado explicitamente
em cada caso.

## 3. Matriz experimental

| Caso | `nc_ativacao_kg1` |
|---|---:|
| `N_LOW` | `5.0e7 kg-1` |
| `CTRL` | `2.0e8 kg-1` |
| `N_HIGH` | `5.0e8 kg-1` |

A única diferença entre os três casos deve ser `nc_ativacao_kg1`.

## 4. Configuração que deve permanecer fixa

Usar a configuração comum definida no README principal:

```text
nx = 90
nz = 110
dx = 100 m
dz = 100 m
dt = 1.5 s
salvar_a_cada = 300 s
tempo = 40 min, salvo decisão comum posterior
bolha CTRL de trabalho = 8 K
delta_t_ambiente_k = 0 K
preservar_rh = True
microfisica = thompson
evap_chuva = True
radiacao = False
ciclo_diurno = False
todos os processos microfisicos = True
```

Se a amplitude comum da bolha ou a duração forem recalibradas antes das
simulações finais, atualizar este arquivo e o README principal no mesmo commit.

## 5. API mínima

O script do grupo deve partir de:

```python
from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d
from lightning import diagnosticar_relampagos_2d
```

Para cada caso, construir explicitamente uma `ConfiguracaoDinamica2D` com todos
os parâmetros científicos relevantes e modificar apenas
`nc_ativacao_kg1`.

Não editar `dinamica_2d/nucleo.py` ou os módulos de microfísica para criar os
três casos.

## 6. Saídas mínimas

Salvar os resultados em:

```text
outputs/group1/N_LOW/
outputs/group1/CTRL/
outputs/group1/N_HIGH/
```

Cada diretório deve conter no mínimo:

```text
resultados_<caso>.npz
comando.txt
commit.txt
configuracao.txt ou configuracao.json
```

O `.npz` deve preservar os campos separados:

```text
qc, Nc
qr, Nr
qi, Ni
qs, Ns
qg, Ng
T
qv
w
u
```

e, quando calculados:

```text
F1
F2
F3
LPI_star
```

## 7. Comparações principais

As comparações entre os casos devem usar os mesmos tempos e o mesmo domínio.
Priorizar:

```text
evolução de Nc e qc
produção de chuva
produção de gelo, neve e graupel
conteúdo de fase mista
w máximo
altura/topo da nuvem
McCaul F1/F2/F3
LPI*
```

O objetivo é avaliar respostas relativas ao `CTRL`; não interpretar diferenças
devidas a mudanças simultâneas de outros parâmetros.

## 8. Critérios de aceitação

Uma execução só deve entrar na análise final se:

```text
terminar sem erro
não violar o limite de CFL
usar o mesmo commit dos outros casos
usar exatamente a mesma configuração, exceto Nc
salvar todas as categorias microfísicas separadamente
```

A questão de conservação de água do transporte 2D deve ser tratada como
diagnóstico comum do modelo e não como diferença específica deste grupo.

## 9. Registro final

Antes da análise, criar uma tabela ou arquivo de metadados contendo:

```text
caso
Nc
bolha
delta T
tempo total
dt
nx, nz
dx, dz
CFL máximo
hash do commit
caminho da saída
```

Não sobrescrever uma execução final sem registrar o motivo e produzir novo
commit ou novo identificador de execução.
