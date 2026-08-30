# Grupo 2 — Aquecimento, intensidade convectiva e diagnósticos elétricos

## 1. Objetivo

Separar os efeitos de:

1. aquecimento do ambiente;
2. maior intensidade do disparo convectivo inicial;

sobre a evolução dinâmica, a microfísica de fase mista e os proxies de
atividade elétrica McCaul e LPI*.

O movimento vertical `w` é resolvido pelo núcleo dinâmico. Portanto,
**não existe caso com `w` prescrito**.

## 2. Fatores experimentais

Os parâmetros utilizados são:

```python
delta_t_ambiente_k
bolha_k
```

`delta_t_ambiente_k` modifica a temperatura real do perfil ambiental.

Nos casos WARM deve ser usado:

```python
delta_t_ambiente_k = 4.0
preservar_rh = True
```

Assim, a umidade específica ambiental é recalculada para preservar a mesma
umidade relativa do CTRL.

`bolha_k` altera somente a amplitude da perturbação térmica localizada usada
para iniciar a convecção.

## 3. Matriz experimental

| Caso | `delta_t_ambiente_k` | `bolha_k` |
|---|---:|---:|
| `CTRL` | 0 K | `B0` |
| `WARM` | +4 K | `B0` |
| `BUBBLE_PLUS` | 0 K | `B1` |
| `WARM_BUBBLE_PLUS` | +4 K | `B1` |

Referência operacional atual:

```text
B0 = 8 K
```

`B1` deve ser maior que `B0`, mas seu valor final deve ser definido por uma
calibração curta comum antes da bateria científica.

### Regra para definir B1

Escolher um único valor que:

```text
produza convecção mais intensa que B0
alcance a região de fase mista
permita formação mensurável de gelo/graupel
não viole CFL
não produza comportamento numericamente patológico
```

Depois da calibração, registrar explicitamente:

```text
B1 = <valor escolhido> K
```

neste README e commitá-lo **antes** das quatro simulações finais.

Não escolher B1 independentemente para cada caso.

## 4. Configuração que deve permanecer fixa

```text
nx = 90
nz = 110
dx = 100 m
dz = 100 m
dt = 1.5 s
salvar_a_cada = 300 s
tempo = 40 min, salvo decisão comum posterior
Nc = 2.0e8 kg-1
microfisica = thompson
evap_chuva = True
radiacao = False
ciclo_diurno = False
todos os processos microfisicos = True
```

O caso `WARM` inevitavelmente pode modificar a dinâmica, porque alterar o
ambiente modifica empuxo, estabilidade e saturação. Ele não deve ser descrito
como um experimento "apenas microfísico".

## 5. API mínima

O script do grupo deve utilizar:

```python
from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d
from lightning import diagnosticar_relampagos_2d
```

Os quatro casos devem ser construídos a partir de uma configuração-base comum,
mudando apenas os dois fatores da matriz.

Não alterar diretamente o núcleo dinâmico para produzir `BUBBLE_PLUS`.

## 6. Diagnósticos elétricos

Aplicar em todos os casos:

```python
diagnosticar_relampagos_2d(resultado)
```

Comparar pelo menos:

```text
F1
F2
F3
LPI_star
fração de colunas/tempos válidos
w em -15 °C
qg em -15 °C
níveis de 0, -15 e -20 °C
```

McCaul e LPI* são proxies; não devem ser reportados como taxa observada de
flashes.

## 7. Saídas

Salvar:

```text
outputs/group2/CTRL/
outputs/group2/WARM/
outputs/group2/BUBBLE_PLUS/
outputs/group2/WARM_BUBBLE_PLUS/
```

Cada caso deve conter:

```text
resultados_<caso>.npz
comando.txt
commit.txt
configuracao.txt ou configuracao.json
```

Preservar no `.npz` todas as categorias individuais e os diagnósticos
elétricos.

## 8. Comparações principais

Realizar comparações pareadas:

```text
WARM - CTRL
BUBBLE_PLUS - CTRL
WARM_BUBBLE_PLUS - WARM
WARM_BUBBLE_PLUS - BUBBLE_PLUS
```

Isso permite distinguir o efeito do aquecimento, do disparo convectivo mais
intenso e da combinação dos dois.

Analisar, no mínimo:

```text
w máximo e sua evolução temporal
topo da nuvem
qc, qr, qi, qs, qg
conteúdo de fase mista
graupel na região de -15 °C
F1/F2/F3
LPI*
```

## 9. Critérios de aceitação

Uma execução final deve:

```text
usar o mesmo commit para os quatro casos
usar o mesmo B0 nos casos CTRL/WARM
usar o mesmo B1 nos casos BUBBLE_PLUS/WARM_BUBBLE_PLUS
preservar RH em ambos os casos WARM
manter ciclo diurno desligado
não violar CFL
salvar campos microfísicos individuais
```

A conservação de água do transporte 2D é uma limitação/diagnóstico comum do
núcleo e deve ser avaliada antes da interpretação quantitativa final.
