# Grupo 3 — Ablação de processos microfísicos

## 1. Objetivo

Quantificar a contribuição de processos microfísicos individuais para a
evolução da fase mista, produção de graupel/gelo e diagnósticos de atividade
elétrica.

Cada experimento deve partir de um mesmo caso CTRL e desligar **somente um
grupo de processos por vez**.

## 2. Interface de configuração

Usar:

```python
from microfisica import OpcoesMicrofisica
```

A configuração padrão:

```python
OpcoesMicrofisica()
```

mantém todos os processos ligados.

As opções disponíveis são:

```text
nucleacao_gelo
deposicao
condensacao_liquida
congelamento_nuvem
congelamento_chuva
riming
hallett_mossop
coleta_chuva_por_gelo
gelo_para_neve
degelo
chuva_quente
```

Não editar as funções das parametrizações para criar uma ablação.

## 3. Matriz experimental

| Caso | Alteração em relação ao CTRL |
|---|---|
| `CTRL` | todas as opções `True` |
| `SEM_NUC` | `nucleacao_gelo=False` |
| `SEM_DEP` | `deposicao=False` |
| `SEM_CONG_NUV` | `congelamento_nuvem=False` |
| `SEM_CONG_CHUVA` | `congelamento_chuva=False` |
| `SEM_RIMING` | `riming=False` |
| `SEM_HM` | `hallett_mossop=False` |
| `SEM_GELO_NEVE` | `gelo_para_neve=False` |

As opções `condensacao_liquida`, `coleta_chuva_por_gelo`, `degelo` e
`chuva_quente` permanecem ligadas na matriz científica atual, salvo decisão
posterior documentada pelo grupo.

## 4. Construção segura das ablações

Preferir criar uma nova configuração para cada caso, sem modificar o objeto
CTRL original.

Exemplo conceitual:

```python
from dataclasses import replace
from microfisica import OpcoesMicrofisica

ctrl = OpcoesMicrofisica()
sem_riming = replace(ctrl, riming=False)
```

O teste `tests/test_opcoes_microfisica.py` verifica que esse procedimento não
altera o objeto-base.

## 5. Configuração dinâmica comum

Todos os casos devem utilizar:

```text
nx = 90
nz = 110
dx = 100 m
dz = 100 m
dt = 1.5 s
salvar_a_cada = 300 s
tempo = 40 min, salvo decisão comum posterior
bolha de controle = 8 K
delta_t_ambiente_k = 0 K
Nc = 2.0e8 kg-1
preservar_rh = True
microfisica = thompson
evap_chuva = True
radiacao = False
ciclo_diurno = False
```

Nenhum desses parâmetros deve variar entre as ablações.

## 6. API mínima

O script do grupo deve utilizar:

```python
from dinamica_2d import ConfiguracaoDinamica2D, rodar_thompson_2d
from microfisica import OpcoesMicrofisica
from lightning import diagnosticar_relampagos_2d
```

O objeto `OpcoesMicrofisica` de cada caso deve ser passado pela configuração
dinâmica.

## 7. Saídas

Usar diretórios separados:

```text
outputs/group3/CTRL/
outputs/group3/SEM_NUC/
outputs/group3/SEM_DEP/
outputs/group3/SEM_CONG_NUV/
outputs/group3/SEM_CONG_CHUVA/
outputs/group3/SEM_RIMING/
outputs/group3/SEM_HM/
outputs/group3/SEM_GELO_NEVE/
```

Cada execução deve conter:

```text
resultados_<caso>.npz
comando.txt
commit.txt
configuracao.txt ou configuracao.json
```

A configuração salva deve listar explicitamente todos os booleanos de
`OpcoesMicrofisica`, não apenas o processo desligado.

## 8. Comparações principais

Cada ablação deve ser comparada diretamente com o `CTRL`.

Avaliar:

```text
qc, qr, qi, qs, qg
Nc, Nr, Ni, Ns, Ng
conteúdo total de condensado
conteúdo de fase mista
w e topo da nuvem
graupel na região de -15 °C
F1/F2/F3
LPI*
```

A interpretação deve relacionar a resposta ao processo removido. Se a remoção
alterar fortemente a dinâmica por meio do empuxo, essa resposta faz parte do
experimento acoplado e deve ser discutida.

## 9. Critérios de aceitação

Para uma comparação válida:

```text
somente um processo é desligado por vez
CTRL e ablação usam o mesmo commit
grade, dt, bolha, Nc e ambiente são idênticos
CFL permanece dentro do limite
todos os campos necessários a McCaul/LPI* são salvos
```

Se uma ablação causar ausência física de determinada categoria, isso não é erro
por si só. Valores NaN nos proxies devem ser diferenciados entre:

```text
diagnóstico fisicamente não aplicável
campo necessário ausente
erro numérico
```

## 10. Registro final

Produzir uma tabela de execução contendo:

```text
caso
processo desligado
hash do commit
tempo total
bolha
Nc
CFL máximo
caminho da saída
status da simulação
```

A conservação de água do transporte 2D deve ser tratada como diagnóstico comum
do núcleo original, e não como um defeito específico de uma determinada
ablação.
