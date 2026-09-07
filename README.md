# Two-moment microphysics and lightning model experiments

Repositório para experimentos idealizados de convecção atmosférica com
dinâmica bidimensional, microfísica *bulk* de dois momentos e diagnósticos
de potencial de eletrificação.

O projeto foi desenvolvido para investigar como alterações microfísicas,
termodinâmicas e dinâmicas modificam a evolução de uma nuvem convectiva
idealizada, com atenção especial à precipitação, à fase mista, à formação
de graupel e às condições favoráveis à eletrificação.

A infraestrutura numérica é compartilhada pelos experimentos, mas os três
grupos científicos possuem atualmente configurações e matrizes experimentais
próprias.

> **Importante:** os scripts de execução de cada grupo são a fonte de verdade
> para os parâmetros científicos utilizados nas simulações.

---

## 1. Componentes do modelo

### 1.1 Dinâmica 2D

O diretório

```text
dinamica_2d/
```

contém o núcleo dinâmico bidimensional utilizado nos experimentos.

O modelo é baseado na aproximação de Boussinesq e na formulação
vorticidade–função de corrente:

```text
u = -d(psi)/dz
w =  d(psi)/dx

Laplaciano(psi) = zeta
```

onde:

- `u` é a velocidade horizontal;
- `w` é a velocidade vertical;
- `psi` é a função de corrente;
- `zeta` é a vorticidade.

A velocidade vertical é prognosticada pelo núcleo dinâmico e não é
prescrita diretamente.

No Grupo 2 existe um forçamento mecânico externo de levantamento, mas esse
forçamento atua na equação dinâmica; `w` continua sendo uma variável
prognóstica do modelo.

---

### 1.2 Microfísica de dois momentos

O diretório

```text
microfisica/
```

contém a parametrização microfísica *bulk* de dois momentos.

São representadas as seguintes categorias de água:

```text
qv             vapor d'água

qc, Nc         água de nuvem
qr, Nr         chuva
qi, Ni         gelo de nuvem
qs, Ns         neve
qg, Ng         graupel
```

Para cada categoria condensada, o modelo pode prognosticar tanto a razão
de mistura quanto a concentração numérica.

Os processos microfísicos individuais podem ser controlados por meio de

```python
OpcoesMicrofisica
```

Essa interface é utilizada diretamente nos experimentos de ablação do
Grupo 3.

---

### 1.3 Diagnósticos de atividade elétrica

O diretório

```text
lightning/
```

contém os diagnósticos de potencial de eletrificação calculados
*offline* a partir dos campos produzidos pelo modelo.

Atualmente são utilizados:

```text
McCaul F1
McCaul F2
McCaul F3
LPI*
```

O módulo `diagnosticos_2d.py` aplica esses diagnósticos às colunas do
domínio bidimensional ao longo dos tempos salvos.

**F1, F2, F3 e LPI\*** devem ser interpretados como *proxies* de condições
favoráveis à atividade elétrica.

O modelo não representa explicitamente:

```text
separação de cargas
campo elétrico
descarga elétrica
taxa observada de flashes
```

---

## 2. Organização do repositório

```text
.
├── dinamica_2d/
│   └── núcleo dinâmico bidimensional
│
├── microfisica/
│   └── microfísica bulk de dois momentos
│
├── lightning/
│   └── McCaul, LPI* e diagnósticos elétricos
│
├── examples/
│   └── exemplos e testes manuais do modelo
│
├── experiments/
│   ├── group1_droplets/
│   │   └── sensibilidade à concentração de gotículas
│   │
│   ├── group2_warming_lightning/
│   │   └── aquecimento, umidade e forçamento dinâmico
│   │
│   ├── group3_process_ablation/
│   │   └── ablação de processos microfísicos
│   │
│   └── lightning_parameterization_consistency/
│       └── testes de consistência dos proxies elétricos
│
├── tests/
│   └── testes numéricos, conservação e consistência
│
├── outputs/
│   └── resultados das simulações
│
├── docs/
│   └── documentação científica
│
└── manuscript/
    └── material associado ao artigo/relatório
```

Os antigos **Passos 1, 2 e 3** encontrados em partes do código correspondem
às etapas históricas de desenvolvimento da microfísica.

Eles não devem ser confundidos com os **Grupos Experimentais 1, 2 e 3**
descritos abaixo.

---

## 3. Instalação

Clone o repositório:

```bash
git clone https://github.com/DeOliveira-CLTF/two_moment_microphysics_lightning_model_experiments.git
```

Entre no diretório:

```bash
cd two_moment_microphysics_lightning_model_experiments
```

Uma instalação mínima utiliza:

```bash
python -m pip install numpy matplotlib pytest
```

Antes de uma bateria científica é recomendável registrar:

```bash
python --version
git rev-parse HEAD
```

Opcionalmente, todo o ambiente Python pode ser arquivado com:

```bash
python -m pip freeze > environment_freeze.txt
```

---

## 4. Testes

A suíte de testes pode ser executada a partir da raiz do repositório:

```bash
python -m pytest -q
```

Testes específicos também podem ser executados individualmente.

Por exemplo:

```bash
python -m pytest tests/test_dinamica_2d_experimentos.py -v
```

```bash
python -m pytest tests/test_opcoes_microfisica.py -v
```

---

# 5. Experimentos científicos

## 5.1 Grupo 1 — Sensibilidade à concentração numérica de gotículas

Diretório:

```text
experiments/group1_droplets/
```

### Objetivo

Investigar como alterações na concentração numérica inicial de gotículas
de nuvem modificam a evolução da convecção, da precipitação, da fase
congelada e dos diagnósticos de potencial elétrico.

O fator experimental é:

```python
nc_ativacao_kg1
```

### Matriz experimental

| Caso | `Nc` |
|---|---:|
| `N_LOW` | `5.0e7 kg-1` |
| `CTRL` | `2.0e8 kg-1` |
| `N_HIGH` | `5.0e8 kg-1` |

Apenas `Nc` deve variar entre os três casos.

### Configuração atual

O driver científico atual utiliza:

```text
nx = 90
nz = 110

dx = 100 m
dz = 100 m

dt = 1.5 s

tempo total = 40 min
saída = 300 s

bolha térmica = 10 K

delta T ambiental = 0 K
Nc CTRL = 2.0e8 kg-1

evaporação de chuva = ligada
radiação = desligada
ciclo diurno = desligado
```

### Execução

Teste curto:

```bash
python experiments/group1_droplets/rodar_experimentos_grupo1.py \
  --modo teste
```

Bateria científica:

```bash
python experiments/group1_droplets/rodar_experimentos_grupo1.py \
  --modo final
```

Um caso individual pode ser selecionado com:

```bash
python experiments/group1_droplets/rodar_experimentos_grupo1.py \
  --modo final \
  --caso CTRL
```

### Saídas

As simulações finais são armazenadas em:

```text
outputs/group1/experimentos_B10/
```

com uma pasta para cada experimento:

```text
N_LOW/
CTRL/
N_HIGH/
```

---

# 5.2 Grupo 2 — Aquecimento, umidade e forçamento dinâmico

Diretório:

```text
experiments/group2_warming_lightning/
```

## Objetivo

Investigar separadamente e de forma acoplada os efeitos de:

1. aquecimento ambiental;
2. conteúdo de umidade no ambiente aquecido;
3. intensidade do mecanismo de levantamento;
4. resposta dinâmica da convecção;
5. evolução da fase mista e do graupel;
6. potencial de eletrificação.

O Grupo 2 foi reformulado em relação à versão inicial do projeto.

### Iniciação convectiva

Neste grupo não é utilizada bolha térmica:

```text
bolha_k = 0
bolha_qv_kgkg = 0
```

A iniciação da convecção é realizada por um forçamento mecânico externo
de levantamento.

O parâmetro principal é:

```python
forc_dyn_amp_m_s2
```

A geometria e a duração do forçamento permanecem fixas entre os casos.

Somente sua amplitude é modificada entre:

```text
D0
D1
```

A velocidade vertical não é prescrita pelo experimento. O forçamento atua
na dinâmica e `w` continua sendo prognosticado pelo modelo.

---

## Aquecimento ambiental

Nos casos aquecidos:

```text
Delta T = +4 K
```

O perfil utilizado pelo Grupo 2 é:

```text
perfil_ambiente = "referencia"
```

São avaliadas duas formas diferentes de modificar a umidade.

### `qv_fixo`

```text
T_warm = T_ctrl + 4 K
qv_warm = qv_ctrl
```

Consequentemente, a umidade relativa diminui no ambiente aquecido.

### `rh_fixa`

A umidade relativa inicial é mantida igual à do CTRL e `qv` é aumentado
de acordo com o novo estado termodinâmico.

---

## Matriz experimental final

| Caso | ΔT | Tratamento da umidade | Forçamento |
|---|---:|---|---|
| `CTRL` | 0 K | referência | `D0` |
| `DYN_PLUS` | 0 K | referência | `D1` |
| `WARM_QV` | +4 K | `qv` fixo, RH variável | `D0` |
| `WARM_QV_DYN_PLUS` | +4 K | `qv` fixo, RH variável | `D1` |
| `WARM_RH` | +4 K | RH fixa, `qv` ajustado | `D0` |
| `WARM_RH_DYN_PLUS` | +4 K | RH fixa, `qv` ajustado | `D1` |

CTRL e DYN_PLUS não precisam ser repetidos entre os dois tratamentos de
umidade porque, na ausência de aquecimento, ambos produzem o mesmo perfil
ambiental inicial.

---

## Configuração atual

```text
nx = 90
nz = 151

dx = 100 m
dz = 100 m

dt = 1.0 s

tempo total = 40 min
saída = 300 s

Delta T WARM = +4 K

Nc = 2.0e8 kg-1

bolha térmica = 0 K
perturbação adicional de vapor = 0

centro vertical do forçamento = 800 m
escala horizontal = 2000 m
escala vertical = 700 m

início do forçamento = 0 s
duração = 900 s
```

A matriz final arquivada atualmente utiliza:

```text
D0 = 0.55 m s-2
D1 = 0.65 m s-2
```

---

## Varredura preliminar do forçamento

O driver permite realizar inicialmente uma varredura de amplitudes.

Exemplo:

```bash
python experiments/group2_warming_lightning/experimento_grupo2_reformulado.py \
  varredura \
  --ambiente ambos \
  --umidade qv_fixo
```

A varredura é utilizada para identificar amplitudes capazes de sustentar
convecção profunda adequada.

Os critérios físicos incluem:

```text
fase mista ativa
graupel na região de aproximadamente -15 °C
updraft na região de aproximadamente -15 °C
convecção profunda
estabilidade segundo CFL
```

Os índices elétricos não devem ser utilizados para escolher o forçamento.

`F3` e `LPI*` são resultados científicos da simulação.

---

## Execução da matriz final

```bash
python experiments/group2_warming_lightning/experimento_grupo2_reformulado.py \
  final \
  --d0 0.55 \
  --d1 0.65 \
  --umidade ambas
```

É possível executar apenas uma família de umidade:

```text
--umidade qv_fixo
```

ou

```text
--umidade rh_fixa
```

---

## Saídas

A matriz de seis casos pode ser organizada em:

```text
outputs/group2/final_decomposto/
```

Os resultados individuais permanecem separados por caso.

---

# 5.3 Grupo 3 — Ablação de processos microfísicos

Diretório:

```text
experiments/group3_process_ablation/
```

## Objetivo

Quantificar a contribuição de processos específicos da microfísica para:

```text
formação de fase sólida
produção de graupel
redistribuição entre hidrometeoros
movimentos verticais
precipitação
potencial de eletrificação
```

As ablações são realizadas por meio de:

```python
OpcoesMicrofisica
```

sem reescrever as parametrizações microfísicas.

---

## Matriz experimental atual

A matriz possui um CTRL, sete ablações individuais e duas ablações
combinadas:

```text
CTRL

SEM-NUC
SEM-DEP
SEM-CONG-NUV
SEM-CONG-CHUVA
SEM-RIMING
SEM-HM
SEM-GELO-NEVE

SEM_RIMING_HM
SEM_CONG_CHUVA_RIMING
```

### Ablações individuais

| Caso | Processo removido |
|---|---|
| `CTRL` | nenhum |
| `SEM-NUC` | nucleação primária de gelo |
| `SEM-DEP` | deposição/sublimação em gelo, neve e graupel |
| `SEM-CONG-NUV` | congelamento de gotículas de nuvem |
| `SEM-CONG-CHUVA` | congelamento de gotas de chuva |
| `SEM-RIMING` | riming/coleta de líquido super-resfriado |
| `SEM-HM` | multiplicação secundária Hallett–Mossop |
| `SEM-GELO-NEVE` | conversão de gelo de nuvem em neve |

### Ablações combinadas

#### `SEM_RIMING_HM`

```text
riming = False
hallett_mossop = False
```

Esse caso investiga a interação entre riming e produção secundária de
cristais de gelo.

#### `SEM_CONG_CHUVA_RIMING`

```text
congelamento_chuva = False
riming = False
```

Esse caso investiga conjuntamente duas importantes vias relacionadas à
produção de graupel.

---

## Configuração dinâmica atual

Todos os casos utilizam, por padrão:

```text
nx = 90
nz = 110

dx = 100 m
dz = 100 m

dt = 1.5 s

tempo total = 40 min
saída = 300 s

bolha térmica = 8 K

Delta T ambiental = 0 K
Nc = 2.0e8 kg-1

evaporação de chuva = ligada
radiação = desligada
ciclo diurno = desligado
```

Os parâmetros acima permanecem iguais entre as ablações.

---

## Execução

Todos os experimentos:

```bash
python experiments/group3_process_ablation/executar_grupo3.py
```

Um subconjunto pode ser executado com:

```bash
python experiments/group3_process_ablation/executar_grupo3.py \
  --casos CTRL SEM-RIMING SEM_RIMING_HM
```

As simulações podem ser executadas em paralelo utilizando os núcleos
disponíveis da máquina.

---

## Saídas

Cada caso é salvo em:

```text
outputs/group3/<CASO>/
```

Por exemplo:

```text
outputs/group3/CTRL/
outputs/group3/SEM-RIMING/
outputs/group3/SEM-HM/
outputs/group3/SEM_RIMING_HM/
```

---

# 6. Campos de saída

Os arquivos de resultados preservam campos microfísicos e dinâmicos,
incluindo, conforme o driver:

```text
t
T
qv

qc
qr
qi
qs
qg

Nc
Nr
Ni
Ns
Ng

u
w

thp
qvp
```

Também são armazenados metadados ambientais, como:

```text
x_m
z_m
p_pa_1d
rho0_1d
theta_env_1d
T_env_1d
qv_env_1d
rh_env_1d
```

Quando os diagnósticos elétricos são executados, podem ser incluídos:

```text
F1
F2
F3
LPI_star
```

além das variáveis auxiliares utilizadas pelos diagnósticos.

---

# 7. Consistência das parametrizações elétricas

O diretório

```text
experiments/lightning_parameterization_consistency/
```

não constitui um quarto grupo experimental.

Ele reúne experimentos de consistência numérica destinados a avaliar o
comportamento dos diagnósticos McCaul e LPI*.

Entre os testes disponíveis está a comparação da resposta dos índices sob
diferentes resoluções verticais:

```text
100 m
50 m
25 m
```

O teste principal pode ser executado com:

```bash
python experiments/lightning_parameterization_consistency/teste_series_temporais_mccaul_lpi.py
```

Também existem ferramentas para inspecionar os perfis verticais das
variáveis que compõem o LPI*.

---

# 8. Estabilidade numérica

As simulações monitoram dois diagnósticos principais:

```text
CFL advectivo/sedimentação
CFL difusivo
```

Cada execução científica deve ser verificada quanto à estabilidade antes
de entrar na análise final.

Os valores máximos são armazenados ou reportados pelos drivers, por exemplo:

```text
cfl_max_adv
cfl_max_diff
```

Uma execução que viole os critérios estabelecidos de CFL não deve ser
utilizada na comparação científica.

---

# 9. Conservação de água

O repositório contém diagnósticos específicos para avaliar a conservação
de água no núcleo bidimensional.

## 9.1 Teste isolado

O script

```text
tests/diagnostico_conservacao_dinamica_2d.py
```

separa duas questões:

1. conservação local de água pela microfísica;
2. deriva causada pelo transporte bidimensional.

A água total é definida como:

```text
qt = qv + qc + qr + qi + qs + qg
```

Para a formulação Boussinesq do núcleo, o diagnóstico primário utiliza o
inventário:

```text
I_B = integral qt dA
```

Também é calculado, separadamente, um inventário ponderado pela densidade
de referência:

```text
M = integral rho0 qt dA
```

O teste pode ser executado com:

```bash
python -m pytest tests/diagnostico_conservacao_dinamica_2d.py -s -v
```

ou diretamente:

```bash
python tests/diagnostico_conservacao_dinamica_2d.py
```

---

## 9.2 Diagnóstico dos experimentos finais

O script

```text
tests/gerar_tabela_conservacao_todos_grupos.py
```

calcula o diagnóstico de conservação para os experimentos já executados
dos Grupos 1, 2 e 3.

Não é necessário rerodar as simulações.

O orçamento Boussinesq considera:

```text
B_B(t) = I_B(t) + P_B(t)
```

onde `P_B` representa a água sedimentada pela base do domínio.

São consideradas as categorias sedimentantes:

```text
chuva
gelo de nuvem
neve
graupel
```

Também é calculado um orçamento auxiliar ponderado por `rho0`.

Execução:

```bash
python tests/gerar_tabela_conservacao_todos_grupos.py
```

As saídas são armazenadas em:

```text
outputs/conservacao_massa/
```

incluindo:

```text
tabela_conservacao_massa_todos_grupos.csv
tabela_conservacao_massa_todos_grupos.tex
nota_metodologica_conservacao.txt
```

A precipitação utilizada nesse diagnóstico é reconstruída a partir dos
tempos salvos nos arquivos `.npz`. Portanto, o fluxo sedimentante integrado
é um diagnóstico *offline* e não substitui um acumulador calculado a cada
passo do núcleo.

---

# 10. Metadados e reprodutibilidade

Os drivers científicos salvam, dependendo do grupo, arquivos como:

```text
resultados_<CASO>.npz

configuracao.json
configuracao_<CASO>.json

resumo_execucao.json
resumo_<CASO>.json

comando.txt
commit.txt
status.txt
```

Esses arquivos permitem registrar:

```text
caso experimental
parâmetros utilizados
tempo de integração
grade
passo de tempo
CFL
configuração microfísica
hash do commit
comando utilizado
```

Para comparações científicas válidas:

1. todos os casos de uma mesma matriz devem utilizar o mesmo commit;
2. somente os fatores definidos pelo desenho experimental devem variar;
3. grade, duração e frequência de saída devem permanecer consistentes dentro
   de cada grupo;
4. os resultados devem ser verificados quanto ao CFL;
5. McCaul e LPI* devem utilizar os mesmos critérios nos casos comparados;
6. o diagnóstico de conservação de água deve ser considerado antes da
   interpretação quantitativa;
7. resultados finais não devem ser sobrescritos sem registro da nova execução.

---

# 11. Fluxo de trabalho com Git

Antes de começar:

```bash
git pull
git status
```

Após alterações:

```bash
git add <arquivos>
git commit -m "Descrição da alteração"
git push
```

É recomendável evitar commits que misturem simultaneamente:

```text
mudanças no núcleo comum
mudanças na matriz experimental
resultados científicos
```

Alterações em

```text
dinamica_2d/
microfisica/
lightning/
```

devem ser claramente documentadas, pois afetam potencialmente mais de um
grupo experimental.

---

# 12. Resumo da matriz científica atual

| Grupo | Pergunta principal | Casos |
|---|---|---:|
| Grupo 1 | Sensibilidade à concentração de gotículas | 3 |
| Grupo 2 | Aquecimento, umidade e intensidade do levantamento | 6 |
| Grupo 3 | Sensibilidade a processos microfísicos | 10 |

Total da matriz científica principal:

```text
19 experimentos
```

além dos experimentos auxiliares de calibração, testes numéricos,
consistência dos proxies elétricos e diagnósticos de conservação.