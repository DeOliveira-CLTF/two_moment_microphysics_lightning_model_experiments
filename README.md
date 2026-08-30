# Two-moment microphysics and lightning model experiments

Repositório para experimentos idealizados de nuvem convectiva com dinâmica 2D,
microfísica bulk de dois momentos e diagnósticos de atividade elétrica.

A infraestrutura comum é separada dos experimentos científicos para que todos os
casos usem o mesmo núcleo numérico, as mesmas parametrizações e o mesmo formato
de saída.

## 1. Componentes do modelo

### Dinâmica 2D

O diretório `dinamica_2d/` contém a versão modular do núcleo 2D fornecido pelo
professor, baseado na aproximação de Boussinesq e na formulação
vorticidade-função de corrente:

```text
u = -d(psi)/dz
w =  d(psi)/dx
Laplaciano(psi) = zeta
```

O movimento vertical `w` é prognosticado pela dinâmica. Ele não é prescrito nos
experimentos.

### Microfísica

O diretório `microfisica/` contém a microfísica de dois momentos com as
categorias:

```text
qv
qc, Nc
qr, Nr
qi, Ni
qs, Ns
qg, Ng
```

A interface recomendada para o modelo dinâmico é
`microfisica.coluna_generica.passo_microfisica_coluna`.

### Diagnósticos de relâmpagos

O diretório `lightning/` contém:

- McCaul: `F1`, `F2` e `F3`;
- LPI*: adaptação em coluna do Lightning Potential Index;
- `diagnosticos_2d.py`: aplica os dois diagnósticos a cada coluna `x` e a cada
  tempo salvo do modelo 2D.

McCaul e LPI* devem ser interpretados como **proxies de potencial elétrico**,
não como observações diretas de taxa de flashes.

## 2. Organização do repositório

```text
.
├── dinamica_2d/                 # núcleo dinâmico 2D reutilizável
├── microfisica/                 # microfísica de dois momentos
├── lightning/                   # McCaul, LPI* e ponte para o modelo 2D
├── examples/                    # exemplos e testes manuais
├── experiments/
│   ├── group1_droplets/         # sensibilidade a Nc
│   ├── group2_warming_lightning/# aquecimento e intensidade convectiva
│   └── group3_process_ablation/ # ablação de processos microfísicos
├── tests/                       # testes unitários e diagnósticos numéricos
├── outputs/                     # saídas geradas pelas simulações
├── docs/                        # documentação científica
└── manuscript/                  # material do artigo/relatório
```

Os **Passos 1, 2 e 3** são etapas históricas de construção da microfísica.
Os **Grupos 1, 2 e 3** são os experimentos científicos finais e não devem ser
confundidos com esses passos.

## 3. Ambiente computacional

Referência atualmente utilizada:

```text
Python 3.11
NumPy
Matplotlib
pytest
```

Uma instalação mínima pode ser feita com:

```bash
python -m pip install numpy matplotlib pytest
```

Clone o repositório e execute os comandos sempre a partir de sua raiz:

```bash
git clone https://github.com/DeOliveira-CLTF/two_moment_microphysics_lightning_model_experiments.git
cd two_moment_microphysics_lightning_model_experiments
```

Antes dos experimentos finais, registre o ambiente:

```bash
python --version
python -m pip freeze > environment_freeze.txt
git rev-parse HEAD
```

O hash do commit usado nas simulações finais deve ser informado no relatório ou
arquivado junto aos resultados.

## 4. Verificação da instalação

Execute primeiro:

```bash
python -m pytest tests/test_dinamica_2d_experimentos.py -v
python -m pytest tests/test_opcoes_microfisica.py -v
```

Também podem ser executados os testes legados de conservação da microfísica:

```bash
python tests/teste_conservacao.py
python tests/teste_conservacao_passo2.py
python tests/teste_conservacao_passo3.py
```

Teste curto de integração dinâmica + microfísica:

```bash
python examples/nuvem_2d_thompson.py \
  --tempo 2 \
  --nx 90 \
  --nz 110 \
  --bolha 7 \
  --cenario teste_integracao \
  --sem-figuras
```

O teste deve terminar sem erro e informar os valores máximos de CFL.

## 5. Configuração comum dos experimentos

Salvo quando o próprio experimento exigir alteração, utilizar explicitamente:

| Parâmetro | Valor de referência |
|---|---:|
| `nx` | 90 |
| `nz` | 110 |
| `dx` | 100 m |
| `dz` | 100 m |
| `dt` | 1.5 s |
| intervalo de saída | 300 s |
| microfísica | Thompson completa |
| evaporação da chuva | ligada |
| radiação | desligada |
| ciclo diurno | desligado |
| `Nc` de controle | `2.0e8 kg-1` |
| umidade no WARM | preservar RH |
| bolha CTRL de trabalho | 8 K |

A amplitude de 8 K é a referência operacional atual. Se uma calibração comum
indicar outro valor para a bolha de controle, o novo valor deve ser alterado
neste README e nos três READMEs dos grupos **antes** das simulações finais.

Não dependa de valores default internos para parâmetros científicos: declare no
script cada parâmetro relevante de forma explícita.

## 6. Driver geral

`examples/nuvem_2d_thompson.py` é um driver para teste manual da infraestrutura.
Os scripts científicos devem ser implementados dentro de `experiments/group*/`.

Exemplo:

```bash
python examples/nuvem_2d_thompson.py \
  --tempo 10 \
  --bolha 8 \
  --nc 2e8 \
  --delta-t 0 \
  --warm-umidade rh \
  --raios \
  --cenario exemplo \
  --saida outputs/dynamic_2d
```

## 7. Saída do modelo

A saída dinâmica contém, entre outros campos:

```text
t
T
qv
qc Nc
qr Nr
qi Ni
qs Ns
qg Ng
u
w
thp
qvp
cfl_adv
cfl_diff
```

Além disso, são preservados os agregados `qc_qi` e `qr_qs_qg` para
visualizações rápidas.

Quando os diagnósticos elétricos forem executados, também são produzidos:

```text
F1
F2
F3
LPI_star
máscaras de validade
níveis de 0, -15 e -20 °C
variáveis auxiliares
```

Não substituir os campos individuais por agregados nas análises de McCaul ou
LPI*.

## 8. Grupos experimentais

### Grupo 1 — concentração numérica de gotículas

Diretório:

```text
experiments/group1_droplets/
```

Fator experimental: `nc_ativacao_kg1`.

Casos:

```text
N_LOW  = 5.0e7 kg-1
CTRL   = 2.0e8 kg-1
N_HIGH = 5.0e8 kg-1
```

### Grupo 2 — aquecimento e intensidade convectiva

Diretório:

```text
experiments/group2_warming_lightning/
```

Fatores:

```text
delta_t_ambiente_k
bolha_k
```

Matriz:

```text
CTRL
WARM
BUBBLE_PLUS
WARM_BUBBLE_PLUS
```

`WARM` corresponde a `+4 K` na temperatura real do ambiente, preservando RH.
`BUBBLE_PLUS` altera a amplitude da perturbação térmica inicial; não prescreve
`w`.

### Grupo 3 — ablação de processos

Diretório:

```text
experiments/group3_process_ablation/
```

Cada caso deve desligar **um único processo** em relação ao CTRL por meio de
`OpcoesMicrofisica`.

Consulte o README do grupo para a matriz completa.

## 9. Convenção de resultados

Cada grupo deve salvar seus resultados em seu próprio diretório:

```text
outputs/group1/
outputs/group2/
outputs/group3/
```

Recomendação:

```text
outputs/group1/CTRL/
outputs/group1/N_LOW/
outputs/group1/N_HIGH/
```

e analogamente para os demais grupos.

Cada execução final deve preservar:

```text
resultados_<caso>.npz
comando.txt
commit.txt
configuracao.txt ou configuracao.json
```

O script deve imprimir ou salvar pelo menos:

```text
cenário
tempo total
nx, nz
dx, dz
dt
amplitude da bolha
delta T ambiental
Nc
opções microfísicas
CFL máximo advectivo/sedimentação
CFL máximo difusivo
```

## 10. Regras de reprodutibilidade

1. Não editar `dinamica_2d/`, `microfisica/` ou `lightning/` para produzir um
   caso experimental. Mudanças de infraestrutura devem ser discutidas e
   commitadas separadamente.
2. Alterar somente o fator definido para cada experimento.
3. CTRL e sensibilidades devem usar a mesma grade, `dt`, duração e frequência
   de saída.
4. Executar todos os casos finais a partir do mesmo commit.
5. Registrar o comando exato de execução.
6. Não sobrescrever resultados de uma execução anterior sem registrar a
   alteração.
7. Verificar CFL antes de aceitar uma simulação.
8. Calcular McCaul e LPI* com os mesmos critérios em todos os casos comparados.
9. Manter `ciclo_diurno=False` nos experimentos principais, salvo decisão
   científica explícita do grupo.
10. Toda mudança na matriz experimental deve ser registrada primeiro no README
    do respectivo grupo.

## 11. Conservação de água e esquema de transporte

Os testes da microfísica local mostram conservação de água próxima à precisão
numérica.

O transporte 2D preserva a discretização upwind fornecida no núcleo original do
professor. Diagnósticos com traçador indicaram que esse operador não é
estritamente conservativo no sentido integral discreto.

Por enquanto, **não substituir silenciosamente o operador de transporte** por
uma formulação diferente, pois isso modificaria a discretização do núcleo de
referência.

Antes da interpretação quantitativa dos experimentos finais, deve ser
documentado o orçamento de água do modelo 2D e avaliada a magnitude dessa
deriva durante os casos científicos.

## 12. Fluxo de trabalho com Git

Cada integrante deve trabalhar preferencialmente em seu diretório
`experiments/group*/`.

Antes de começar:

```bash
git pull
git status
```

Após uma alteração:

```bash
git add <arquivos-do-seu-grupo>
git commit -m "Implementa experimento <grupo/caso>"
git push
```

Evite commits contendo simultaneamente alterações no núcleo comum e resultados
de um experimento científico.
