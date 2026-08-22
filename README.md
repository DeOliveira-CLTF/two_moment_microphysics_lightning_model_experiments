# Two-moment microphysics and lightning model experiments

Repositório de um modelo de coluna com microfísica de dois momentos e
diagnósticos de atividade elétrica. O código está separado entre o núcleo
físico reutilizável, demonstrações originais, testes e os espaços reservados
aos experimentos científicos do trabalho.

## Organização

- `microfisica/`: núcleo físico reutilizável do modelo de coluna.
- `lightning/`: módulos reutilizáveis de diagnóstico ou parametrização de
  relâmpagos. `lightning_mod.py` mantém o diagnóstico herdado, `mccaul.py`
  implementa F1/F2/F3 e `lpi.py` implementa a adaptação unidimensional LPI*.
- `examples/`: demonstrações incrementais originais: chuva quente, fase de
  gelo, fase mista e diagnóstico de relâmpagos.
- `experiments/`: espaço reservado aos experimentos científicos deste trabalho e aos testes de consistência de parametrizações.
- `tests/`: testes de conservação dos três passos demonstrativos.
- `outputs/baseline/`: resultados dos casos demonstrativos originais.
- `outputs/group1/`, `outputs/group2/` e `outputs/group3/`: resultados futuros
  dos grupos de experimentos científicos.
- `experiments/lightning_parameterization_consistency/`: validações numéricas de McCaul e LPI*.
- `outputs/lightning_parameterization_consistency/`: figuras desses testes de consistência.
- `docs/`: planejamento e documentação de apoio.
- `manuscript/`: material associado ao artigo.

## Passos e grupos

Os Passos 1, 2 e 3 não correspondem aos Grupos experimentais 1, 2 e 3. Os
**Passos** são etapas incrementais de construção e demonstração do modelo. Os
**Grupos** correspondem aos experimentos científicos deste trabalho, ainda não
implementados nesta estrutura.

## Execução dos exemplos

Execute a partir da raiz do repositório:

```bash
python examples/simulacao_passo1.py
python examples/simulacao_passo2.py
python examples/simulacao_passo3.py
python examples/simulacao_passo3_relampago.py
```

Todos os arquivos gerados são gravados em `outputs/baseline/`.

## Execução dos testes

```bash
python tests/teste_conservacao.py
python tests/teste_conservacao_passo2.py
python tests/teste_conservacao_passo3.py
```

Os diretórios em `experiments/` e `outputs/group*/` estão apenas preparados
para trabalho futuro; esta organização não inclui os novos experimentos dos
Grupos 1, 2 ou 3.
