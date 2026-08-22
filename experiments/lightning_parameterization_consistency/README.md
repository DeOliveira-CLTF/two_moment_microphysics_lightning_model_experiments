# Testes de consistência das parametrizações de raios

Este diretório reúne diagnósticos de consistência numérica para McCaul
(F1/F2/F3) e LPI*. Não corresponde a um Grupo experimental científico.

O script principal compara a sensibilidade à resolução vertical (100, 50 e
25 m), mantendo fixas as funções físicas do seed de chuva super-resfriada e
do updraft:

```bash
python experiments/lightning_parameterization_consistency/teste_series_temporais_mccaul_lpi.py
```

As figuras são gravadas em `outputs/lightning_parameterization_consistency/`.
O script `perfis_lpi_19_22min.py` permite inspecionar a estrutura vertical de
`qL`, `qF`, `epsilon` e do integrando real de LPI* em instantes específicos.
