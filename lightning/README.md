# Diagnósticos de atividade elétrica

Este pacote contém diagnósticos reutilizáveis que recebem perfis NumPy da
microfísica e/ou da dinâmica. Eles não integram a microfísica nem executam
simulações completas.

```text
lightning/
├── lightning_mod.py  # módulo herdado, mantido por compatibilidade
├── mccaul.py         # McCaul et al. (2009): F1, F2 e F3
└── lpi.py            # Lightning Potential Index adaptado à coluna: LPI*
```

## McCaul et al. (2009)

`mccaul.py` implementa independentemente:

```text
F1 = 0.042 [w (1000 qg)]_-15°C
F2 = 0.20 ∫ rho (qg + qs + qi) dz
F3 = 0.95 F1 + 0.05 F2
```

F1 utiliza somente graupel e movimento ascendente interpolados exatamente na
isoterma de -15 °C. A API recebe `qg` em kg kg⁻¹ (SI) e o converte
internamente para g kg⁻¹ antes de aplicar o coeficiente empírico 0,042,
preservando a convenção numérica empregada na calibração. F2 usa a profundidade
completa da coluna e a coordenada
vertical real. Se -15 °C estiver fora do domínio, F1 e F3 são inválidos (`NaN`),
sem extrapolação ou escolha do nível mais próximo.

Os coeficientes empíricos publicados são preservados. Os resultados serão
usados principalmente como diagnósticos relativos; uma interpretação absoluta
na coluna idealizada exigiria recalibração. O módulo não normaliza pelo CTRL.

## LPI*

`lpi.py` implementa a adaptação unidimensional:

```text
LPI* = [1 / (H_-20 - H_0)] ∫ w² g(w) epsilon dz
qL = qc + qr
qF = qg [sqrt(qi qg)/(qi + qg) + sqrt(qs qg)/(qs + qg)]
epsilon = 2 sqrt(qL qF)/(qL + qF)
```

A integral usa as alturas exatas das isotermas de 0 e -20 °C, construídas por
interpolação linear. Denominadores nulos contribuem com zero. Nesta
implementação, `g(w)=1` somente para `w > 0,5 m s-1`.

Os filtros espaciais horizontais da formulação completa não podem ser
reproduzidos por uma única coluna. Portanto, `f1=f2=1` e o diagnóstico é
explicitamente denominado **LPI\***. Sua unidade é m² s⁻² (J kg⁻¹). O valor
precisa de calibração observacional para estimar taxa de relâmpagos e não é
convertido automaticamente para flashes por minuto nem normalizado pelo CTRL.

## Módulo herdado

`lightning_mod.py` permanece intacto e disponível por compatibilidade. Sua
função histórica relacionada a McCaul não substitui `mccaul.py`.

## Referências

- McCaul, E. W. Jr., Goodman, S. J., LaCasse, K. M., & Cecil, D. J. (2009).
  *Forecasting Lightning Threat Using Cloud-Resolving Model Simulations*.
  Weather and Forecasting, 24, 709-729.
  DOI: 10.1175/2008WAF2222152.1.
- Brisson, E., Blahak, U., Lucas-Picher, P., Purr, C., & Ahrens, B. (2021).
  *Contrasting lightning projection using the lightning potential index
  adapted in a convection-permitting regional climate model*.
  Climate Dynamics, 57, 2037-2051. DOI: 10.1007/s00382-021-05791-z.
- Lynn & Yair (2010) e Yair et al. (2010), trabalhos de origem do LPI conforme
  descritos por Brisson et al. (2021).
