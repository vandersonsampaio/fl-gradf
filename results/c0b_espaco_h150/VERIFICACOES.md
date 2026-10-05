# Verificações pós-grade do C0b (2026-10-04)

Pedidas pelo autor depois do RESULTADO. Análises baratas sobre o código e os dados já existentes.

**Terminologia:** o oráculo-8 uniforme é um **teto de referência** (excluir os atacantes e ponderar os honestos por igual), **não um limite superior**: nada garante que outra ponderação dos honestos não o supere (por exemplo, balancear classes sob rótulos enviesados).

## 1. O código do oráculo com a máscara toda em 1 reproduz o FedAvg-10? **SIM, exatamente**

- **Teste:** cópia literal do `_Oracle8` de `scripts/b27_horizonte.py::run_ceiling`, mas agregando os 10 clientes (ninguém excluído). α 0,1, sementes 72–74, mesma ressemeadura.
- **Resultado:** igual ao `teto_fedavg10` do C0b em H = 15, 50 e 150 nas 3 sementes (|Δ| = 0).
- **Conclusão:** **não há bug no código do oráculo.** C0, B2.7 e C0b não precisam ser recalculados por esse motivo.

## 2. Massa de peso nos atacantes do melhor sistema em cada célula de α 0,1 (dos `scores_*.csv.gz`)

Os pesos por cliente só foram gravados para as 4 variantes do esqueleto. Nas células em que o melhor é uma regra estática (Clustering em `label_flipping` e `trim_attack`), a medida é da melhor variante. Saída completa em `massa_atacantes_a0.1.csv`.

| ataque | melhor existente | sistema medido | massa nos atacantes (rodadas 1–150) | rodadas 101–150 | rodadas com os 2 atacantes excluídos | honestos excluídos por rodada (de 8) |
|---|---|---|---|---|---|---|
| `gaussian_noise` | sr_bin | sr_bin | 0,0000 | 0,0000 | 99,9% | 0,01 |
| `krum_collusion` | sr_b025 | sr_b025 | 0,0000 | 0,0000 | 94,2% | 0,20 |
| `label_flipping` | Clustering | cosserver_only | 0,0033 | 0,0021 | 58,9% | 3,38 |
| `low_mag_backdoor` | sr_b025 | sr_b025 | **0,22** | **0,26** | 4,1% | 1,65 |
| `sign_flipping` | sr_b025 | sr_b025 | 0,038 | 0,054 | 24,5% | 0,85 |
| `trim_attack` | Clustering | sr_b025 | 0,0001 | 0,0000 | 63,6% | 0,22 |

- Em 4 das 6 células, a massa nos atacantes é ≈ 0. Mesmo assim, o melhor método fica 3,9 a 5,6 p.p. **acima** do oráculo FedAvg-8.
- A exceção é o `low_mag_backdoor`: a massa (0,22–0,26) é ≈ a de um peso uniforme (0,2), ou seja, os atacantes **entram**. É a única célula em que "aproveitar os atacantes" pode contribuir: o update deles é pequeno, e eles treinam em dados quase reais.

## 3. Por que métodos sob ataque superam o "oráculo" (diagnóstico feito para explicar o item 2)

- No `gaussian_noise` α 0,1, o `sr_bin` exclui exatamente os 2 atacantes em toda rodada e quase nunca um honesto. Com máscara binária, os 8 honestos recebem peso **uniforme**.
- O oráculo FedAvg-8 também agrega só os 8 honestos, mas o `FedAvgStrategy` do framework pondera **por tamanho de amostra**. Com α 0,1, os tamanhos são muito desiguais: na semente 74, de 110 a 18.561 exemplos por cliente.
- **Teste:** oráculo FedAvg-8 com ponderação uniforme dos 8 honestos, α 0,1, sementes 72–74, H = 150:

| semente | oráculo-8 por tamanho de amostra (o do C0b) | **oráculo-8 uniforme** | sr_bin sob `gaussian_noise` | FedAvg-10 sem ataque |
|---|---|---|---|---|
| 72 | 84,97% | **89,49%** | 89,47% | 89,35% |
| 73 | 84,97% | **89,46%** | 89,50% | 89,34% |
| 74 | 84,99% | **89,42%** | 89,45% | 89,41% |

- **Os "gaps negativos" em α ≤ 0,1 são um artefato da ponderação do oráculo, não do oráculo em si nem do aproveitamento dos atacantes.**
  - *(Mecanismo corrigido em 2026-10-05.)* A "média das acurácias nos test sets dos clientes" é, na prática, a acurácia no **test set global do MNIST, IID e balanceado por classe** (dividido em partes iguais, 1000 exemplos por cliente; `data/download_datasets.py`; verificado no B2.8s). Com rótulos enviesados por cliente (Dirichlet α ≤ 0,1), a ponderação por tamanho de amostra dá peso demais às classes dos clientes grandes, e o modelo agregado fica desbalanceado num teste balanceado. A ponderação uniforme entre clientes dilui esse viés.
  - Com ponderação uniforme, o oráculo-8 reproduz o `sr_bin`, que exclui os atacantes e pondera os honestos por igual, e fica ≈ no FedAvg-10.
- **Consequências** (a decidir; nada foi recalculado):
  - O oráculo FedAvg-8 (com ponderação por amostra), usado como teto no C0, no B2.7 e no C0b, **subestima o teto de referência em α ≤ 0,1** em ~4,5 p.p. (α 0,1, H = 150).
  - O critério de "espaço" do C0b contra um oráculo-8 **uniforme** pode mudar o veredito em células como `label_flipping` α 0,05 (melhor existente 85,1%). Isso pede uma sensibilidade: 30 jobs de teto com o oráculo uniforme, minutos de CPU, declarada como post hoc.
  - A leitura do A0(a) e do C0b ("ponderar os honestos melhor que a média uniforme") deve ser corrigida para: **ponderar os honestos de forma uniforme, e não por tamanho de amostra**, é o que leva ao teto, porque, sob rótulos enviesados, a ponderação por amostra desbalanceia as classes no teste global balanceado.
