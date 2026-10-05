# C0b-iii — CANCELADO (2026-10-05)

**Decisão do autor**, em 05/10, depois do resultado do B2.8s. Nenhum run da grade foi executado; só houve o smoke de reprodução de 04/10, no scratchpad, que reproduziu exatamente o C0b e a sensibilidade.

## Motivo

O plano (`PLANO.md`, commit `7c0311b`) comparava dois pares de métrica e teto: **U** (métrica uniforme sobre os test sets dos clientes + oráculo uniforme) e **P** (métrica ponderada pelo tamanho do test set + oráculo ponderado por amostra).

O B2.8s mostrou que **as duas métricas são idênticas por construção**. Os test sets dos clientes são partes iguais (1000 exemplos), divididas ao acaso, do test set global do MNIST, que é IID e balanceado (`data/download_datasets.py`). Logo:
- a parte de métrica da sensibilidade seria vazia (pares U e P com a mesma métrica);
- a parte de teto (oráculo uniforme × por amostra) já foi medida na sensibilidade do C0b (`results/c0b_espaco_h150/sensibilidade_oraculo_uniforme/`).

Rodar ~18 h de CPU não acrescentaria informação.

## O que fica em aberto (não planejado)

A preocupação de fundo ("parte da vantagem do filtro por cliente vem de ponderar os honestos uniformemente") só pode ser testada com uma métrica **sobre test sets locais com a distribuição de rótulos de cada cliente**, que o pipeline atual não tem. Se for retomada, é um experimento novo, com plano próprio.

O código (`scripts/c0b_iii_metrica.py`, `scripts/run_grid_c0b_iii.sh`) fica versionado como estava, sem uso.
