# Plano — C0b-iii: a comparação esqueleto × regras estáticas (e o mapa do C0b) depende da métrica?

**Status:** FINAL antes de qualquer run (hash em `PLANO.sha256`). Sensibilidade **post hoc** do C0b, restrita a **α ≤ 0,1**, onde os tamanhos dos clientes são desiguais (de ~10² a ~10⁴).
**Data:** 2026-10-04
**Antecedentes:** C0b (`results/c0b_espaco_h150/`: RESULTADO (iii), VERIFICACOES.md §3, sensibilidade com oráculo uniforme). Inventário de ponderação: nenhum dos 8 sistemas do C0b pondera por tamanho de amostra; o FedAvg dos tetos pondera.

## 1. Pergunta

As conclusões do C0b, (iii) "a melhor variante do esqueleto supera a melhor regra estática na maioria das células" e (i) o mapa de espaço restante, se mantêm com **dois pares coerentes de métrica e teto**?
- **Par U:** métrica **uniforme** (média das acurácias nos test sets dos 10 clientes) com teto oráculo-8 de **ponderação uniforme**;
- **Par P:** métrica **ponderada pelo tamanho do test set** com teto oráculo-8 **ponderado por amostra** (o FedAvg do framework).

## 2. Desenho

- **Células:** as 12 válidas com α ≤ 0,1 (α 0,05 e 0,1 × `trim_attack`, `krum_collusion`, `low_mag_backdoor`, `sign_flipping`, `gaussian_noise` e `label_flipping`).
- **Sistemas e regime idênticos ao C0b:** `sr_only`, `sr_bin`, `sr_b025`, `cosserver_only`, FLTrust, Trimmed-Mean, Clustering e Median; H aninhado 15/50/150; sementes 72–81; `_reseed(seed)` antes de cada sistema.
- **Novidade:** grava as **acurácias por cliente** do modelo global em H = 15, 50 e 150, para que as duas métricas saiam do mesmo run.
- **Tetos**, por α × semente: FedAvg-10 sem ataque e oráculo-8 (só os honestos), cada um com ponderação uniforme e por amostra, com as acurácias por cliente.
- **Total:** 12 × 10 × 8 = 960 runs de célula + 20 jobs de teto (4 runs cada). ~18 h de CPU com 12 processos.
- **Verificação de reprodução:** a métrica uniforme destes runs deve reproduzir **exatamente** o `grade_raw.csv` do C0b (|Δ| < 1e-9). Os tetos devem reproduzir os do C0b (por amostra) e os da sensibilidade (uniforme). Se não reproduzirem, os dados deste plano valem só internamente, e isso é declarado.

## 3. Critério (fixado agora)

Para cada par (U, P) e cada H, por célula (média pareada por semente, IC95):
- **(iii):** D = melhor variante − melhor regra estática (cada uma escolhida por célula, na métrica do par). Classe da célula: "variante" (IC95 > 0), "regra" (IC95 < 0) ou "empate".
- **(i):** gap = teto do par − melhor existente. "Espaço" se gap > 2 p.p. com IC95 > 0 (critério do C0).

**Leitura em H = 150 (primária):**
- **"(iii) blindado"** se a classe coincidir entre U e P em **≥ 75% das 12 células** **e** a direção majoritária (nº de células "variante" contra nº de células "regra") for a mesma nos dois pares.
- **"(iii) depende da métrica"** se a direção majoritária mudar. Nesse caso, o P2 diz que parte da vantagem do filtro por cliente vem de ponderar os honestos uniformemente sob a métrica uniforme.
- Caso intermediário: "blindado na direção, sensível por célula", com as células que mudam.
- **(i):** reportar o nº de células com espaço em cada par e quais são. A expectativa (não critério) é que `label_flipping` α 0,05 continue sendo a única ou a principal.

## 4. Regras

- Código com hash em adendo antes do disparo; análise pré-escrita, rodada uma vez com a grade completa.
- Nenhuma acurácia é inspecionada antes do fim.
- **Fila:** roda na CPU depois do D2.
- Logs não versionados.
