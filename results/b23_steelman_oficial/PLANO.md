# Plano — B2.3: steelman do TD3 no AdaAggRL oficial

**Status:** FINAL antes dos dados (hash em `PLANO.sha256`). Exploratório.
**Data:** 2026-09-28
**Roadmap:** `references/roadmap_tese_gradf_v2.md` §4, B2.3 ("exploratório, essencial") e Portão B-a.
**Registro:** local, com hash; o commit fica a critério do autor.

## 1. Pergunta

A configuração publicada do TD3 (lr 1e-5, 100 rodadas de aquecimento aleatório, ~66 atualizações do ator em 500 rodadas) não aprende (Passo 2, exploratório; B2.2 em confirmação). Dando ao TD3 condições muito mais favoráveis, ele passa a aprender algo que dependa da entrada e a superar a ação fixa?

## 2. Configuração (uma só, fixada antes dos dados)

| parâmetro | oficial | steelman |
|---|---|---|
| learning_rate (ator e crítico) | 1e-5 | **1e-3** |
| learning_starts | 100 | **10** |
| demais (MlpPolicy [256,128], buffer 1000, batch 64, train_freq 3, ruído N(0; 0,1), γ 0,99) | — | iguais |

Justificativa: lr 100× maior dá passos de atualização da ordem dos usados em TD3 padrão (o default do SB3 é 1e-3); aquecimento curto remove a fase aleatória que causou a maioria dos resets sob EB e aumenta o nº de atualizações. Uma configuração só, por custo (~16 h); se o resultado for ambíguo, a grade 2×2 fica como passo seguinte.

## 3. Grade

- Sementes **100–104** (gastas; exploratório), LMP e EB → **10 runs** (~16 h, depois do fim do B2.1).
- Pareamento por (ataque, semente) com os runs `fixed` e `td3` do Passo 2 (mesmo ambiente, mesmos atacantes e partição; sequência de clientes pareada pela semente). Não é preciso rodar as linhas de base de novo.
- Runner `scripts/passo2_oficial/run_b23.py` (reusa `run_b21.py` sem alterá-lo; salva estados e checkpoints do ator a cada 50 passos).

## 4. Análise (pré-escrita em `scripts/passo2_oficial/analisar_b23.py`)

- Primária: mediana da acurácia nas rodadas 401–500 (a mesma do B2.1). Secundária: média 451–500.
- C1: steelman − fixed; C2: steelman − td3 oficial. n = 10 pares, Wilcoxon bilateral, Δ, IC95, d; Holm por ataque (descritivo).
- Mecanismo: drift da política determinística ao longo dos checkpoints (π_t contra π_0) e S_swap (troca do estado pelo do outro ataque), comparados com σ_a = 0,0475.

## 5. Critério do Portão B-a (fixado antes dos dados)

"O steelman aprende e supera a ação fixa" exige **as três** condições:
1. C1 com Δ > 0 e Wilcoxon p < 0,05 (primária);
2. mediana de drift_500 > σ_a (a política se afastou da inicial);
3. mediana de S_swap > σ_a (a política depende da entrada).

- Se as três valem → o P2 vira "a configuração publicada não aprende; bem ajustado, o RL entrega X" (roadmap, B-a).
- Se C1 não vale → a tese do P2 fica forte ("nem o steelman supera a fixa").
- Se C1 vale sem 2–3 → investigar (ganho sem aprendizado dependente da entrada, por exemplo pela redução do aquecimento aleatório).

## 6. Limitações declaradas

- Exploratório e com n = 10 pares; as linhas de base do Passo 2 já foram inspecionadas pelo autor. Uma afirmação confirmatória exigiria sementes novas.
- Uma única configuração não esgota o espaço de hiperparâmetros; "o steelman não supera" significa "esta configuração favorável não supera", não "nenhuma configuração supera".
