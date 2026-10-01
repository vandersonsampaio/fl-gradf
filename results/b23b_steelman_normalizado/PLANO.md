# Plano — B2.3b: último steelman do TD3 (recompensa normalizada)

**Status:** FINAL antes dos dados (hash em `PLANO.sha256`). Exploratório.
**Data:** 2026-09-30
**Roadmap:** `references/roadmap_tese_gradf_v3.md` §4.1 (ordem 1) e §7.2; Portão B-a definitivo.
**Registro:** local, com hash; o commit fica a critério do autor.

## 1. Pergunta

No B2.3 (lr 1e-3, `learning_starts` 10) a política do TD3 saiu da inicialização, mas saturou num **canto constante** do Box, independente do estado, e colapsou sob EB quando a₅ ≈ 0. A hipótese de condicionamento é a escala da recompensa oficial: a **soma** da loss sobre ~156 batches do teste, com dezenas a centenas por rodada e sem normalização. Com a recompensa bem condicionada, o TD3 aprende uma política **dependente do estado** e supera a ação fixa?

## 2. Configuração (uma só, fixada antes dos dados)

| parâmetro | oficial | B2.3 | **B2.3b** |
|---|---|---|---|
| recompensa | bruta | bruta | **normalizada** (`VecNormalize`: `norm_reward=True`, `gamma=0,99`, `clip_reward=10`; `norm_obs=False`) |
| learning_rate (ator e crítico) | 1e-5 | 1e-3 | **1e-4** |
| learning_starts | 100 | 10 | 10 |
| demais (MlpPolicy [256,128], buffer 1000, batch 64, train_freq 3, ruído N(0; 0,1), γ 0,99) | — | iguais | iguais |

- As **observações não são normalizadas**, então o ator continua avaliável pelas ferramentas do B2.2.
- A recompensa gravada no JSON é a **bruta**. O SB3 guarda a bruta no replay buffer e normaliza na amostragem para o treino; conferido em 2026-09-30: sd 86 → 0,5.
- `learning_starts = 10` é mantido igual ao B2.3 para que as diferenças sejam só a normalização e o lr.

## 3. Grade

- Sementes **100–104** (gastas; exploratório), LMP e EB → **10 runs** (~15–16 h de GPU; roadmap §4.1).
- Pareamento por (ataque, semente) com `fixed` e `td3` do Passo 2 e com o steelman do B2.3.
- Runner `scripts/passo2_oficial/run_b23b.py`, que reusa `run_b21.py` sem alterá-lo; launcher `run_grid_b23b.sh` (retomável, 6 processos).

## 4. Análise (pré-escrita em `scripts/passo2_oficial/analisar_b23b.py`)

- **Métricas:** primária, mediana da acurácia nas rodadas 401–500; secundária, média 451–500.
- **Comparações** (n = 10 pares, Wilcoxon bilateral, Δ, IC95, d; Holm por ataque, descritivo):
  - C1: B2.3b − fixed;
  - C2: B2.3b − td3 oficial;
  - C3: B2.3b − steelman do B2.3.
- **Mecanismo:**
  - drift_500 = média |π₅₀₀(s) − π₀(s)|;
  - **sd_estados** = média, sobre as 5 dimensões, do desvio-padrão de π₅₀₀(s) entre os estados das rodadas 401–500. É a estatística de constância, conforme o roadmap: o S_swap satura com o tanh;
  - S_swap, reportado sem critério;
  - curva de drift por checkpoint.

## 5. Critério do Portão B-a definitivo (fixado antes dos dados)

"O steelman aprende e supera a ação fixa" exige **as três** condições:
1. C1 com Δ > 0 e Wilcoxon p < 0,05 (primária);
2. mediana de drift_500 > σ_a = 0,0475;
3. mediana de sd_estados > σ_a = 0,0475 (a política depende do estado).

- **As três valem** → confirmar no **B2.3c** (sementes 115–124, confirmatório, pré-registro próprio).
- **C1 não vale** → **B-a definitivo**: nenhum dos dois steelmen supera a fixa. O B2.3c não roda.
- **C1 vale sem 2–3** → investigar (ganho sem política dependente do estado). **Não dispara** o B2.3c.

## 6. Cláusula de término (roadmap §2)

O B2.3b é o **último steelman**. Qualquer que seja o resultado, não há outra configuração de TD3 nesta linha. Os **dois** steelmen (B2.3 e B2.3b) são reportados no P2.

## 7. Limitações declaradas

- Exploratório, n = 10 pares, linhas de base do Passo 2 já inspecionadas.
- Duas mudanças em relação ao B2.3 (normalização e lr). O desenho não separa os efeitos, porque a pergunta é se **alguma** configuração bem condicionada aprende, não qual fator importa.
