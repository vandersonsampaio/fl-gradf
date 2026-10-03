# Adendo 1 — B2.7a: código no congelamento

**Data:** 2026-10-02 20:36:26 -0300, antes de qualquer run do B2.7a. O código do B2.7b terá o próprio adendo, também antes do disparo dele. O critério do B2.7b, fixado no `PLANO.md`, não muda.

## Implementação

- `scripts/b27b_td3_constante.py b27a`: roda o `td3_ref` (learner da ablação, modo td3) com `_reseed(seed)` antes, como no B2.7.
- O registro é feito por wrappers de `agent.select_action` e `learner._run_round` que só copiam valores e não consomem RNG: estado médio e ação executada a cada rodada, e `W_actor`/`b_actor` após 0, 50, 100 e 150 rodadas concluídas. O `train` numera as rodadas a partir de 1, e isso foi corrigido no snapshot antes do congelamento.
- **Smoke** (15 rodadas, `label_flipping` α 0,05, semente 42): acurácia idêntica ao `td3_ref` do B2.7 em H = 15 (|Δ| = 0).
- O lançador `scripts/run_grid_b27a.sh` roda 30 jobs, com 10 processos na CPU.

## Código no congelamento
d4c600290e5999b14919fd921750c0c6fd81b9782adb57ffab6a3f83ee2a6458  scripts/b27b_td3_constante.py
57923a053b402e7418c58710d086c4abe5be81a732d309b6c5b4dafc6968fdda  scripts/run_grid_b27a.sh
