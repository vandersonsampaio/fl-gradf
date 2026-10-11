# Adendo 1 — B2.7c reformulado: código no congelamento

**Data:** 2026-10-11 11:18:56 -0300, antes de qualquer run das sementes 82–91. O plano (`PLANO.md`, hash em `PLANO.sha256`) não muda.

## Implementação

- `scripts/b27c_b025_vs_td3.py run`: `_reseed(seed)` e depois o learner da ablação em modo td3. No `fixed_b025`, `select_action` → constante [0,5]×4 com b = 0,25 (via `scripts/b27b_explicativo.constant_for`, inalterado) e `train_step` → no-op. É o mesmo caminho do B2.7b.
- `analisar`: t pareado unilateral por célula, Holm nas 3 células (α = 0,05), veredito do PLANO §2, H1b agregada e os descritivos. Foi testado em dados **sintéticos** no scratchpad, fora do repositório: os p e a decisão de Holm conferem à mão.
- Lançador `scripts/run_grid_b27c.sh`: 60 jobs, 12 processos na CPU, retomável.

## Verificação (PLANO §3): smoke de 15 rodadas, semente 42, `label_flipping` α 0,05

- td3_ref: new 0.719400 vs reference 0.719400 |Δ| = 0.00e+00 OK
- fixed_b025: new 0.725400 vs reference 0.725400 |Δ| = 0.00e+00 OK

## Código no congelamento
4ca49c72f1972891f5c361847481971240b9334a70f389488077a4c286de2d8e  scripts/b27c_b025_vs_td3.py
f93c1c51e166be007cb88a9c7041dcb2e0fb798da2228539539ee50ad401cedf  scripts/run_grid_b27c.sh
7b4c3fd2d8692e9156c938031f2548a71ceb482e5e9fc6f7d28067734d893d34  scripts/b27b_explicativo.py
c5310907f80fee535196b22cac3d5e2d71929d9c9cf9987b773a87cd5ae8df3c  scripts/b27_horizonte.py
4525cd80e2d0b9ec70be759f4323e7285cbf4feac6107e89e93ec20a20de2e41  scripts/frente1_ablacao_adaaggrl.py
