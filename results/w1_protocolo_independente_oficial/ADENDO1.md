# Addendum 1 — W1: code and checks (PLANO §6)

**Date:** 2026-10-25, before any run on seeds 160–179. Plan hash re-verified (`PLANO.sha256`: OK).

## Part A — change to check 2 (written before running the new check)

**What happened.** Checks 1 and 3 pass (`verificacao.txt`). Check 2, as written in PLANO §6 (a 5-round CPU run with and without the held-out evaluation must give identical histories, |Δ| = 0), fails: the two histories differ from round 1. Diagnosis (development runs only, seed 160, 5 rounds, MNIST EB fixed, outputs not kept in the repo): **two runs with the same seed and no held-out evaluation also differ**, by 1–3 p.p. of validation accuracy per round. This holds with the default threads, with a single torch/OMP/MKL thread, and with `PYTHONHASHSEED=0`. So the official code is not deterministic even on CPU (previously known only for the GPU, B2.4r), the non-determinism comes from inside the official code, and |Δ| = 0 cannot hold for any instrument. The check as written cannot discriminate.

**Replacement (approved by the author on 2026-10-25).** Check 2 is replaced by a statistical form, `run_w1.py verificar_ruido`:

- MNIST, EB, fixed, 5 rounds, CPU, 1 thread, development seeds **150–154** (not used anywhere else in the project; no held-out result of seeds 160–179 is involved).
- Per seed: two runs without the held-out evaluation (A, B) and one with it (E).
  - noise_s = mean over rounds 1–5 of |A − B| (`history_acc`);
  - eval_s = mean over rounds 1–5 of (|E − A| + |E − B|) / 2.
- **PASS** if all three hold:
  1. mean(eval_s) ≤ 1.5 × mean(noise_s);
  2. the one-sided paired Wilcoxon test of eval_s > noise_s has p ≥ 0.05;
  3. every E run reports `rng_intact` = true, i.e. the torch (CPU/CUDA), numpy and random states are bit-identical before and after every held-out evaluation (the direct state comparison that PLANO §6.2 also asks for).
- If it fails, the grid is not launched and the author is consulted.

**Code changes (before any confirmation run):**

- `HeldOutAdapter.step` now restores the network's train/eval mode after the evaluation (the official `test()` leaves the network in eval mode; the official local training calls `net.train()` itself, so this is a precaution) and records `rng_intact` per step.
- `run --no_eval` (checks only; refused for the grid directory) and the `verificar_ruido` subcommand.

The original check-2 output stays in `verificacao.txt` as evidence.

## Part B — check results (written after the checks, before the grid)

1. **Sizes and disjointness** (`verificacao.txt`): MNIST pool 60,000 → train_rest 50,000 + validation 10,000, test 10,000; BloodMNIST pool 13,671 → 10,250 + 3,421, test 3,421; validation ∩ train_rest = ∅ and validation ⊂ non-test pool. **OK.**
2. **Held-out evaluation does not perturb the run**, statistical form from part A (`verificacao_ruido.txt`):

```
[2'] MNIST EB fixed, seeds [150, 151, 152, 153, 154], 5 rounds on cpu (1 thread), history_acc (p.p.):
     seed 150: noise (no-eval vs no-eval) 0.801; eval vs no-eval 1.168; rng_intact True
     seed 151: noise (no-eval vs no-eval) 2.296; eval vs no-eval 2.390; rng_intact True
     seed 152: noise (no-eval vs no-eval) 1.601; eval vs no-eval 1.415; rng_intact True
     seed 153: noise (no-eval vs no-eval) 1.244; eval vs no-eval 1.339; rng_intact True
     seed 154: noise (no-eval vs no-eval) 1.897; eval vs no-eval 1.950; rng_intact True
     mean noise 1.568; mean eval 1.652 (ratio 1.05, limit 1.5); Wilcoxon eval > noise p=0.2188 (limit ≥ 0.05) -> OK
```
   **OK.** The original |Δ| = 0 form fails for the reason given in part A (`verificacao.txt`, kept as evidence).
3. **Reward scale:** validation batches = test batches (156 MNIST, 53 BloodMNIST). **OK.**
4. **GPU smoke** (10 rounds, EB, seed 160, td3 and fixed, both datasets; `verificacao/smoke_gpu/`): all 4 runs finish on cuda:0 and write every field (`split`, `history_acc`, `resets`, `held_out`, `rng_intact` = true, batch counts 156/156 and 53/53). Only the fields were inspected, not the held-out values. td3 runs log 12 steps for 10 timesteps (SB3 collects in chunks of `train_freq` = 3); the analysis reads the first 500 steps. **OK.**

## Code hashes (sha256)

```
eee4dfff3de8ee57530c420ecfe855ba4495881194d2b12134a77d86a35dd726  scripts/passo2_oficial/run_w1.py
179bbd69badb1608c1cc42f6836b56e1e78b910923b83113ae6a5075ba682e25  scripts/passo2_oficial/analisar_w1.py
0679fa5231a1bbd0dc2b379f9128c9a07564314b553d603d75c246365a9748a6  scripts/passo2_oficial/run_grid_w1.sh
b82e0c23b5d4bcee89e90d96d708ea2a4ab98e08c9acc8606414dd27661b4882  scripts/passo2_oficial/run_b21.py
7a0290204e82702797c219ef6db25697483e7c53aa746407cf32c0835479155a  scripts/passo2_oficial/run_b3.py
45b0f352ddeb54a2c0e650f52299b2c6fbf061d56d151afb6f8873b36575bacf  scripts/passo2_oficial/bloodmnist_shim.py
```

External code: `external/AdaAggRL` at commit 27b9c18, unedited.
