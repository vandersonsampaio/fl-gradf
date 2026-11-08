# Addendum 1 — W1 rerun: code, before the checks and the grid

**Date:** 2026-11-01, before any run on seeds 220–239 and before the checks of `PLAN.md` §2. The plan's hash was recorded before these code changes (`PLAN.sha256`).

## What was done

- **`run_w1.py`:** `HeldOutAdapter` evaluates `inner.aggregate_weights` on a separate copy of the network and no longer touches `inner.net` (except, in the checks only, to log the old measure). It repeats check 1 at round 1 and every 10 rounds, and aborts on a mismatch. Each run records `eval_target` and `global_checks`. New subcommand `verificar_global`. Output folder and seeds changed as in the plan.
- **`analisar_w1.py`:** new paths and seeds; refuses runs without a verified series; one added descriptive column. Tests and margins are unchanged.
- **`run_grid_w1.sh`:** new paths and seeds. The controller of the pause window is no longer started.
- **`run_w1_rerun_chain.sh` (new):** waits for the P1 parity chain, runs the checks and starts the grid only if they pass.

## Test done so far

The adapter was run against a fake environment with a tiny network, CPU, in the official venv. The fake step leaves `inner.net` with a different "client" weight, as the official step does. Results: the held-out series follows the global model and the old measure does not; `net_is_global` is false; `rng_intact` is true; and a corrupted logged value makes the guard abort at round 1.

The checks on the real environment (`PLAN.md` §2) **have not run yet**. They run right after the P1 chain ends, because available RAM is limited while it runs. Their output goes to `verification.txt`.

## Code hashes (sha256)

```
75678cd7fdf8f1a908bcc3feb7cfc23179a1bccfb416638210e59bbd649b02c0  scripts/passo2_oficial/run_w1.py
11c54e5e1f87d7981c77de22c6aaacc17c6f70908537ecc58eb159a70a3fba42  scripts/passo2_oficial/analisar_w1.py
d0c5c57cde5c6ce61d19e7c1fbca8b3d3d33d32a88c46f890add0abe69ec2f75  scripts/passo2_oficial/run_grid_w1.sh
53b95264e5bc2757a9e20bf46c89a3a4f39a8c076e52e772dc9708e1ab7545ac  scripts/passo2_oficial/run_w1_rerun_chain.sh
b82e0c23b5d4bcee89e90d96d708ea2a4ab98e08c9acc8606414dd27661b4882  scripts/passo2_oficial/run_b21.py
7a0290204e82702797c219ef6db25697483e7c53aa746407cf32c0835479155a  scripts/passo2_oficial/run_b3.py
45b0f352ddeb54a2c0e650f52299b2c6fbf061d56d151afb6f8873b36575bacf  scripts/passo2_oficial/bloodmnist_shim.py
```

External code: `external/AdaAggRL` at commit 27b9c18, unedited. The previous versions of the three modified scripts match the hashes in `results/w1_protocolo_independente_oficial/ADENDO1.md` and are in commit `ba98905`.
