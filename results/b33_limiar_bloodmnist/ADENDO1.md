# Addendum 1 — B3.3: runner test and code at freeze time

**Date:** 2026-10-09, before any grid run. Order from `PLANO.md` §8: plan (`ab42d47`) → code and §4 test → this addendum (commit) → launch. The plan and its criterion **do not change**.

## Runner test (PLANO §4)

2 rounds, seed 100, outside the grid (outputs in a temporary folder, discarded), 2026-10-09 12:00–12:01:

| a₅ | recorded `a_fixed` | steps | executed action = `a_fixed` in every step |
|---|---|---|---|
| 0.475 | [0.475, 0.475, 0.475, 0.475, 0.475] | 2 | yes |
| 0.75 | [0.475, 0.475, 0.475, 0.475, 0.75] | 2 | yes |
| 0.95 | [0.475, 0.475, 0.475, 0.475, 0.95] | 2 | yes |

→ **PASSED.** Only the action and the step count were checked; no accuracy was looked at.

## Implementation

- `scripts/passo2_oficial/run_b33.py`: reuses `run_b3.run` (BloodMNIST shim, official environment) and only replaces `R.A_FIXED` in the process; one output directory per a₅ (`raw/a5_<value>/`).
- `scripts/passo2_oficial/run_grid_b33.sh`: 15 runs (a₅ ∈ {0.475, 0.75, 0.95} × seeds 145–149, EB, 500 rounds), queue interleaved by seed, 6 GPU processes, resumable; records the PGID for `scripts/janela_execucao.sh`.
- `scripts/passo2_oficial/analisar_b33.py`: the pre-written analysis of PLANO §3/§5. Before using a run, it checks that the recorded `a_fixed` is the expected one. Dry-run on synthetic files outside the repository before this addendum: OK.
- The code comments and messages are in English (repository translation, commit `b591d67`); the hashes below are those of the current files. `run_b3.py`, `bloodmnist_shim.py`, `run_oficial.py` and `run_b21.py` differ from B3.1's frozen versions only in comments and strings (see `results/HASH_PROVENANCE.md`).
- BloodMNIST extractor and data: the same as B3.1 (hashes in `results/b31_medmnist_oficial/ADENDO1.md`, unchanged).

## Code at freeze time
881992db431add62453c72c48c2dd8a90592554c0a40bc465ae1a64bacd9220c  scripts/passo2_oficial/run_b33.py
15906b6ddcab489b492fc764f223b6c5494ebf687dded8d11c94219128fac875  scripts/passo2_oficial/run_grid_b33.sh
e3b68c0e9df7f17d4b93eb45e63c273b2d38261b82a40beebf02e478261465f9  scripts/passo2_oficial/analisar_b33.py
7a0290204e82702797c219ef6db25697483e7c53aa746407cf32c0835479155a  scripts/passo2_oficial/run_b3.py
45b0f352ddeb54a2c0e650f52299b2c6fbf061d56d151afb6f8873b36575bacf  scripts/passo2_oficial/bloodmnist_shim.py
88907e267295be9d197ec71ea0a98d1e7ae3798ad9f5d0b0844987383361d58f  scripts/passo2_oficial/run_oficial.py
b82e0c23b5d4bcee89e90d96d708ea2a4ab98e08c9acc8606414dd27661b4882  scripts/passo2_oficial/run_b21.py
de8cd79ec37daccafb6f52852d465d19d591c5cf45f69969291dd9df40d17d65  scripts/janela_execucao.sh
