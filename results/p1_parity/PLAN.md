# P1 step-parity reruns (items 1–4)

**Date:** 2026-10-30 16:30 -0300, before any of these runs, on top of commit `2fb7c63`.

## Decision

FLTrust's server update g₀ is computed with **step parity** with the clients. This is the main version for every new P1 run (MNIST rerun and CIFAR-10 campaign). The epoch-based version stays as a sensitivity analysis.

The decision follows the specification, as fixed in `results/p1_cifar10/e0b/ADDENDUM3.md`, not the numbers. FLTrust (Cao et al., NDSS 2021, Alg. 1–2) gives server and clients the same number of local iterations. The diagnostic (`results/p1_g0_probe/RESULTS.md`) only shows that it matters.

The mapping is server steps = median over clients of ceil(n_i / batch) × local epochs, computed from the partition only. It **preserves the specification's step parity**. It is not faithful to the paper's values (R_l, b, β), because here the clients train by epochs.

## Implementation

- **`FederatedLearner(server_steps=...)`**, with values `"native"` (default), `"median"` or an int. One method, `FederatedLearner._fit_server`, now computes g₀ at every call site:
  - `federated_learner.py` (`_run_round`);
  - `attacked_learner.py` (FLTrust rule, informed attacks, `compute_param_updates_auto`);
  - `gradf_learner.py`;
  - `data_loader.py` (GRADF pretraining);
  - `exp10_selector_comparison.py` (FedStrategist, Discrete+AdaAggRL-detector);
  - `exp4_adaptive.py` (not used by P1; changed for consistency).
- **Native path:** unchanged, `model.fit(root)`.
- **Parity path:** `steps` mini-batches drawn from the root with the model's RNG (the server RNG when the run is seeded) and one pass. This is the procedure of `scripts/p1/g0_probe.py`.
- **CLI:** `exp9_dominance_grid` and `exp10_selector_comparison` take `--server_steps`. Non-native outputs get `_steps<value>` in their names and a `server_steps` column.
- **`e0_pilot.py`:** takes `--server_steps`. Non-native runs go to `results/p1_parity/cifar_blockC/raw` with `_st<value>` in the name, and save ‖g₀‖ per round and the step count. Native runs are unchanged.
- **`cost_probe_d.py` (part D):** every system now runs with `server_steps="median"`, the campaign's setting, and FLTrust is timed as its own fixed rule. The rest is as in `results/p1_cifar10/e0b/ADDENDUM2.md`.

## Checks before this hash

- **T1, native, exp9 seeded:** seed 42, root 100, α ∈ {0.5, 0.05} × {`trim_attack`, `fltrust_aligned`, `label_flipping`} × {fltrust, trimmed_mean, krum}. All 18 rows equal `results/p1_invariance/raw/root100_seed42`, to the CSV's text precision: 17 bit for bit, 1 within 1 ulp.
- **T2, parity, exp9 seeded:** seed 0, root 100, α ∈ {0.5, 0.05} × {`trim_attack`, `sign_flipping`}, fltrust. The 4 final accuracies equal the `median` runs of `results/p1_g0_probe/raw` bit for bit.
- **T3, native, exp10 seeded:** seed 3, α 0.05, {`trim_attack`, `label_flipping`}, 3 rounds; Random, attack-aware and FedStrategist (variant b), pretraining included. Results equal those of the `2fb7c63` source (`git archive`, run from a scratch copy) bit for bit.
- **T4, parity, CIFAR-10 CNN:** `e0_pilot.py --server_steps median`, 2 rounds, seed 3, root 100, `trim_attack`. 303 server steps, ‖g₀‖ 0.93 and 1.33 (native at this root: ≈ 0.05). The output was moved out of the repository.

## What runs (`scripts/p1/run_parity_chain.sh`)

The chain starts when W1 has finished (86 final JSONs, no W1 process). The author authorized it on 2026-10-30 to start as soon as W1 finishes, inside the Mon–Fri 8–18 window if need be. Order: items 1 and 3 together, then item 2, then item 4 alone.

| item | what | jobs | expected wall time |
|---|---|---|---|
| 1 `mnist_fixed` | exp9 seeded, `--server_steps median`, seeds 42–51: root 100 with the 7 fixed rules; roots 3,000 and 50 with fltrust, median, trimmed mean, Krum (the `p1_invariance` design) | 30 | ~1–1.5 h (8 processes) |
| 3 `cifar_blockC` | E0b block C with parity: α 0.05 × {`sign_flipping`, `trim_attack`} × root {100, 2,500} × seeds 0–2, 180 rounds | 12 | ~1–1.5 h (12 processes) |
| 2 `mnist_rq3` | exp10 seeded, `--server_steps median`, root 100, seeds 42–51: variant b with Random, attack-aware, GRADF, FedStrategist; variant a with FedStrategist (GRADF's detector). No AdaAggRL (it does not use g₀) and no Discrete+AdaAggRL-detector (deferred) | 20 | ~3 h (10 processes) |
| 4 part D | `run_cost_probe_d.sh 12` with the parity code → `results/p1_cifar10/e0b/costs.json` | 12 | ~1 h |

## Notes fixed before the runs

- **Free regression check.** Under item 1, every non-FLTrust rule outside `fltrust_aligned` must equal `results/p1_invariance` exactly. The server fit uses its own RNG and only `fltrust_aligned` reads g₀ through the attack. Mismatches are reported, not explained away.
- **`fltrust_aligned` changes for every rule.** Its Byzantine update is g₀ itself, so its cells change for all rules under parity, not only for FLTrust.
- **Item 2 is seeded** (`--seed_rng`), unlike the published exp10 runs. GRADF's DQN remains non-deterministic across executions (`fl-fedmed/results/d2_variancia_p1`). Tables are reported as regenerated with the corrected implementation. No change is attributed to the correction by comparing with the published runs.
- **Item 3 replaces E0b block C** for σ_E0^E1, σ_E0^E3 and the τ_CNN check, read with the Sec. 3–4 formulas of `results/p1_cifar10/e0b/PLAN.md` at R = 110. The native block C is kept as the sensitivity version.
- **Next, before the pre-registration:** the analysis (LOSO, d\*, headroom, Tables 1–5, σ_E0, τ, budget decision with the E1 factorial) and the guard of the confirmatory campaign, which records `server_steps` in every job and aborts a job in seeds 42–51 whose value is not `median`, unless it is flagged as the E1 native arm.

## Code hashes (sha256)

```
e6fc76e4c437e1b2569ad10830b97181ebadb89623f952721d0e4820a69a67ec  src/fl/federated_learner.py
c7f1e6a34026e223c3a48b92bb82a429be174ec715c2b2b30652080774004a8b  src/fl/attacked_learner.py
04dd54391a8854a7e8a6ff06c05d412ffd86a94a47901b9cb961545258ff3b24  src/fl/gradf_learner.py
28285584be823b45ec3ea81c9ac14560e284c29ec635bc2dfe45688ef8da4c93  src/utils/data_loader.py
e2438485002cb90daca1be0b0ab52a02013754256f0082ab5ace4736d325e9b5  src/experiments/exp4_adaptive.py
03bf633324f0f2dbb2be29df16634b1c89d3cae672b037332f786de65c5c972b  src/experiments/exp9_dominance_grid.py
090aabf888edd0788704037ee39a1485def5dc44fb0ea0d31c8502ae6211673b  src/experiments/exp10_selector_comparison.py
509ccf9a546c31cf092e754b919e73cb13d41a4021766e0cd7a722c4eba9f0ba  scripts/p1/e0_pilot.py
a3e4852627c18f661d54f8d75e5552ec4e0377fda5c058bb8c3625b03477e90c  scripts/p1/cost_probe_d.py
cfa7db7776acaaa6c1ed987f92bc7e3e91fbadfe22c0de0b45274488dab60304  scripts/p1/run_cost_probe_d.sh
64c8d432f961e853a111c6a947dba789b3565c8004ed9482f94a20e20b81317d  scripts/p1/run_parity.sh
e86951f13e55419f0e7ae92bc6d5afcef2ff4e25ab933d64ed6959856bfd1e24  scripts/p1/run_parity_chain.sh
```
