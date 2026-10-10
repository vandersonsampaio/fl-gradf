> English translation of `NOTAS.md`. The Portuguese original is the frozen record (the hash of its §5 criterion is in `NOTAS.sha256`); if the two ever disagree, the original prevails.

# C.0 — remaining headroom: interpretation notes

**Date:** 2026-09-30 · Descriptive (C.0), exploratory seeds 42–51, in-house framework (MNIST, 10 clients, 2 Byzantine, 15 rounds, root 100).
**Files:** `analise.txt` / `espaco_restante_por_celula.csv` (ceiling = FedAvg without attack) and `analise_teto_corrigido.txt` / `espaco_restante_teto_corrigido.csv` (ceiling = best of the 5 rules without attack; gap also against fixed rules under attack). Scripts: `scripts/c0_espaco_restante.py`, `scripts/c0_teto_corrigido.py`.

## 1. The ceiling is FedAvg without attack at every alpha

Accuracy without attack (mean of 10 seeds):

| α | FedAvg | FLTrust | Krum | Median | Trimmed-Mean |
|---|---|---|---|---|---|
| 0.05 | **0.854** | 0.721 | 0.328 | 0.616 | 0.765 |
| 0.10 | **0.855** | 0.757 | 0.546 | 0.740 | 0.831 |
| 0.50 | **0.905** | 0.828 | 0.884 | 0.896 | 0.899 |

The correction **did not change the ceiling**. The hypothesis that motivated the correction ("FedAvg does not converge in 15 rounds and Krum advances faster") **was wrong**: without attack, Krum is the worst of all at α ≤ 0.1 (0.33 and 0.55).

## 2. The two cells with a negative gap are an attack artifact, not a lack of convergence

On `fltrust_aligned`, with α = 0.05 and α = 0.1, Krum **under attack** reaches ~0.895, above the ceiling (0.854), although without attack it stays at 0.33–0.55. This can only be explained if the attack itself helps Krum. `fltrust_aligned` is an informed attack that aligns the malicious update with the server update on the root; Krum picks that update (the most "central" one), which carries root-dataset information and works as an almost centralized step. **These 2 cells do not measure robustness** and must stay out of the remaining-headroom map (or be treated as a pathological case of the attack in this regime).

## 3. Remaining-headroom map (global gap = ceiling − best existing method, including fixed rules)

- **> 5 p.p. (4/21):** `label_flipping` α=0.05 and 0.1 (~14 p.p.), `sign_flipping` α=0.05 (12 p.p.), `low_mag_backdoor` α=0.05 (7 p.p.).
- **2–5 p.p. (6/21):** `gaussian_noise`, `krum_collusion`, `trim_attack` at α=0.05 (~4.9); `sign_flipping` α=0.1 (4.6); `trim_attack` α=0.1 (3.6); `low_mag_backdoor` α=0.1 (2.4).
- **≤ 2 p.p. (9/21):** practically all of α=0.5 (0.5–1.9), plus `gaussian_noise` and `krum_collusion` at α=0.1.
- **Excluded (artifact, §2):** `fltrust_aligned` α=0.05 and 0.1.

Against the **skeleton** (fixed/sr_only/AdaAggRL) the headroom is larger in 3 cells where a classic fixed rule already does better: `label_flipping` α=0.5 (Trimmed-Mean reduces the gap from 13.1 to 1.9 p.p.), `low_mag_backdoor` α=0.1 (6.1 → 2.4) and α=0.5.

## 4. Reading for Gate C0

- The remaining headroom **is not small everywhere**: it is concentrated at **high heterogeneity (α ≤ 0.1)** and on **label and sign attacks** (`label_flipping`, `sign_flipping`), exactly where S_R has a weak signal (Paper 1). At α = 0.5 almost everything is within ≤ 2 p.p. of the ceiling.
- Implication for GRADF-v2 (R1/R2): there is a possible accuracy gain **in the high-heterogeneity cells with attacks that the per-client signal does not detect**. Outside them, the differentiator would have to come from cost and privacy (R3/R4).
- No numeric threshold for "small" had been fixed; the 1/2/5 p.p. cuts above are descriptive. **The Gate C0 decision is the author's.**
- Limitation: 15 rounds, MNIST, 10 clients; the no-attack FedAvg ceiling at α ≤ 0.1 (0.85) is low in absolute terms, so "remaining headroom" here is relative to this short regime.

---

## 5. Gate C0 criterion (declared on 2026-09-30, BEFORE running the oracle ceiling)

Local record with a hash in `NOTAS.sha256`, uncommitted at the time (the author decided not to commit at that moment).

- **Oracle ceiling:** FedAvg without attack aggregating **only the 8 honest clients** (indices 2–9; the Byzantine ones are indices 0 and 1 in `byzantine_ids`), with the **same partitions** as C.0 (no repartitioning). Metric identical to C.0's: accuracy of the final global model averaged over the **10** clients' `X_test`. α ∈ {0.05; 0.1; 0.5} × seeds 42–51 × 15 rounds = 30 runs.
- **Cell with headroom:** gap against the oracle ceiling **> 2 p.p.** with the **whole CI95 above 0** (threshold fixed here).
- **Excluded cells:** `fltrust_aligned` at α = 0.05 and α = 0.1 (attack artifact, §2). 19 valid cells remain.
- The criterion is applied to the **global gap** (oracle ceiling − best existing method, including fixed rules). The skeleton's gap is reported alongside, without its own criterion.
- Recorded prediction (not a criterion): large headroom on `label_flipping` α ≤ 0.1 and `sign_flipping` α = 0.05; borderline on `sign_flipping`/`trim_attack` α = 0.1 and `low_mag_backdoor` α = 0.05; at the ceiling on `gaussian_noise`, `krum_collusion`, `trim_attack` α = 0.05 and almost all of α = 0.5.

---

## 6. Gate C0 decision (2026-09-30, application of the §5 criterion)

§5 hash checked before applying: `d0e07ea0…bc55d` (equal to the one in `NOTAS.sha256`). Full result in `analise_teto_oraculo.txt` and `espaco_restante_teto_oraculo.csv`; script `scripts/c0_teto_oraculo.py`.

### 6.1 Mechanical application of the criterion

**Cells WITH headroom (2/19):** `label_flipping` α = 0.1 (gap 8.0 p.p., CI95 4.0–12.0) and `label_flipping` α = 0.05 (6.0 p.p., CI95 5.5–6.5). → **target of R1** (accuracy) in GRADF-v2.

**Cells WITHOUT headroom (17/19):** all the others. → GRADF-v2's differentiator in them is **R3/R4** (cost and privacy at equal performance). Borderline case: `sign_flipping` α = 0.05 (gap 4.2 p.p., but CI95 −0.4 to 8.7).

**Skeleton vs. classic rule** (the cell's best method is a fixed rule, not the skeleton): `label_flipping` α = 0.5 (Trimmed-Mean; skeleton gap 12.4 p.p. vs. 1.3 for the global), `low_mag_backdoor` α = 0.1 and 0.5 (Trimmed-Mean), `fltrust_aligned` α = 0.5 (Median). → motivates **R2** and the C.2 direction 1 (per-client filter with memory followed by a robust rule); a **mandatory result in P2**: the skeleton dominates only where S_R separates the clients.

**§5 prediction vs. result:** `label_flipping` α ≤ 0.1 confirmed; `sign_flipping` α = 0.05 did **not** (the CI crosses 0); the predicted "at the ceiling" for α = 0.05 and α = 0.5 was confirmed.

### 6.2 Caveat: the oracle ceiling is not a ceiling at α ≤ 0.1 (a finding, not a change of criterion)

| α | oracle ceiling (8 honest) | FedAvg without attack (10) |
|---|---|---|
| 0.05 | 0.772 | 0.854 |
| 0.10 | 0.790 | 0.855 |
| 0.50 | 0.899 | 0.905 |

At α ≤ 0.1, removing the 2 clients costs ~7–8 p.p.: with strong heterogeneity, they hold classes poorly represented in the others, and the (IID) test set charges for that. That is why **10 of the 17 "no headroom" cells have a negative gap with the whole CI95 below 0** (down to −5.4 p.p.): methods under attack beat the "ceiling", because several of the simulator's attacks (`gaussian_noise`, `trim_attack`, `krum_collusion`…) start from the attacker's true update and still carry part of the information in its data.

Declared consequences:
- **The 2 cells with headroom are robust:** `label_flipping` α ≤ 0.1 has headroom against both ceilings (8.0/6.0 p.p. against the oracle; 14.6/14.2 against FedAvg with 10).
- **At α = 0.5 the "no headroom" classification is reliable:** the two ceilings almost coincide (0.899 vs. 0.905) and the gaps are ≤ 1.3 p.p. in both.
- **At α ≤ 0.1, "no headroom" is not demonstrated** for the 7 cells other than `label_flipping`: neither ceiling is a valid upper bound there (FedAvg with 10 overestimates what is achievable under attack; the oracle with 8 underestimates it). The criterion was applied as frozen; **reinterpreting these cells is the author's decision** and would call for a ceiling that preserves the information in the attackers' data (e.g. an "attack oracle": the same clients sending the honest update), to be registered before running.

### 6.3 Declared limitations
- With 15 rounds, the gap mixes robustness with convergence speed → resolve in **B2.7** (include the C.0 cells in the horizon grid).
- **The `low_mag_backdoor` ASR was not logged** (neither in the ablation nor in exp9): clean accuracy may hide a successful backdoor. Recorded as a limitation.
- Krum's selection on `fltrust_aligned`: not logged; confirming the artifact is left for a later phase (C.3).

→ **Gate C0 closed by the frozen criterion**, with the §6.2 caveat about α ≤ 0.1 pending the author's decision.
