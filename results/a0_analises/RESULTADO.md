# Result — A0: cheap analyses

**Date:** 2026-10-01. Plan frozen in `PLANO.md` (English translation in `PLANO.en.md`; hash in `PLANO.sha256`, commit `79f9ce1`) before the runs and the analyses.
**Outputs:** `analise.txt` (items a–c), `concordancia_sinais.csv`, `massa/*.csv` (48 runs), `analise_regimes_td3.txt` and `regimes_td3_por_checkpoint.csv` (item d).

## (a) Weight mass on attackers → P3 direction 6 without motivation

- **Check:** the 48 runs exactly reproduce B2.6 (max. |Δacc| = 0). Recording the weights did not perturb the trajectory.
- **Pre-registered rule:** "real exploitation" requires a mean mass ≥ 0.05 in ≥ 4 of the 8 cells. Result: **2/8 cells**, both for `sr_only` and `sr_bin` → **no relevant exploitation.**
- In 6 cells (`gaussian_noise`, `krum_collusion` and `trim_attack`, at α = 0.05 and 0.1), the mass is **0.0000–0.0001**: the attackers are fully excluded. The gain over the FedAvg-8 oracle in those cells (C0 negative gap) therefore comes from **how the honest clients are weighted**, not from using the attackers. *Clarification of 2026-10-04:* concretely, from the **uniform** weighting of the honest clients vs. the oracle's **sample-size** weighting. Since the metric is the accuracy on the IID, balanced global test set, and the labels are skewed per client, sample-size weighting unbalances the classes; the uniform oracle-8 reproduces the best methods (mechanism corrected on 2026-10-05) (`results/c0b_espaco_h150/VERIFICACOES.md` §3; `results/c0_espaco_restante/CORRECAO_2026-10-04.md`).
- The 2 exceptions are not "exploitation" in the sense of direction 6:
  - `low_mag_backdoor` α = 0.1: mass ≈ 0.19, close to uniform (0.2). S_R does not separate this attack (AUC 0.51 in item c), so the attackers **get through**; they are not exploited.
  - `sign_flipping` α = 0.1: mass 0.06–0.07, concentrated in rounds 2–7 (0.08–0.29), which drop to ~0 after round 8. It is a late-exclusion transient.
- **Reading:** the C0 negative gaps are **not** explained by exploiting the attackers. Direction 6 ("exploit instead of exclude") loses its empirical motivation.
- *Note:* in these variants the seed changes the trajectory little (the partition is fixed; the seed only affects the root and the agents). For `sr_bin`/`sign_flipping`, the 3 seeds give the same mass per round.

## (b) Signal-selection ceiling → direction 1 headroom is ~+1.7 p.p.

Mean over B2.6's 19 valid cells, per seed (n = 10):

| estimate | accuracy (%) | CI95 |
|---|---|---|
| per-seed max. (biased upwards) | 84.30 | 83.72–84.88 |
| per-cell choice, in-sample | 83.59 | 82.82–84.36 |
| **per-cell choice, LOSO (reference)** | **83.24** | 82.61–83.87 |
| sr_b025 (best fixed single signal) | 82.47 | 82.40–82.53 |
| sr_cosserver | 82.11 | 81.01–83.21 |
| sr_bin | 81.92 | 81.52–82.32 |
| cosserver_only | 81.32 | 80.48–82.16 |
| sr_only | 80.90 | 80.51–81.28 |

- **Gain of the per-cell signal choice (LOSO) over the best single signal per seed: +1.75 p.p. (CI95 +1.31 to +2.19).** This is the headroom that P3 direction 1 competes for. It is small, but its CI lies entirely above zero. Against the fixed `sr_b025`, the margin drops to ~+0.8 p.p.
- The choice pattern is readable: **cos_server** on `label_flipping`, `sign_flipping` and `low_mag_backdoor` (attacks that S_R does not separate); **S_R** on `gaussian_noise`, `krum_collusion` and `trim_attack` at low α.

- *Note of 2026-10-04:* (1) this value was superseded by C0b at H = 150: **+1.24 p.p.** over the best single signal per seed (LOSO, CI95 +0.98 to +1.49; `results/c0b_espaco_h150/RESULTADO.md` (ii)); (2) it is the ceiling of **choosing one signal per cell** (the same signal for all clients and rounds), and **does not bound a per-client combination of the signals**.

## (c) S_R × cos_server agreement (preliminary: seed 42, td3 trajectory)

- **Mean per-round Spearman across clients:** from −0.36 to +0.35. The two signals are **nearly orthogonal**.
- **Complementary AUC:**
  - **S_R** separates `gaussian_noise` (≈ 1.0), `krum_collusion` and `trim_attack` (0.86–0.97);
  - **cos_server** separates `label_flipping` (0.80–1.0) and `sign_flipping` (0.83–0.99);
  - neither separates `low_mag_backdoor` well;
  - on `fltrust_aligned`, cos_server is **inverted** (AUC 0), a known artifact of the attack.
- **Reading:** a non-learned selector (e.g. "use the signal with the highest separation") is plausible, because the signals fail on different attacks. Confirmation requires more seeds.

## (d) Three TD3 regimes (P2 figure)

Median across runs, states of rounds 401–500, σ_a = 0.0475:

| checkpoint | drift published | drift B2.3 | drift B2.3b | sd_estados published | sd_estados B2.3 | sd_estados B2.3b |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.0046 | 0.0047 | 0.0045 |
| 100 | 0 | 0.314 | 0.023 | 0.0046 | 0.0195 | 0.0048 |
| 250 | 0.0035 | 0.467 | 0.063 | 0.0046 | 0.0010 | 0.0072 |
| 500 | **0.0097** | **0.470** | **0.174** | **0.0047** | **0.0011** | **0.0134** |

- **Published:** the policy barely moves (drift ≪ σ_a) and does not depend on the state.
- **B2.3 (lr 1e-3):** the policy moves a lot, saturates at a corner of the Box by step ~200 and becomes **even less** state-dependent (sd 0.001).
- **B2.3b (lr 1e-4, normalized reward):** the drift grows monotonically and `sd_estados` triples (0.0045 → 0.0134). Even so, it stays at ~¼ of σ_a in 500 rounds. It is the only configuration with growing state dependence, which is the already-recorded horizon limitation (a long horizon on the official code was cut).

## Decisions for P3

1. **Direction 6** (exploit attackers): **no motivation** by the pre-registered criterion.
2. **Direction 1** (signal selection): real but modest headroom (~+1.7 p.p. over the best single signal per seed; ~+0.8 over `sr_b025`). The signals are complementary by attack type.
   - *Note of 2026-10-04:* (1) this value was superseded by C0b at H = 150: **+1.24 p.p.** (LOSO, CI95 +0.98 to +1.49; `results/c0b_espaco_h150/RESULTADO.md` (ii)); (2) it is the ceiling of **choosing one signal per cell**, and **does not bound a per-client combination of the signals**, which can use different signals for different clients in the same round.
