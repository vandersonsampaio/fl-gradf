# Pre-registration — P1 confirmatory campaign on CIFAR-10

**Status: FROZEN on 2026-11-04, before any run on seeds 42–51 of CIFAR-10.** Content approved by the author on 2026-11-03; the checks of Sec. 8 passed (`checks/SUMMARY.md`). Hashes: `PREREGISTRATION.sha256` for this file, `CODE.sha256` for the code. Changes go through dated addenda.

**Base plan:** `docs/plano_p1_completo.md`, which applies except where this document differs. Sec. 9 lists every difference.

## 1. Questions

- **RQ1 (E1):** does FLTrust's advantage over the trimmed mean depend on the size of the root?
- **RQ2 (E2):** is there headroom across conditions outside the sample?
- **RQ3 (E3):** how much of it do adaptive systems capture, compared with a random selector and with the best fixed rule?

P1 does not attribute causally a change in capture to internal components of any system, and it does not present GRADF's hardening layers as a defense (Sec. 7).

## 2. Common setup

| item | value | source |
|---|---|---|
| Dataset, model | CIFAR-10, small CNN | plan, Sec. 5.0 |
| Clients | 10, of which 2 Byzantine | plan, Sec. 5.0 |
| Rounds | R = 110 | `e0b/RESULTS.md` |
| Local training | 2 epochs, `local_lr` 0.01 | E0b |
| Server update | `server_steps = median` (step parity) | `p1_parity/PLAN.md` |
| Global rate α | **factor 1/3, common to all rules and systems** | author, 2026-11-03 |
| Root | 100 (canonical), 2,500 (enlarged) | plan, Sec. 5.0 |
| Heterogeneity | α_Dir ∈ {0.5, 0.1, 0.05} | E0 |
| Seeds | 42–51 | plan, Sec. 5.0 |
| Layer-1 threshold | τ_CNN = 7.86, not re-tuned | `p1_parity/RESULTS.md` |
| Primary metric | **mean test accuracy over rounds 106–110** | author, 2026-11-03 |
| Execution | CPU, 1 thread per process; on this PC at most 6 processes while W1 exists, up to 12 after; 4 on each VM | author, 2026-11-03 and 2026-11-04 |

**The global rate.** The factor is α in w ← w + α·g, a parameter of FLTrust's method that the paper sets per dataset and in common for all methods compared (Cao et al., NDSS 2021, Sec. VI-A). It was tuned off the confirmatory seeds with no attack, by training error (`p1_parity/lr_tuning/`). **The 1.0 p.p. margin of that tuning failed on both datasets:** FedAvg and the trimmed mean are best at factor 1 and FLTrust at smaller factors. The paper states this.

**How the factor is applied:**
- to the aggregate of every rule and system, before Layer 4 evaluates the candidate, so that Layer 4 judges the step actually taken;
- in AdaAggRL, as w ← w + α·(Σ wₖθₖ − w);
- to the runs that pretrain GRADF and the attack-aware reference, which produce their experience at the same factor.

The default `global_lr = 1` must reproduce earlier results exactly (Sec. 8).

**The final round** is recorded for every run and reported as descriptive. All curves are stored.

## 3. Grids and arms

**Full grid (E1, E2):** 3 α × 7 attacks = 21 cells. **Common confirmatory grid (E3), level 1:** 3 α × {`sign_flipping`, `label_flipping`, `trim_attack`, `krum_collusion`} = 12 cells, the author's decision of 2026-11-03.

| block | what | factor | cells | runs | cost, h at 12 processes |
|---|---|---|---|---|---|
| **E1 main** | 7 fixed rules at root 100, FLTrust at root 2,500 | 1/3 | 21 | 1,680 | 106.4 |
| E1 invariant | trimmed mean at root 2,500: `fltrust_aligned` (30 runs), one seed on the other 18 cells | 1/3 | 21 | 48 | 3.0 |
| **E1 sensitivity, factor** | FLTrust at both roots, trimmed mean at root 100, trimmed mean at root 2,500 under `fltrust_aligned` | 1 | 21 | 660 | 41.8 |
| **E1 sensitivity, server steps** | FLTrust with `server_steps = native` at both roots | 1 | 21 | 420 | 26.6 |
| **E3 main** | Random, attack-aware, GRADF, FedStrategist, AdaAggRL | 1/3 | 12 | 600 | ~50, with pretraining |
| **E3 hardened** | FLTrust, median, trimmed mean and Krum through Layers 1, 2 and 4 | 1/3 | 12 | 480 | ~30 |
| E3 sensitivity | AdaAggRL with pretrained extractor, FedStrategist with GRADF's detector | 1/3 | 12 | 240 | ~19 |
| | **Total** | | | **~4,130** | **~277** |

The costs come from one cell of part D at 12 processes, and there is no margin.

**Budget.** The plan's limit was 236 h, and its frozen rule picks the lowest level that fits it. **The author accepted the overrun on 2026-11-03,** so level 1 is a declared departure from that rule, not its output. Under 236 h only level 2 would fit.

**Measured after the costs above:** the arms with pretraining (GRADF, attack-aware, FedStrategist with GRADF's detector, 360 runs) each add 136 pretraining rounds, about 49,000 rounds or 10% over the 454,000 of the runs themselves, which may exceed the E3 figures. It changes no setting.

**Order:** the invariant check first; E1 main; then the sensitivity blocks; E3 last, since its analysis needs the fixed rules. Expected end: about 10 to 12 November with this PC and 2 VMs, starting on 4 November. This is an estimate and no result depends on it.

**Machines.** The campaign runs on this PC and on 2 cloud VMs (Azure D4as_v7, AMD EPYC 9V45). **Assignment is free:** with `ONEDNN_MAX_CPU_ISA=AVX2` and `OPENBLAS_CORETYPE=Haswell` on the VMs, final parameters match this PC bit for bit on FedAvg, FLTrust, Krum, Random, FedStrategist and AdaAggRL with both extractors, 2 rounds at factor 1/3 (`checks/SUMMARY.md`). The guard refuses a job on an AVX-512 machine without those settings. Every job records its machine. GRADF's DQN is not deterministic on any machine, so no match is expected or claimed for it. A machine effect on systems with learned components is not excluded by this check, and the machine of each job is available for a descriptive check.

## 4. Statistical protocol

As in the plan, Sec. 4.3, implemented once in `scripts/p1/inference.py`:

- **Unit: the seed.** Each aggregate reduces a seed to one value, the mean over the cells of the grid in use.
- Paired Wilcoxon, with the t-test as a complement. Holm within each family. Effect size and CI95 everywhere.
- **Categories:** superiority (CI95 above 0), inferiority (CI95 below 0), equivalence (TOST, CI90 within the margin), no evidence otherwise. Inferiority is checked before equivalence.
- **Margin:** δ = 1.0 p.p., not relaxed. Sensitivity: δ = 0.5·σ_E0, a different question (Sec. 6).
- Win counts per cell are descriptive.

## 5. Hypotheses

**E1 (one hypothesis).** With V(c, s, r) = Acc_FLTrust(c, s, r) − Acc_TM(c, s) and D_s = mean_c [V(c, s, 2,500) − V(c, s, 100)] over 21 cells: **H1: D > 0,** tested at factor 1/3.

**Reading of the sensitivity at factor 1, fixed now.** D is also tested at factor 1, in the same way:
- **Same category at both factors:** the conclusion is stated without qualification about the factor.
- **Different categories:** the root effect is reported as **dependent on the global rate**, and no general claim about the root is made. Pilot data point this way on 2 cells (`cifar_factor_sweep/RESULTS.md`), so this outcome is plausible and informative.
- The same is reported for the level of V, that is, whether FLTrust leads the trimmed mean, since that sign changed between factors in the tuning.

**The server-steps sensitivity** runs at factor 1, the author's decision: factor 1/3 was tuned under step parity, and the native g₀ has a much smaller norm (about 0.05 against about 1), so a third of that step would be a different experiment. It is reported as a factorial of steps × root against FLTrust under parity at factor 1, with no separate confirmatory claim.

**E2.** Ĥ_LOSO on the 21 cells, at root 100, factor 1/3. **H2: the CI95 of Ĥ_LOSO lies above 0,** from a seed-level bootstrap that repeats selection and evaluation in each resample. In-sample headroom is reported as an optimistic upper bound. No p-value from the bootstrap.

**E3.** On the 12 cells, with d\*, B and O recomputed on them. Two families, each with Holm:
- A_sys − A_Random, for GRADF, attack-aware, FedStrategist, AdaAggRL;
- A_sys − B₋ₛ, for the same systems and Random.

**Like-for-like comparison.** GRADF and Random pass through Layers 1, 2 and 4, and the plain fixed rules do not. So for GRADF and Random the comparison with B₋ₛ is made **both** with the plain fixed rules and with the hardened ones. The confirmatory one for these two is **against the hardened rules,** the author's decision, because `gradf_ablation` showed that their gain over the plain rules comes from the layers. The other is descriptive. FedStrategist and AdaAggRL do not pass through the layers and are compared with the plain rules.

**Exploratory:** per α and per attack, win counts, capture counts, Layer-4 rejection rate, honest rejection rate at Layer 1, GRADF's votes, FLTrust's swing.

## 6. Values from the pilots

| quantity | value | status |
|---|---|---|
| τ_CNN | 7.86 | from E0b/parity; block C's maximum honest norm is 1.66 |
| σ_E0^E1 | 2.72 (FLTrust, factor 1/3, rounds 106–110, seeds 0–2); 0.5σ = 1.36 | computed from the pilot; **wider than the primary margin** |
| σ_E0^E3 | 1.76 (FLTrust at root 100 − FedAvg, factor 1/3, rounds 106–110, seeds 0–2); 0.5σ = 0.88 | `factor_attack_check/RESULTS.md`; narrower than the primary margin, and the primary margin stays binding |

σ_E0 is redefined on the primary metric at the adopted factor. The plan read it at the final round with factor 1, where it was 1.69 and 2.93. It rests on 2 cells, 3 seeds, 2 degrees of freedom.

## 7. Framing fixed in advance

- **Layer 2 is not presented as a defense.** On MNIST its gain comes from the noise and fits a smaller step: a smaller step with no noise reproduces it (`gradf_layer2/step/RESULTS.md`). The cause on CIFAR-10 is not tested here.
- The hardened arms exist to make the comparison like for like, with no defense claim.
- `low_mag_backdoor` is reported as null on MNIST. Its status on the CNN was not verified. It stays in the full grid and out of E3.
- **Hyperparameter tuning is reported as such:** the factor was chosen after seeing FLTrust under attack on seeds 0–2 (`lr_tuning/ADDENDUM.md`, Sec. 6).
- MNIST uses the trailing mean of rounds 11–15, after its final-round results were known (`mnist_trailing/ADDENDUM.md`).

## 8. Guard and checks before the first run

**Runner guard.** Every job on seeds 42–51 records and checks `server_steps`, `global_lr`, root, R, metric window and code hash. A job aborts if a value is not the one this document assigns to its block. Non-main values are accepted only with the block's flag.

**Before the first confirmatory run:**
1. `global_lr = 1` reproduces earlier runs bit for bit, on the logistic model and the CNN, per learner: FedAvg, FLTrust, GRADF path (where deterministic), FedStrategist, AdaAggRL.
2. `global_lr = 1/3` reproduces `cifar_step_pilot` for FLTrust over a few rounds.
3. `global_lr = 1/3` gives 1/3 of the step in each learner, including AdaAggRL.
4. Pretraining receives the factor.
5. A job with a wrong setting is refused.

**During the campaign:**
- The trimmed mean's invariance outside `fltrust_aligned` is the first output. **If it fails, E1 stops** and it is treated as a possible leak (plan, Sec. 5).
- Monitoring reads counts, errors and process state only. **No accuracy of seeds 42–51 is read before each block is complete,** and the analysis runs once on complete blocks.
- A failed job is reported and is not rerun without the author's request. This includes a job whose connection to its VM is lost: it is logged as `INFRA dropped` and waits for the author. A job on a machine that does not answer is not started.

## 9. Differences from the base plan

| item | plan | here | reason |
|---|---|---|---|
| Global rate | fixed at 1, implicit | 1/3, with 1 as sensitivity | FLTrust unstable; tuned by the paper's criterion |
| FLTrust's server update | by epochs | step parity, epochs as sensitivity | specification of the method |
| Metric | final round | mean of rounds 106–110 | oscillation of about 12 p.p. per round |
| E3 grid | full if it fits | level 1 by decision, above budget | author |
| Budget | 236 h | ~277 h | author accepted |
| Fixed rules in E3 | plain only | plain and hardened | like-for-like (`gradf_ablation`) |
| σ_E0 | final round, factor 1 | primary metric, factor 1/3 | consistency |
| Root 50 | optional | not run | cost |
| Machines | this PC | this PC and 2 VMs, free assignment | throughput; bit-for-bit check |

## 10. Limits stated in advance

- **R = 110 is short at factor 1/3:** accuracy still rises 0.7 to 2 p.p. between rounds 71–90 and 91–110 in the sweep. Conclusions hold at this horizon.
- **No common factor is neutral between FLTrust and the trimmed mean.** This is why the sensitivity exists, and it covers E1 only. **E3 is not run at factor 1,** so its conclusions hold at factor 1/3 only.
- **The adopted factor favors FLTrust under attack on the 2 pilot cells:** it gains 11 to 21 p.p. while FedAvg loses 6.76 and the trimmed mean 4.77 (`factor_attack_check/RESULTS.md`). FedAvg crosses the 5.0 p.p. threshold. E1's level of V is read with this in mind.
- FLTrust remains unstable at factor 1/3 (11.82 p.p. per round in training error with no attack on CIFAR-10).
- The factor was tuned on one α with 3 seeds, at root 100, and the tuning's margin failed.
- The other 5 rules and the adaptive systems were not tuned, and they take the common factor.
- Non-IID partition fixed per α: seeds vary the training, not the partition (`p1_audit/AUDIT.md`).

## 11. After the campaign

E4a (MNIST), the P-2 gate and E4b are outside this document and this budget.
