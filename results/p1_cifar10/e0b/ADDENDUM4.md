# Addendum 4 — How the server-step diagnostic is read

**Date:** 2026-10-30 15:23 -0300, while the diagnostic of `ADDENDUM3.md` is running (43/508 jobs done) and before any of its outputs has been read. Addendum 3 and its code do not change; this adds to its "How it is read" section.

## 1. What the candidate correction is

The candidate (server steps = median over clients of ceil(n_i / 32) × epochs) is a translation of the specification, not the specification itself. FLTrust uses R_l = 1 iteration for clients and server alike; here the clients train by epochs. The mapping **preserves the specification's step parity between server and clients**. It is not "faithful to the paper" in its values (R_l, batch, learning rate), and the text will say so in these terms.

## 2. Equal steps bring a new confound at the small root

The number of server SGD steps and the number of passes over the root cannot be equal across root sizes at the same time. At equal steps the small root is passed over many more times per round (MNIST, batch 32: 470 steps are ≈ 150 passes over 100 examples and 5 passes over 3,000; the median mapping, 752–918 steps, is ≈ 240–290 passes over 100 examples). g₀ at the small root may then overfit those 100 examples, and its norm may grow beyond what the root's content alone would give (smoke test, seed 3, 2 rounds: ‖g₀‖ ≈ 2.1 at root 100 with 470 steps, against 0.25 at root 100 native; the root-3,000 figures are not yet measured at the same seed and rounds).

## 3. The reading, fixed before the results

On MNIST, per cell, mean over seeds 0–2, from the 2×2 {root 100, 3,000} × {20, 470 steps} (20 and 470 are the two roots' native step counts):

- **Step effect** (main effect of the step budget at fixed root): mean over the two roots of FLTrust(r, 470) − FLTrust(r, 20).
- **Residual root effect** (main effect of the root at fixed steps): mean over the two step levels of FLTrust(3,000, N) − FLTrust(100, N). It mixes the data's diversity with g₀ overfitting at the small root; it is **not a "pure content" effect**.
- **Interaction:** D_root@470 − D_root@20.

These are reported per cell, as means over the 21 cells and over the 18 without `fltrust_aligned`, and by α, next to the native contrast D_native (Addendum 3).

**Interpretation:**

- If the step effect dominates and the residual root effect is small, the RQ1 finding of the FGCS version (FLTrust's dominance depends on the root size) was mostly a step-budget effect, and the narrative changes accordingly.
- If both are relevant, RQ1 stands, with the caveat that the root dependence is partly mediated by the server's step budget, which epoch-based implementations couple to the root size.

"Dominate", "small" and "relevant" are judged against the primary E1 margin (1.0 p.p.) and the per-cell contrasts. This is descriptive (3 seeds, no test).

## Code hash (sha256)

`scripts/p1/analyze_g0_probe.py` was written after the launch and before any output was read; it now also computes the three 2×2 quantities above:

```
3ae0144359d58427d5f471a6a550f62aef09865b8e7c07606c88f7022a1ee5c6  scripts/p1/analyze_g0_probe.py
```
