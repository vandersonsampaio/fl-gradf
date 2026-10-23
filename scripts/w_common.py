"""
Shared pieces of the W2/W3 framework experiments (results/w2_backdoor_asr/,
results/w3_adaptativo_limiar/). Changes nothing in `src/`.

  WeightTap     records the final aggregation weights and the action of every
                `compute_weights_and_penalty` call. It works by replacing the module
                attribute BEFORE `build_learner_cls()` runs, so the ablation learner's
                closure imports the wrapper. The wrapper returns exactly what the
                original returns and consumes no RNG (checked by the regression
                smoke in each experiment's `verificar`).
  classify      the paper's verdict convention for a paired difference vector:
                superiority/inferiority by a two-sided t test (CI95), equivalence by
                TOST (CI90 inside ±margin), Holm over a family applied to each test type.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats


class WeightTap:
    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.action: Optional[np.ndarray] = None

    def clear(self):
        self.weights, self.action = None, None

    def normalized(self) -> Optional[np.ndarray]:
        """Weights as the ablation learner applies them (uniform when degenerate)."""
        if self.weights is None:
            return None
        tw = float(self.weights.sum())
        n = len(self.weights)
        return np.ones(n) / n if tw < 1e-8 else self.weights / tw


TAP = WeightTap()
_installed = False


def install_weight_tap() -> WeightTap:
    global _installed
    import src.defense.adaaggrl_agent as A
    if not _installed:
        orig = A.compute_weights_and_penalty

        def tapped(state_matrix, action, prev_h, lam=2.0):
            out = orig(state_matrix, action, prev_h, lam=lam)
            TAP.weights = np.array(out[0], dtype=np.float64)
            TAP.action = np.array(action, dtype=np.float64)
            return out
        A.compute_weights_and_penalty = tapped
        _installed = True
    return TAP


def pooled_test(participants):
    return (np.concatenate([p.X_test for p in participants]),
            np.concatenate([p.y_test for p in participants]))


def holm_adjust(pvals: List[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_stats(d: np.ndarray, margin: float) -> Dict:
    d = np.asarray(d, dtype=float)
    n, m, s = len(d), float(d.mean()), float(d.std(ddof=1))
    se = s / np.sqrt(n)
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    p_lo = stats.t.sf((m + margin) / se, n - 1)   # H0: mean <= -margin
    p_hi = stats.t.cdf((m - margin) / se, n - 1)  # H0: mean >= +margin
    try:
        p_w = float(stats.wilcoxon(d).pvalue)
    except ValueError:  # all differences zero
        p_w = 1.0
    return {"n": n, "mean": m, "sd": s, "ic95_lo": m - t95 * se, "ic95_hi": m + t95 * se,
            "ic90_lo": m - t90 * se, "ic90_hi": m + t90 * se,
            "p_diff": float(stats.ttest_1samp(d, 0).pvalue), "p_tost": float(max(p_lo, p_hi)),
            "p_wilcoxon": p_w, "d_z": m / s if s > 0 else np.nan, "margin": margin}


def classify_family(rows: List[Dict], alpha: float = 0.05) -> List[Dict]:
    """`rows` from `paired_stats`. Adds Holm-adjusted p for each test type and the verdict
    (superiority/inferiority first, then equivalence, else inconclusive), with the sign
    convention 'positive mean = superiority' left to the caller's definition of d."""
    if not rows:
        return rows
    adj_diff = holm_adjust([r["p_diff"] for r in rows])
    adj_tost = holm_adjust([r["p_tost"] for r in rows])
    for r, pd_, pt in zip(rows, adj_diff, adj_tost):
        r["p_diff_holm"], r["p_tost_holm"] = float(pd_), float(pt)
        if pd_ < alpha and r["mean"] > 0:
            r["verdict"] = "superiority"
        elif pd_ < alpha and r["mean"] < 0:
            r["verdict"] = "inferiority"
        elif pt < alpha:
            r["verdict"] = "equivalence"
        else:
            r["verdict"] = "inconclusive"
    return rows
