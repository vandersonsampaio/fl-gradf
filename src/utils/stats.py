"""
Statistical infrastructure shared by the multi-seed experiments
(src/experiments/exp1, exp2, exp4, exp7) — mean/std/CI95 across seeds and
paired significance, needed for a results table that holds up under
conference review.
"""

from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def run_over_seeds(fn: Callable[..., pd.DataFrame], seeds: Sequence[int], **kwargs) -> pd.DataFrame:
    """Calls `fn(seed=s, **kwargs)` for each seed and concatenates the results.

    `fn` must return a `pd.DataFrame` (one or more rows) — each row is
    tagged with a `seed` column before concatenation. Used to give the
    experiments statistical rigor: they were already parameterized by `seed`
    but historically ran with a single fixed seed.
    """
    frames = []
    for seed in seeds:
        df = fn(seed=seed, **kwargs)
        df = df.copy()
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """Aggregates `value_col` by `group_cols` across seeds: mean, standard
    deviation, and 95% confidence interval (t distribution, appropriate for
    the small N of seeds typical in FL experiments)."""

    def _agg(group: pd.Series) -> pd.Series:
        values = group.to_numpy(dtype=float)
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        if n > 1 and std > 0:
            half_width = float(scipy_stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
        else:
            half_width = 0.0
        return pd.Series({
            "mean": mean,
            "std": std,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
            "n": n,
        })

    return df.groupby(group_cols)[value_col].apply(_agg).unstack().reset_index()


def cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d effect size for paired samples: mean of the difference
    divided by the standard deviation of the difference (d_z — the usual
    definition when pairing, here by seed, removes the variance common to
    both arms). Used alongside the paired t-test for Gate 1 (GRADF vs.
    FLTrust/TARS): the p-value says whether the difference is statistically
    distinguishable from zero, d says whether it's large enough to matter."""
    diff = np.asarray(diff, dtype=float)
    std = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    if std == 0.0:
        return 0.0 if np.allclose(diff, 0.0) else float("inf") * np.sign(diff.mean())
    return float(np.mean(diff) / std)


def paired_significance(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    baseline_label: str,
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Paired t-test (by seed) of each `group_col` value against
    `baseline_label`, on `value_col`. Returns one row per group (except the
    baseline itself) with the mean difference, the p-value, and the effect
    size (paired Cohen's d, `cohens_d_paired`).

    Pairing by seed is what makes the test valid here: the same seed
    controls the same initialization/partitioning in both compared arms, so
    the between-seed variance is removed from the comparison.
    """
    baseline = df[df[group_col] == baseline_label].set_index(seed_col)[value_col]

    rows = []
    for group_value, group_df in df[df[group_col] != baseline_label].groupby(group_col):
        paired = group_df.set_index(seed_col)[value_col]
        common_seeds = baseline.index.intersection(paired.index)
        if len(common_seeds) < 2:
            rows.append({
                group_col: group_value, "mean_diff": float("nan"),
                "p_value": float("nan"), "cohens_d": float("nan"), "n": len(common_seeds),
            })
            continue

        a = paired.loc[common_seeds].to_numpy(dtype=float)
        b = baseline.loc[common_seeds].to_numpy(dtype=float)
        diff = a - b
        if np.allclose(diff, diff[0]):
            # scipy raises an error/NaN when the difference is constant (zero variance).
            p_value = 0.0 if diff[0] != 0 else 1.0
        else:
            _, p_value = scipy_stats.ttest_rel(a, b)
        rows.append({
            group_col: group_value,
            "mean_diff": float(np.mean(diff)),
            "p_value": float(p_value),
            "cohens_d": cohens_d_paired(diff),
            "n": len(common_seeds),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    demo = pd.DataFrame({
        "system": ["FedAvg", "FedAvg", "GRADF", "GRADF"],
        "seed": [42, 43, 42, 43],
        "accuracy": [0.70, 0.72, 0.85, 0.87],
    })
    print(summarize(demo, ["system"], "accuracy"))
    print(paired_significance(demo, "system", "accuracy", baseline_label="FedAvg"))
