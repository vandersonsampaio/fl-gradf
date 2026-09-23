"""
Byzantine-robust aggregation methods, registered into `src.fl.federated_learner`
via `register_strategy()`. Together with FedAvgStrategy/FLTrustStrategy (already
defined in federated_learner.py), this completes the 7 strategies in GRADF's
action space (see src/defense/rl_selector.py).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.fl.federated_learner import AggregationStrategy, register_strategy


class MedianStrategy(AggregationStrategy):
    """Coordinate-wise median — robust to a minority of extreme outliers."""

    def aggregate(self, updates: List[np.ndarray], **kwargs) -> Tuple[np.ndarray, None]:
        stacked = np.stack(updates, axis=0)
        return np.median(stacked, axis=0), None


class TrimmedMeanStrategy(AggregationStrategy):
    """Coordinate-wise trimmed mean: removes `trim_ratio` of the most extreme
    values (top and bottom) in each coordinate before averaging."""

    def __init__(self, trim_ratio: float = 0.2) -> None:
        self.trim_ratio = trim_ratio

    def aggregate(self, updates: List[np.ndarray], **kwargs) -> Tuple[np.ndarray, None]:
        stacked = np.sort(np.stack(updates, axis=0), axis=0)  # (n_clients, dim)
        n = stacked.shape[0]
        k = int(np.floor(n * self.trim_ratio))
        if 2 * k >= n:
            k = max(0, (n - 1) // 2)
        trimmed = stacked[k: n - k] if k > 0 else stacked
        return trimmed.mean(axis=0), None


class FedProxStrategy(AggregationStrategy):
    """Server-side simplification of FedProx (Li et al., MLSys 2020).

    The original FedProx adds a proximal term (mu/2)*||theta - theta_prev||^2
    to the LOCAL TRAINING objective — not observable from the already-computed
    deltas that reach this aggregation interface. As an approximation, this
    implementation applies a weighted average (equivalent to FedAvg) followed
    by damping proportional to `mu`, giving RLDefenseSelector an action with
    distinct (more conservative) behavior from the others.
    """

    def __init__(self, mu: float = 0.1) -> None:
        self.mu = mu

    def aggregate(
        self,
        updates: List[np.ndarray],
        sample_sizes: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, None]:
        if sample_sizes is None:
            w = np.ones(len(updates)) / len(updates)
        else:
            total = float(sum(sample_sizes))
            w = np.array(sample_sizes, dtype=float) / total
        agg = sum(wi * u for wi, u in zip(w, updates))
        return agg / (1.0 + self.mu), None  # type: ignore[return-value]


class KrumStrategy(AggregationStrategy):
    """Krum (Blanchard et al., NeurIPS 2017): selects the single update whose
    sum of Euclidean distances to the `n - f - 2` closest updates is minimal
    (f = assumed number of Byzantine clients)."""

    def __init__(self, n_byzantine: Optional[int] = None) -> None:
        self.n_byzantine = n_byzantine

    def aggregate(self, updates: List[np.ndarray], **kwargs) -> Tuple[np.ndarray, Dict]:
        n = len(updates)
        f = self.n_byzantine if self.n_byzantine is not None else max(0, (n - 3) // 2)
        n_closest = max(1, n - f - 2)

        stacked = np.stack(updates, axis=0)
        dists = np.linalg.norm(stacked[:, None, :] - stacked[None, :, :], axis=-1)

        scores = []
        for i in range(n):
            sorted_d = np.sort(dists[i])
            scores.append(np.sum(sorted_d[1:1 + n_closest]))  # excludes distance to self

        selected = int(np.argmin(scores))
        return updates[selected].copy(), {"selected_index": selected}


class ClusteringStrategy(AggregationStrategy):
    """Groups updates into 2 clusters (K-means) and aggregates only the
    majority cluster, assuming most participants are honest."""

    def aggregate(self, updates: List[np.ndarray], **kwargs) -> Tuple[np.ndarray, Dict]:
        n = len(updates)
        if n < 3:
            return np.mean(updates, axis=0), {"cluster_sizes": [n]}

        from sklearn.cluster import KMeans

        stacked = np.stack(updates, axis=0)
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(stacked)

        counts = np.bincount(labels)
        majority_label = int(np.argmax(counts))
        selected = [u for u, lbl in zip(updates, labels) if lbl == majority_label]

        return (
            np.mean(selected, axis=0),
            {"cluster_sizes": counts.tolist(), "majority_label": majority_label},
        )


register_strategy("median", MedianStrategy)
register_strategy("trimmed_mean", TrimmedMeanStrategy)
register_strategy("fedprox", FedProxStrategy)
register_strategy("krum", KrumStrategy)
register_strategy("clustering", ClusteringStrategy)
