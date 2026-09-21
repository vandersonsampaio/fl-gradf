"""
`ModalityRecorder` groups the 4 detectors from `src.detection` and keeps
per-hospital history across rounds, producing the 4-score vector used by the
combiner/classifier.

Lives in `src.detection` (not in `src.utils.data_loader`) to avoid a circular
import: `src.fl.gradf_learner` needs `ModalityRecorder`, and
`src.utils.data_loader` needs `src.fl` — putting `ModalityRecorder` in
`src.utils.data_loader` would make `src.fl` depend on `src.utils.data_loader`
and vice versa. `src.detection` depends on neither `src.fl` nor `src.utils`,
so it is the safe common base.
"""

from typing import List

import numpy as np

from src.classification.attack_simulator import AttackSimulator
from src.detection.accuracy_detector import AccuracyDetector
from src.detection.clustering_detector import ClusteringDetector
from src.detection.gradient_analyzer import GradientAnalyzer
from src.detection.temporal_detector import TemporalDetector

# Possible labels: 'none' (honest update) + every simulated attack type.
ATTACK_LABELS: List[str] = ["none"] + list(AttackSimulator.ALL_ATTACK_TYPES)


class ModalityRecorder:
    def __init__(self, window_size: int = 5):
        self.gradient = GradientAnalyzer(window_size=window_size)
        self.accuracy = AccuracyDetector(window_size=window_size)
        self.temporal = TemporalDetector(window_size=window_size)
        self.clustering = ClusteringDetector()

    def score(
        self,
        hospital_id: str,
        update: np.ndarray,
        peer_updates: List[np.ndarray],
        old_accuracy: float,
        new_accuracy: float,
    ) -> np.ndarray:
        g = self.gradient.analyze(hospital_id, update)
        a = self.accuracy.analyze(hospital_id, old_accuracy, new_accuracy)
        t = self.temporal.analyze(hospital_id, update)
        c = self.clustering.analyze(hospital_id, update, peer_updates)
        return np.array(
            [g["anomaly_score"], a["anomaly_score"], c["anomaly_score"], t["anomaly_score"]],
            dtype=np.float32,
        )
