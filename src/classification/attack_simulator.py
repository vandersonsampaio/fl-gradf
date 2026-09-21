"""
Attack simulator for Federated Learning adversarial experiments.

Implements the 11 GRADF attack types plus the explicit attack variants
described in the TARS paper (Ahmed et al., 2025):
  - Label Flipping:  y ← (y + 1) mod K  (data-level)
  - Sign Flipping:   w = -Δ              (gradient-level)
  - Gaussian Attack: w = Δ + ε, ε ~ N(0, σ²I)  (gradient-level)
  - Pretense Attack: honest for T rounds, then attack (PretenseAttacker class)

Plus 4 "informed" attacks (Phase 1, Strategy B —
references/estrategia_B_recriar_headroom.md), designed to specifically stress
FLTrust/Median/Trimmed-Mean/Krum rather than generic Byzantine noise, in the
omniscient-attacker threat model of Fang et al. ("Local Model Poisoning
Attacks against Byzantine-robust Federated Learning", USENIX Security 2020):
  - fltrust_aligned : clones the server root-dataset direction to maximize
                      FLTrust's cosine-similarity trust score.
  - trim_attack     : per-coordinate push just beyond the honest range,
                      targeting Median/Trimmed-Mean.
  - krum_collusion  : colluding sybils converge on one direction so their
                      updates cluster tightly, targeting Krum's
                      nearest-neighbour selection.
  - low_mag_backdoor: small, persistent single-target shift scaled to the
                      update's own norm, instead of a loud multiplicative
                      spike.
These need extra context (`reference` = server root gradient this round,
`peer_updates` = other clients' honest updates this round) that a "blind"
attacker doesn't have — supplied by
`src.fl.attacked_learner.InformedAttackedFederatedLearner`. Called without
that context (e.g. from the plain AttackedFederatedLearner), each falls back
to a documented, weaker blind variant.

Two attack surfaces
-------------------
- poison_update(update, attack_type)  — modifies the parameter delta vector
- poison_data(X, y, attack_type)      — modifies training labels (data poisoning)

Usage
-----
    sim = AttackSimulator(seed=42)

    # Gradient-level attack
    poisoned = sim.poison_update(clean_delta, 'sign_flipping')

    # Data-level attack
    X_p, y_p = sim.poison_data(X_train, y_train, 'label_flipping', n_classes=10)

    # Simulate a full round: replace Byzantine fraction with poisoned updates
    updates, is_byz = sim.simulate_round(client_updates, byzantine_fraction=0.2,
                                         attack_type='gaussian_noise')

    # Pretense (TARS): honest for 5 rounds, then sign flipping
    pretender = PretenseAttacker(pretense_rounds=5, attack_type='sign_flipping')
    delta = pretender.get_update(clean_delta, round_num=7)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Attack type sets
# ---------------------------------------------------------------------------

# Attacks that operate on parameter update vectors (gradients)
_UPDATE_ATTACKS: frozenset = frozenset({
    'poisoning',          # Extreme magnitude scaling
    'byzantine',          # Negate + Gaussian noise
    'backdoor',           # Manipulate target-class weights
    'inference',          # Perturb to leak local distribution
    'sybil',              # Boosted sign-flip simulating multiple fake clients
    'model_poisoning',    # Pure sign flip (TARS: sign flipping variant)
    'sign_flipping',      # TARS: w = -Δ (explicit alias)
    'free_riding',        # Submit zero update
    'colluding',          # Coordinated sign-flip amplification
    'dp_attack',          # Gaussian noise disguised as DP (TARS: Gaussian attack)
    'gaussian_noise',     # TARS: w = Δ + ε, ε ~ N(0, σ²I) (explicit alias)
    'gradient_inversion', # Randomise update to obstruct defences
    'fltrust_aligned',    # Phase 1: clones server root direction (evades FLTrust trust score)
    'trim_attack',        # Phase 1: coordinate-wise push against Median/Trimmed-Mean
    'krum_collusion',     # Phase 1: colluding sybils cluster together (evades Krum)
    'low_mag_backdoor',   # Phase 1: small persistent single-target shift
})

# Attacks that operate on training data (before local training)
_DATA_ATTACKS: frozenset = frozenset({
    'label_flipping',     # TARS: c → (c + 1) mod K
})

ALL_ATTACKS: frozenset = _UPDATE_ATTACKS | _DATA_ATTACKS


class AttackSimulator:
    """
    Generates adversarial parameter updates and poisoned training data for FL
    security experiments.

    Parameters
    ----------
    sigma : float
        Base standard deviation for Gaussian noise attacks.
    seed : int | None
        Random seed for reproducibility.
    """

    # Full list including explicit TARS aliases
    ALL_ATTACK_TYPES: List[str] = sorted(ALL_ATTACKS)

    # Phase 1 (Strategy B): attacks that need `reference`/`peer_updates`
    # (omniscient attacker) to work as designed — see
    # InformedAttackedFederatedLearner. Exposed here so experiments know
    # which learner to use per attack type without hardcoding the list.
    INFORMED_ATTACK_TYPES: List[str] = [
        'fltrust_aligned', 'trim_attack', 'krum_collusion', 'low_mag_backdoor',
    ]

    def __init__(self, sigma: float = 1.0, seed: Optional[int] = None) -> None:
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def poison_update(
        self,
        update: np.ndarray,
        attack_type: str,
        intensity: float = 0.8,
        target_class: int = 0,
        reference: Optional[np.ndarray] = None,
        peer_updates: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Apply a gradient-level attack to a parameter update vector.

        Parameters
        ----------
        update : ndarray
            Clean parameter delta from local training (flat vector).
        attack_type : str
            One of _UPDATE_ATTACKS.
        intensity : float
            Strength parameter for magnitude-based attacks (0–1).
        target_class : int
            Class index for backdoor attacks.
        reference : ndarray | None
            Server root-dataset gradient this round — only used by
            'fltrust_aligned' (an omniscient-attacker input; see module
            docstring). Ignored by every other attack type.
        peer_updates : list[ndarray] | None
            Other clients' honest updates this round — only used by
            'trim_attack'/'krum_collusion' (omniscient-attacker input).
            Ignored by every other attack type.

        Returns
        -------
        ndarray of the same shape as update.
        """
        name = attack_type.lower()
        if name not in _UPDATE_ATTACKS:
            if name in _DATA_ATTACKS:
                raise ValueError(
                    f"'{name}' is a data-level attack. Use poison_data() instead."
                )
            raise ValueError(
                f"Unknown attack type: {name!r}. Valid gradient attacks: {sorted(_UPDATE_ATTACKS)}"
            )
        dispatch: Dict = {
            'poisoning':          self._poisoning,
            'byzantine':          self._byzantine,
            'backdoor':           self._backdoor,
            'inference':          self._inference,
            'sybil':              self._sybil,
            'model_poisoning':    self._sign_flipping,
            'sign_flipping':      self._sign_flipping,
            'free_riding':        self._free_riding,
            'colluding':          self._colluding,
            'dp_attack':          self._gaussian_noise,
            'gaussian_noise':     self._gaussian_noise,
            'gradient_inversion': self._gradient_inversion,
            'fltrust_aligned':    self._fltrust_aligned,
            'trim_attack':        self._trim_attack,
            'krum_collusion':     self._krum_collusion,
            'low_mag_backdoor':   self._low_mag_backdoor,
        }
        return dispatch[name](
            update, intensity=intensity, target_class=target_class,
            reference=reference, peer_updates=peer_updates,
        )

    def poison_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_type: str,
        n_classes: int = 10,
        poison_fraction: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a data-level poisoning attack to training samples.

        Parameters
        ----------
        X : ndarray (N, features)
        y : ndarray (N,) int labels
        attack_type : str
            Currently only 'label_flipping'.
        n_classes : int
            Total number of classes K (used for label cycling).
        poison_fraction : float
            Fraction of samples to corrupt (0–1).

        Returns
        -------
        X_poisoned, y_poisoned (copies; original arrays are not modified)
        """
        name = attack_type.lower()
        if name != 'label_flipping':
            if name in _UPDATE_ATTACKS:
                raise ValueError(
                    f"'{name}' is a gradient-level attack. Use poison_update() instead."
                )
            raise ValueError(f"Unknown data attack: {name!r}")
        return self._label_flipping(X, y, n_classes=n_classes, poison_fraction=poison_fraction)

    # ------------------------------------------------------------------
    # Gradient-level attack implementations
    # ------------------------------------------------------------------

    def _poisoning(self, update: np.ndarray, intensity: float = 0.8, **_) -> np.ndarray:
        """Extreme gradient scaling.

        Formula: ḡ = Δ · (10 + 40·intensity)
        """
        return update * (10.0 + 40.0 * float(intensity))

    def _byzantine(self, update: np.ndarray, intensity: float = 0.8, **_) -> np.ndarray:
        """Negate + Gaussian noise.

        Formula: ḡ = -Δ + ε,  ε ~ N(0, σ²·I)
        """
        noise = self._rng.normal(0.0, self.sigma * (0.1 + intensity), size=update.shape)
        return -update + noise

    def _sign_flipping(self, update: np.ndarray, **_) -> np.ndarray:
        """Pure sign reversal (TARS paper — Sign Flipping / Model Poisoning).

        Formula: ḡ = -Δ
        """
        return -update

    def _gaussian_noise(self, update: np.ndarray, intensity: float = 0.8, **_) -> np.ndarray:
        """Additive Gaussian noise (TARS paper — Gaussian Attack / DP Attack).

        Formula: ḡ = Δ + ε,  ε ~ N(0, σ²·I),  σ scaled by intensity
        """
        noise = self._rng.normal(0.0, self.sigma * (0.5 + intensity), size=update.shape)
        return update + noise

    def _backdoor(
        self, update: np.ndarray, target_class: int = 0, intensity: float = 0.8, **_
    ) -> np.ndarray:
        """Manipulate the weight slice associated with a target class.

        Negates and amplifies the target_class element of the flat update
        vector, causing the global model to misclassify that class.
        """
        poisoned = update.copy()
        if 0 <= target_class < len(poisoned):
            poisoned[target_class] *= -(1.0 + intensity * 10.0)
        return poisoned

    def _inference(self, update: np.ndarray, intensity: float = 0.8, **_) -> np.ndarray:
        """Perturb the update to leak membership information.

        Adds structured noise proportional to the update norm so the
        submitted gradient reveals distributional information about local data.
        """
        noise = self._rng.standard_normal(update.shape)
        noise *= np.linalg.norm(update) * intensity * 0.3
        return update * 0.5 + noise

    def _sybil(self, update: np.ndarray, intensity: float = 0.8, **_) -> np.ndarray:
        """Boosted sign-flip simulating N synthetic fake identities.

        A single malicious client submits an amplified inverted update,
        mimicking the effect of multiple colluding Sybil nodes.
        """
        n_virtual = 1.0 + intensity * 9.0
        return -update * n_virtual

    def _free_riding(self, update: np.ndarray, **_) -> np.ndarray:
        """Submit a zero update — receive global model without contributing.

        Formula: ḡ = 0
        """
        return np.zeros_like(update)

    def _colluding(self, update: np.ndarray, intensity: float = 0.8, **_) -> np.ndarray:
        """Coordinated sign-flip with intensity-scaled amplification.

        Models a group of attackers sharing a common poisoning direction.
        Single-client view: sign flip + coordinated boost.
        """
        return -update * (1.0 + intensity * 4.0)

    def _gradient_inversion(self, update: np.ndarray, intensity: float = 0.8, **_) -> np.ndarray:
        """Randomise the update to obstruct gradient-based defences.

        The attacker submits noise at the server gradient norm to blend in
        while reconstructing training data locally using the true gradient.
        """
        noise = self._rng.standard_normal(update.shape)
        noise /= np.linalg.norm(noise) + 1e-10
        return noise * np.linalg.norm(update) * (0.5 + intensity)

    # ------------------------------------------------------------------
    # "Informed" attacks (Phase 1, Strategy B) — need reference/peer_updates
    # ------------------------------------------------------------------

    def _fltrust_aligned(
        self, update: np.ndarray, reference: Optional[np.ndarray] = None, **_
    ) -> np.ndarray:
        """Evade FLTrust's cosine-similarity trust filter by cloning the
        server's own root-dataset direction (Fang et al., 2020-style
        omniscient attack).

        FLTrust weights each update by ReLU(cos(update, reference)) and
        clips its norm to the reference's norm before averaging — so a
        client that submits `reference` itself gets cos=1 (maximal trust
        weight) and its post-clip contribution is just `reference` again
        (amplifying it is moot, the clip normalizes magnitude away). The
        damage isn't a loud spike: under severe non-IID heterogeneity the
        server's root sample (a small slice of one client's data) is
        unrepresentative of the true population, so a client that always
        looks "maximally trusted" while contributing exactly the root
        direction can starve honest, imperfectly-aligned clients of
        influence and bias the aggregate toward the narrow root sample
        instead of the real heterogeneous data.

        Needs `reference` (the server/root gradient this round — see
        InformedAttackedFederatedLearner). Falls back to a plain sign-flip
        if unavailable, so this type is still safe to call blind.
        """
        if reference is None or np.linalg.norm(reference) < 1e-10:
            return -update
        return reference.copy()

    def _trim_attack(
        self,
        update: np.ndarray,
        peer_updates: Optional[List[np.ndarray]] = None,
        intensity: float = 0.8,
        **_,
    ) -> np.ndarray:
        """Coordinate-wise attack against Median/Trimmed-Mean (Fang et al.,
        2020-style omniscient attack): per coordinate, push the malicious
        value just beyond the honest range, in the direction opposite the
        honest mean, so the attack drags the robust per-coordinate estimator
        as far as a single/small colluding group can.

        `intensity` scales how far beyond the honest [min, max] spread the
        malicious value goes — our own documented instantiation of "just
        beyond the trim boundary" (the paper's closed-form optimisation
        against a specific estimator is not implemented here).

        Needs `peer_updates` (other clients' honest updates this round — an
        omniscient-attacker input; see InformedAttackedFederatedLearner).
        Falls back to the plain magnitude-scaling 'poisoning' attack if
        unavailable.
        """
        if not peer_updates:
            return update * (10.0 + 40.0 * float(intensity))
        honest = np.stack(peer_updates, axis=0)
        mean_j = honest.mean(axis=0)
        spread = honest.max(axis=0) - honest.min(axis=0) + 1e-8
        direction = np.sign(mean_j)
        direction[direction == 0] = 1.0
        return mean_j - direction * spread * (0.5 + intensity)

    def _krum_collusion(
        self,
        update: np.ndarray,
        peer_updates: Optional[List[np.ndarray]] = None,
        intensity: float = 0.8,
        **_,
    ) -> np.ndarray:
        """Colluding-sybil attack against Krum (and clustering-style
        defences): every byzantine client converges on the SAME malicious
        direction — the negated centroid of the honest peers' updates this
        round, so no explicit inter-client communication channel is needed
        in simulation — plus small independent jitter, so their submitted
        updates cluster tightly together. Krum picks the single update
        whose distance-sum to its `n-f-2` nearest neighbours is smallest;
        a tight malicious cluster can out-score a genuinely diverse honest
        population under this metric.

        Needs `peer_updates` (an omniscient-attacker input; see
        InformedAttackedFederatedLearner). Falls back to a boosted
        sign-flip if unavailable.
        """
        if not peer_updates:
            return -update * (1.0 + intensity * 4.0)
        centroid = np.mean(peer_updates, axis=0)
        direction = -centroid
        jitter = self._rng.normal(
            0.0, 0.02 * (np.linalg.norm(centroid) + 1e-8), size=update.shape
        )
        return direction * (0.5 + intensity) + jitter

    def _low_mag_backdoor(
        self,
        update: np.ndarray,
        target_class: int = 0,
        intensity: float = 0.15,
        **_,
    ) -> np.ndarray:
        """Low-magnitude, persistent single-target backdoor. Unlike
        `_backdoor`'s large multiplicative spike (`intensity * 10`+), this
        nudges the target-class slice by an additive shift scaled to the
        update's OWN per-coordinate typical magnitude, so a single round's
        poisoned update sits close to the honest norm — evading
        magnitude-based outlier detection while remaining persistently
        harmful to the target class across rounds. Recommended `intensity`
        is smaller than other attacks' default (≤0.2ish) precisely because
        the point is to stay quiet, not loud.
        """
        poisoned = update.copy()
        if 0 <= target_class < len(poisoned):
            typical = float(np.linalg.norm(update)) / np.sqrt(len(update))
            poisoned[target_class] -= intensity * typical * 5.0
        return poisoned

    # ------------------------------------------------------------------
    # Data-level attack implementation
    # ------------------------------------------------------------------

    def _label_flipping(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int = 10,
        poison_fraction: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cyclic label shift: c → (c + 1) mod K  (TARS paper).

        Parameters
        ----------
        poison_fraction : float
            Fraction of samples whose labels are flipped. 1.0 flips all.
        """
        y_out = y.copy()
        n_poison = max(1, int(len(y) * poison_fraction))
        idx = self._rng.choice(len(y), n_poison, replace=False)
        y_out[idx] = (y[idx] + 1) % n_classes
        return X, y_out


# ---------------------------------------------------------------------------
# Pretense attack wrapper (TARS paper)
# ---------------------------------------------------------------------------

class PretenseAttacker:
    """
    TARS pretense attack: mimics an honest client for the first T rounds,
    then switches to a specified gradient-level attack.

    This makes static anomaly detectors ineffective because the attacker
    accumulates trust during the pretense phase before launching attacks.

    Parameters
    ----------
    pretense_rounds : int
        Number of honest rounds before poisoning begins.
    attack_type : str
        Gradient-level attack to apply after the pretense phase.
    simulator : AttackSimulator | None
        Shared simulator instance (creates a new one if None).

    Example
    -------
        pretender = PretenseAttacker(pretense_rounds=5, attack_type='sign_flipping')
        for t in range(30):
            delta = pretender.get_update(clean_delta, round_num=t)
    """

    def __init__(
        self,
        pretense_rounds: int = 5,
        attack_type: str = 'sign_flipping',
        simulator: Optional[AttackSimulator] = None,
        **attack_kwargs,
    ) -> None:
        if attack_type.lower() in _DATA_ATTACKS:
            raise ValueError(
                f"PretenseAttacker requires a gradient-level attack, got '{attack_type}'. "
                "Data attacks (label_flipping) are applied before local training."
            )
        self.pretense_rounds = pretense_rounds
        self.attack_type = attack_type.lower()
        self.simulator = simulator if simulator is not None else AttackSimulator()
        self.attack_kwargs = attack_kwargs

    def get_update(self, clean_update: np.ndarray, round_num: int) -> np.ndarray:
        """Return the honest update during pretense; poisoned update after.

        Parameters
        ----------
        clean_update : ndarray
            The legitimate parameter delta from local training.
        round_num : int
            Current FL round index (0-based).
        """
        if round_num < self.pretense_rounds:
            return clean_update.copy()
        return self.simulator.poison_update(
            clean_update, self.attack_type, **self.attack_kwargs
        )
