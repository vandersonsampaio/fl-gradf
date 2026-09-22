"""
FederatedLearner extended with per-round attack injection.

Factored out of the `AttackedFederatedLearner` pattern (originally prototyped
in notebooks/08_fl_attack_simulation.ipynb) so it can be reused by other
modules: src/utils/data_loader.py, src/defense/rl_selector.py, and
src/fl/gradf_learner.py.

Byzantine clients (by index in byzantine_ids) have their parameter updates
poisoned after local training. For label_flipping, training labels are
corrupted before local training (a data-level attack).

pretense_rounds > 0: clients behave honestly in rounds 1..pretense_rounds and
then launch attacks from round pretense_rounds+1 onward (TARS pretense attack).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.classification.attack_simulator import AttackSimulator
from src.fl.federated_learner import FederatedLearner, ParticipantData, RoundResult


class AttackedFederatedLearner(FederatedLearner):
    def __init__(
        self,
        *args,
        attack_type: str = "none",
        byzantine_ids: Optional[List[int]] = None,
        pretense_rounds: int = 0,
        n_classes_data: int = 10,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attack_type = attack_type
        self.byzantine_ids = set(byzantine_ids or [])
        self.pretense_rounds = pretense_rounds
        self.n_classes_data = n_classes_data
        self._sim = AttackSimulator(seed=seed)

    def _is_active(self, round_num: int) -> bool:
        return self.attack_type not in ("none", "") and round_num > self.pretense_rounds

    def _compute_param_updates(
        self, participants: List[ParticipantData], active: bool
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """Locally train each participant and inject the attack into active Byzantine ones.

        Returns (param_updates, is_byzantine_list) in the same order as `participants`.
        """
        param_updates: List[np.ndarray] = []
        is_byz_list: List[bool] = []

        for i, p in enumerate(participants):
            is_byz = (i in self.byzantine_ids) and active
            model = self._make_model(p.n_features)

            if is_byz and self.attack_type == "label_flipping":
                _, y_p = self._sim.poison_data(
                    p.X_train, p.y_train, "label_flipping",
                    n_classes=self.n_classes_data,
                )
                delta = model.fit(p.X_train, y_p)
            else:
                delta = model.fit(p.X_train, p.y_train)
                if is_byz:
                    delta = self._sim.poison_update(delta, self.attack_type)

            param_updates.append(delta)
            is_byz_list.append(is_byz)

        return param_updates, is_byz_list

    def _run_round(
        self,
        round_num: int,
        participants: List[ParticipantData],
        root_data: Optional[Dict],
    ) -> RoundResult:
        active = self._is_active(round_num)
        param_updates, _is_byz_list = self._compute_param_updates(participants, active)

        agg_kwargs: Dict = {}
        if self.aggregation == "fltrust":
            srv = self._make_model(participants[0].n_features)
            srv_delta = srv.fit(root_data["X"], root_data["y"])  # type: ignore[index]
            agg_kwargs["server_update"] = srv_delta
        else:
            agg_kwargs["sample_sizes"] = [p.n_train for p in participants]

        agg_delta, metadata = self._strategy.aggregate(param_updates, **agg_kwargs)
        self._global_params = self._global_params + agg_delta  # type: ignore[operator]

        per_acc = {
            p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test)
            for p in participants
        }
        global_acc = float(np.mean(list(per_acc.values())))

        trust_scores: Optional[Dict[str, float]] = None
        n_accepted: Optional[int] = None
        if metadata and "trust_scores" in metadata:
            raw = metadata["trust_scores"]
            trust_scores = {p.id: ts for p, ts in zip(participants, raw)}
            n_accepted = sum(1 for ts in raw if ts > 0)

        return RoundResult(round_num, global_acc, per_acc, trust_scores, n_accepted)


class InformedAttackedFederatedLearner(AttackedFederatedLearner):
    """AttackedFederatedLearner whose Byzantine clients see, before forging
    their update, (a) the server's reference direction (`server_update`, the
    same one FLTrust uses) and (b) the HONEST updates from the other clients
    this round — an "omniscient" attacker (the threat model from Fang et al.,
    2020, "Local Model Poisoning Attacks against Byzantine-robust Federated
    Learning") used specifically by the informed attack types in
    `AttackSimulator.INFORMED_ATTACK_TYPES` (`fltrust_aligned`,
    `trim_attack`, `krum_collusion`, `low_mag_backdoor`) — see
    `references/estrategia_B_recriar_headroom.md`.

    Deliberately stronger (worst-case) than `AttackedFederatedLearner` (a
    blind attacker that doesn't see other clients' updates) — used only for
    this stress test, and does NOT replace the blind attacker in the already
    published experiments (exp1/exp2/exp4/exp7), which keep their original
    behavior.

    `server_update` is ALWAYS computed, even when `self.aggregation !=
    'fltrust'` — same reason documented in `GRADFFederatedLearner`/
    `TARSLearner` (see CLAUDE.md): `fltrust_aligned` needs it regardless of
    the aggregation rule chosen for the round.
    """

    def _run_round(
        self,
        round_num: int,
        participants: List[ParticipantData],
        root_data: Optional[Dict],
    ) -> RoundResult:
        active = self._is_active(round_num)

        root = root_data or self._carve_root(participants[0])
        server_delta = self._make_model(participants[0].n_features).fit(root["X"], root["y"])

        honest_updates: List[np.ndarray] = []
        for p in participants:
            model = self._make_model(p.n_features)
            honest_updates.append(model.fit(p.X_train, p.y_train))

        param_updates: List[np.ndarray] = []
        is_byz_list: List[bool] = []
        for i, p in enumerate(participants):
            is_byz = (i in self.byzantine_ids) and active

            if is_byz and self.attack_type == "label_flipping":
                model = self._make_model(p.n_features)
                _, y_p = self._sim.poison_data(
                    p.X_train, p.y_train, "label_flipping",
                    n_classes=self.n_classes_data,
                )
                delta = model.fit(p.X_train, y_p)
            elif is_byz:
                peers = [u for j, u in enumerate(honest_updates) if j != i]
                delta = self._sim.poison_update(
                    honest_updates[i], self.attack_type,
                    reference=server_delta, peer_updates=peers,
                )
            else:
                delta = honest_updates[i]

            param_updates.append(delta)
            is_byz_list.append(is_byz)

        agg_kwargs: Dict = {}
        if self.aggregation == "fltrust":
            agg_kwargs["server_update"] = server_delta
        else:
            agg_kwargs["sample_sizes"] = [p.n_train for p in participants]

        agg_delta, metadata = self._strategy.aggregate(param_updates, **agg_kwargs)
        self._global_params = self._global_params + agg_delta  # type: ignore[operator]

        per_acc = {
            p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test)
            for p in participants
        }
        global_acc = float(np.mean(list(per_acc.values())))

        trust_scores: Optional[Dict[str, float]] = None
        n_accepted: Optional[int] = None
        if metadata and "trust_scores" in metadata:
            raw = metadata["trust_scores"]
            trust_scores = {p.id: ts for p, ts in zip(participants, raw)}
            n_accepted = sum(1 for ts in raw if ts > 0)

        return RoundResult(round_num, global_acc, per_acc, trust_scores, n_accepted)


def compute_param_updates_auto(
    learner: "AttackedFederatedLearner",
    participants: List[ParticipantData],
    active: bool,
    root_data: Optional[Dict],
) -> Tuple[List[np.ndarray], List[bool]]:
    """Dispatches to the blind (`AttackedFederatedLearner._compute_param_updates`)
    or informed (inlined below, mirroring `InformedAttackedFederatedLearner._run_round`)
    update computation depending on whether `learner.attack_type` is one of
    `AttackSimulator.INFORMED_ATTACK_TYPES`.

    Exists because `InformedAttackedFederatedLearner` only overrides
    `_run_round` wholesale — fine for a learner dedicated to informed
    attacks, but callers that need to support BOTH families through a
    single learner class (any subclass whose `attack_type` varies per call,
    e.g. `src/fl/gradf_learner.py::GRADFFederatedLearner` and
    `src/experiments/exp10_selector_comparison.py`'s custom learners)
    previously had no way to get the informed path without duplicating that
    class's logic — they called the plain blind `_compute_param_updates`
    unconditionally, which silently degrades informed attack types to their
    blind fallback (e.g. `AttackSimulator._fltrust_aligned` without
    `reference` falls back to a plain sign-flip, indistinguishable from
    `sign_flipping` — a real bug found and fixed via this helper; see
    `references/resultado_experimento_seletores_adaptativos.md`).
    """
    from src.classification.attack_simulator import AttackSimulator

    if learner.attack_type not in AttackSimulator.INFORMED_ATTACK_TYPES:
        return learner._compute_param_updates(participants, active)

    root = root_data or learner._carve_root(participants[0])
    server_delta = learner._make_model(participants[0].n_features).fit(root["X"], root["y"])

    honest_updates: List[np.ndarray] = []
    for p in participants:
        model = learner._make_model(p.n_features)
        honest_updates.append(model.fit(p.X_train, p.y_train))

    param_updates: List[np.ndarray] = []
    is_byz_list: List[bool] = []
    for i, p in enumerate(participants):
        is_byz = (i in learner.byzantine_ids) and active
        if is_byz:
            peers = [u for j, u in enumerate(honest_updates) if j != i]
            delta = learner._sim.poison_update(
                honest_updates[i], learner.attack_type,
                reference=server_delta, peer_updates=peers,
            )
        else:
            delta = honest_updates[i]
        param_updates.append(delta)
        is_byz_list.append(is_byz)

    return param_updates, is_byz_list


def attack_for_round(sequence: List[str], round_num: int) -> str:
    """Cycle through `sequence` by round (1-indexed). Factored out here (from
    `src/experiments/exp4_adaptive.py`) so it can be reused by
    `src.utils.data_loader.generate_rotating_selector_experiences` without a
    circular import (`utils` cannot import from `experiments`, which already
    imports from `utils`)."""
    return sequence[(round_num - 1) % len(sequence)]


class RotatingAttackedFederatedLearner(AttackedFederatedLearner):
    """AttackedFederatedLearner whose `attack_type` changes every round,
    cycling through `attack_sequence` — used both to simulate an adversary
    that alternates attack strategy (`AdaptiveAttacker`/`TARSLearner` in
    `exp4_adaptive.py`, which override `_run_round` more elaborately but
    reuse `attack_for_round`) and to generate selector pretraining
    experiences UNDER rotation (`generate_rotating_selector_experiences`),
    reducing the mismatch between training (which previously only used
    single, fixed-attack episodes) and the real deployment tested in Gate 1
    (`exp4_adaptive.py`)."""

    def __init__(
        self, *args,
        strategy: str = "median",
        attack_sequence: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.attack_sequence = attack_sequence or ["sign_flipping", "gaussian_noise", "label_flipping"]
        kwargs.setdefault("attack_type", self.attack_sequence[0])
        kwargs.setdefault("aggregation", strategy)
        super().__init__(*args, **kwargs)

    def _run_round(self, round_num, participants, root_data):
        self.attack_type = attack_for_round(self.attack_sequence, round_num)
        return super()._run_round(round_num, participants, root_data)
