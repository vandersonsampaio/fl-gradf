"""W2 (results/w2_backdoor_asr/PLANO.md §5.2): the pixel_backdoor data attack."""

import numpy as np
import pytest

from src.classification.attack_simulator import AttackSimulator, apply_trigger, attack_success_rate


def _data(n=40, side=28, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 0.5, size=(n, side * side)).astype(np.float32)
    y = rng.integers(0, 10, size=n)
    return X, y


@pytest.mark.parametrize("ts", [4, 6])
def test_trigger_changes_only_the_corner(ts):
    X, _ = _data()
    Xt = apply_trigger(X, ts)
    img, imgt = X.reshape(-1, 28, 28), Xt.reshape(-1, 28, 28)
    assert np.all(imgt[:, 28 - ts:, 28 - ts:] == 1.0)
    mask = np.ones((28, 28), bool)
    mask[28 - ts:, 28 - ts:] = False
    assert np.array_equal(img[:, mask], imgt[:, mask])
    assert int((X != Xt).sum()) == len(X) * ts * ts  # inputs are < 1, so every trigger pixel changes


def test_trigger_rejects_non_square():
    with pytest.raises(ValueError):
        apply_trigger(np.zeros((2, 10)), 4)


@pytest.mark.parametrize("p", [0.5, 0.8])
def test_poison_fraction_target_and_no_mutation(p):
    X, y = _data()
    X0, y0 = X.copy(), y.copy()
    Xp, yp = AttackSimulator(seed=1).poison_data(X, y, "pixel_backdoor", poison_fraction=p,
                                                 target_class=0, trigger_size=4)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)
    stamped = np.all(Xp.reshape(-1, 28, 28)[:, 24:, 24:] == 1.0, axis=(1, 2))
    assert stamped.sum() == round(p * len(y))
    assert np.all(yp[stamped] == 0)
    assert np.array_equal(yp[~stamped], y[~stamped]) and np.array_equal(Xp[~stamped], X[~stamped])


def test_poison_update_rejects_data_attack():
    with pytest.raises(ValueError):
        AttackSimulator().poison_update(np.zeros(5), "pixel_backdoor")


def test_asr_counts_only_non_target_samples():
    class Const:
        def __init__(self, c):
            self.c = c

        def predict(self, X):
            return np.full(len(X), self.c)

    X, y = _data()
    assert attack_success_rate(Const(0), X, y, target_class=0) == 1.0
    assert attack_success_rate(Const(3), X, y, target_class=0) == 0.0
