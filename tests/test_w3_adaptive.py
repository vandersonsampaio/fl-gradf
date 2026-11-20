"""W3 (results/w3_adaptativo_limiar/PLANO.md §5.2) and the shared W helpers."""

import numpy as np

from scripts.w3_adaptativo_limiar import aimd, mix_update
from scripts.w_common import WeightTap, classify_family, holm_adjust, paired_stats


def test_mix_endpoints():
    rng = np.random.default_rng(0)
    h, hb = rng.normal(size=7), rng.normal(size=7)
    assert np.array_equal(mix_update(h, hb, 0.0, 2.0), h)
    assert np.allclose(mix_update(h, hb, 1.0, 2.0), -2.0 * hb)


def test_aimd_bounds():
    assert aimd(0.1, True, 0.1) == 0.2
    assert aimd(0.95, True, 0.1) == 1.0
    assert aimd(0.4, False, 0.1) == 0.2
    e = 0.1
    for _ in range(50):
        e = aimd(e, False, 0.1)
    assert 0 < e < 1e-10


def test_tap_normalized():
    t = WeightTap()
    assert t.normalized() is None
    t.weights = np.array([0.0, 1.0, 3.0])
    assert np.allclose(t.normalized(), [0, 0.25, 0.75])
    t.weights = np.zeros(4)
    assert np.allclose(t.normalized(), 0.25)


def test_holm_and_verdicts():
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    big = paired_stats(np.array([5.0, 6, 7, 5.5, 6.5]), margin=1.0)
    eq = paired_stats(np.array([0.01, -0.02, 0.0, 0.015, -0.01]), margin=1.0)
    rows = classify_family([big, eq])
    assert rows[0]["verdict"] == "superiority"
    assert rows[1]["verdict"] == "equivalence"
