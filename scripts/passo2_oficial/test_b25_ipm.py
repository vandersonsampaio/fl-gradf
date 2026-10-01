"""
Teste unitário do B2.5 (plano em results/b25_ipm_oficial/PLANO.md):
  1. IPM real: delta forjado tem cos ≈ −1 com a média dos deltas honestos e
     razão de normas ≈ ε (ε = 2 e 10), em pesos sintéticos e numa rodada real
     do ambiente oficial (MNISTClassifier).
  2. IPM oficial (attack_utilities.IPM_attack): o update forjado tem norma zero
     (o atacante devolve o próprio modelo global).

Uso (venv oficial):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/test_b25_ipm.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b25 as B25  # noqa: E402

R = B25.R
import numpy as np  # noqa: E402


def _vec(ws):
    return np.concatenate([np.asarray(w, dtype=np.float64).ravel() for w in ws])


def _check_ipm_real(old, honest, eps, label):
    crafted = B25.ipm_real_attack(eps)(None, old, [0, 1, 2], [7, 9], None, honest)
    assert set(crafted) == {7, 9}
    mean_delta = np.mean([_vec(h) - _vec(old) for h in honest], axis=0)
    for cid, w in crafted.items():
        d = _vec(w) - _vec(old)
        cos = d @ mean_delta / (np.linalg.norm(d) * np.linalg.norm(mean_delta))
        ratio = np.linalg.norm(d) / np.linalg.norm(mean_delta)
        print(f"  {label} ε={eps:g} cid={cid}: cos={cos:.6f} razão={ratio:.6f}")
        assert cos < -0.9999, cos
        assert abs(ratio - eps) / eps < 1e-4, ratio


def main():
    rng = np.random.default_rng(0)
    shapes = [(32, 1, 3, 3), (32,), (10, 128), (10,)]
    old = [rng.normal(size=s).astype(np.float32) for s in shapes]
    honest = [[o + 0.01 * rng.normal(size=o.shape).astype(np.float32) + 0.005 for o in old] for _ in range(8)]
    print("1a. IPM real, pesos sintéticos")
    for eps in (2.0, 10.0):
        _check_ipm_real(old, honest, eps, "sintético")

    os.chdir(R.OFICIAL)
    import torch
    from utilities import MNISTClassifier, get_parameters, set_parameters, train_real
    from attack_utilities import IPM_attack
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    net = MNISTClassifier()
    # cópias: na CPU get_parameters devolve views da rede (no ambiente oficial há deepcopy)
    old = [w.copy() for w in get_parameters(net)]
    loader = DataLoader(TensorDataset(torch.randn(128, 1, 28, 28), torch.randint(0, 10, (128,))), batch_size=64)
    honest = []
    for _ in range(4):
        set_parameters(net, old)
        train_real(net, loader, epochs=1, lr=0.05)
        honest.append([w.copy() for w in get_parameters(net)])
    print("1b. IPM real, MNISTClassifier do código oficial")
    for eps in (2.0, 10.0):
        _check_ipm_real(old, honest, eps, "MNISTClassifier")

    print("2. IPM oficial (como chamado em exp_environments.step: rede posta em old_weights antes)")
    set_parameters(net, old)
    crafted = {c: [w.copy() for w in ws] for c, ws in IPM_attack(net, old, [0, 1, 2, 7], [7]).items()}
    n = np.linalg.norm(_vec(crafted[7]) - _vec(old))
    print(f"  norma do update forjado = {n:.3e}")
    assert n == 0.0, n
    print("OK")


if __name__ == "__main__":
    main()
