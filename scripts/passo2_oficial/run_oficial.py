"""
Step 2 (results/frente1_passo2_oficial/): does TD3 contribute in the official
AdaAggRL, at the official horizon?

Runs the published code (external/AdaAggRL, commit 27b9c18) WITHOUT editing any
of its files, under three conditions:

  td3     SB3 TD3 exactly as in main.py (MlpPolicy [256,128], lr 1e-5,
          buffer 1000, batch 64, train_freq 3, noise N(0, 0.1), gamma 0.99,
          default learning_starts = 100).
  fixed   constant action at the center of the official Box: [0.475]*5. Constant a[:4]
          -> uniform softmax; a5 = 0.475 = mean of the TD3 actions in the random
          phase (rounds 1-100).
  random  uniform action in [0, 0.95]^5 for the whole run (what TD3
          executes in the first 100 rounds).

Minimal environment adjustments, all outside the official code:

  1. `inversefed.data.consts` (not distributed) -> shim/ with Geiping's original
     constants.
  2. Old gym 0.26 API (4-value step, reset without seed) -> Gymnasium adapter
     `OfficialEnvAdapter`; SB3 2.3.2 does not accept the raw environment.
  3. Seeds: the official __init__ calls random.seed(150), which fixes attackers and
     partition and is kept. After it, SB3's `set_random_seed(seed)` (random,
     numpy, torch) is called in ALL conditions, so the sequence of
     sampled clients (random.sample) is paired across conditions until the
     first reset.
  4. SummaryWriter -> no-op; main.py's tensorboard_log and CheckpointCallback
     removed (logging only).
  5. --dataset MNIST (main.py's default is CIFAR10).

Nothing changes in the logic of the environment, the attacks, the reward (including the use
of the testloader) or TD3.

Usage (always with the isolated venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_oficial.py \
      --attack LMP --condition td3 --seed 100 --rounds 500 --q 0.5
"""

import argparse
import contextlib
import json
import os
import random
import sys
import time
from argparse import Namespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OFICIAL = os.path.join(REPO, "external", "AdaAggRL")
sys.path[:0] = [os.path.join(HERE, "shim"), OFICIAL]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

A_FIXED = [0.475] * 5
OFFICIAL_COMMIT = "27b9c18"


class _NoOpWriter:
    def __init__(self, *a, **k):
        pass

    def add_scalar(self, *a, **k):
        pass


def _official_args(dataset: str, attack: str, q: float) -> Namespace:
    """Same defaults as main.py::get_args, except dataset/attack/q."""
    return Namespace(
        batch_size=64, q=q, num_clients=100, subsample_rate=0.1, num_attacker=20,
        num_class=10, fl_epoch=500, lr=0.05, dataset=dataset, dummy_batch_size=16,
        attack=attack,
    )


def _preview_weights(env, action):
    """Replicates exp_environments.py:122-137, WITHOUT mutating state, to record
    how much weight the action gives to attackers in the aggregation the step is about to do."""
    old_state = np.asarray(env.old_state, dtype=np.float64)
    a0 = torch.softmax(torch.tensor(np.asarray(action[:4], dtype=np.float32)), dim=0).numpy()
    k = torch.tensor(np.dot(old_state, a0))
    rng = torch.max(k) - torch.min(k)
    if not torch.isfinite(rng) or rng == 0:
        return None
    k = (k - torch.min(k)) / rng
    k = 0.5 * (1 - torch.cos(3.14 * k))
    k = k / torch.sum(k)
    delta = torch.max(k) * float(action[4])
    w = []
    for i, cid in enumerate(env.cids):
        if k[i] <= delta:
            w.append(0.0)
        else:
            w.append(float(k[i]) * (0.9 ** env.client_state[cid]["flag"]))
    w = np.asarray(w)
    return w / w.sum() if w.sum() > 0 else w


class OfficialEnvAdapter(gymnasium.Env):
    """Gymnasium adapter for exp_environments.FL_mnist. Forwards step/reset
    without changing anything and records what the experiment needs."""

    metadata = {"render_modes": []}

    def __init__(self, inner, log_stream):
        super().__init__()
        self.inner = inner
        self.log_stream = log_stream
        self.action_space = gymnasium.spaces.Box(
            low=inner.action_space.low, high=inner.action_space.high,
            shape=inner.action_space.shape, dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=inner.observation_space.shape, dtype=np.float32,
        )
        self.steps = []
        self.resets = []
        self._real_att = set()  # attackers that actually attacked in the pending updates
        self._t = 0
        self.on_step = None  # checkpoint callback (logging only)

    def reset(self, seed=None, options=None):
        with contextlib.redirect_stdout(self.log_stream):
            obs = self.inner.reset()
        self.resets.append({"env_step": self._t, "acc": float(self.inner.acc), "loss": float(self.inner.loss)})
        self._real_att = set()  # the official reset trains every client honestly
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        inner = self.inner
        action = np.asarray(action, dtype=np.float32)
        att = set(inner.att_ids)
        agg_cids = list(inner.cids)
        w = _preview_weights(inner, action)
        att_mask = np.array([c in self._real_att for c in agg_cids])
        times_before = {c: inner.client_state[c]["times"] for c in range(len(inner.client_state))}

        t0 = time.perf_counter()
        with contextlib.redirect_stdout(self.log_stream):
            obs, reward, done, info = inner.step(action)
        dt = time.perf_counter() - t0
        self._t += 1

        obs = np.asarray(obs, dtype=np.float32)
        new_cids = list(inner.cids)
        self._real_att = {c for c in new_cids if c in att and times_before[c] >= 1}
        self.steps.append({
            "t": self._t,
            "action": [float(x) for x in action],
            "reward": float(reward),
            "loss": float(inner.history["loss"][-1]),
            "acc": float(inner.history["acc"][-1]),
            "n_att_sampled": int(sum(c in att for c in agg_cids)),
            "n_att_real": int(att_mask.sum()),
            "att_weight_mass": None if w is None else float(w[att_mask].sum()),
            "n_excluded": None if w is None else int((w == 0).sum()),
            "n_att_excluded": None if w is None else int(((w == 0) & att_mask).sum()),
            "simlc_rule_rows": int(np.sum((obs[:, 0] == 0) & (obs[:, 1] == 0))),
            "done": bool(done),
            "sec": dt,
        })
        if self.on_step is not None:
            self.on_step(self._t)
        return obs, float(reward), bool(done), False, {}


CHECKPOINT_EVERY = 25


def _record(inner, env, E, attack, condition, seed, rounds, q, dataset, elapsed):
    return {
        "official_commit": OFFICIAL_COMMIT, "dataset": dataset, "attack": attack, "q": q,
        "condition": condition, "seed": seed, "rounds": rounds,
        "a_fixed": A_FIXED if condition == "fixed" else None,
        "att_ids": [int(c) for c in inner.att_ids],
        "history_acc": [float(x) for x in inner.history["acc"]],
        "history_loss": [float(x) for x in inner.history["loss"]],
        "resets": env.resets, "steps": env.steps,
        "elapsed_sec": elapsed, "device": str(E.DEVICE),
        "torch": torch.__version__,
    }


def _dump(record, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f)
    os.replace(tmp, path)


def run(attack: str, condition: str, seed: int, rounds: int, q: float, dataset: str, out_dir: str) -> str:
    out_dir = os.path.abspath(out_dir)  # the chdir below would break relative paths
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{dataset}_{attack}_q{q}_{condition}_seed{seed}_R{rounds}"
    out_path = os.path.join(out_dir, f"{tag}.json")
    partial_path = os.path.join(out_dir, f"{tag}.partial.json")
    log_path = os.path.join(out_dir, f"{tag}.stdout.log")

    cwd = os.getcwd()
    os.chdir(OFICIAL)  # the official code reads ./extract_feature.pt and ./data
    try:
        import exp_environments as E
        E.SummaryWriter = _NoOpWriter

        t_start = time.perf_counter()
        with open(log_path, "w", buffering=1) as log_stream:  # line-buffered: visible progress
            with contextlib.redirect_stdout(log_stream):
                inner = E.FL_mnist(_official_args(dataset, attack, q))  # calls random.seed(150)
            set_random_seed(seed, using_cuda=torch.cuda.is_available())
            env = OfficialEnvAdapter(inner, log_stream)
            env.action_space.seed(seed)

            def _checkpoint(t):  # logging only: saves the partial so a crashed run is not lost
                if t % CHECKPOINT_EVERY == 0:
                    rec = _record(inner, env, E, attack, condition, seed, rounds, q, dataset,
                                  time.perf_counter() - t_start)
                    _dump(rec, partial_path)
            env.on_step = _checkpoint

            if condition == "td3":
                n_actions = env.action_space.shape[-1]
                noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                model = TD3(
                    "MlpPolicy", env, buffer_size=1000, policy_kwargs={"net_arch": [256, 128]},
                    verbose=0, gamma=0.99, action_noise=noise, learning_rate=1e-5,
                    train_freq=(3, "step"), batch_size=64, seed=seed,
                )
                model.learn(total_timesteps=rounds, log_interval=None)
            else:
                env.reset()
                for _ in range(rounds):
                    action = np.asarray(A_FIXED, dtype=np.float32) if condition == "fixed" else env.action_space.sample()
                    _obs, _r, term, _trunc, _ = env.step(action)
                    if term:
                        env.reset()
        elapsed = time.perf_counter() - t_start
    finally:
        os.chdir(cwd)

    _dump(_record(inner, env, E, attack, condition, seed, rounds, q, dataset, elapsed), out_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, choices=["IPM", "LMP", "EB"])
    p.add_argument("--condition", required=True, choices=["td3", "fixed", "random"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--q", type=float, default=0.5)
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--out_dir", default=os.path.join(REPO, "results", "frente1_passo2_oficial", "raw"))
    a = p.parse_args()
    path = run(a.attack, a.condition, a.seed, a.rounds, a.q, a.dataset, a.out_dir)
    print(path)


if __name__ == "__main__":
    main()
