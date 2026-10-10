"""
B2.1 + B2.2 (results/b21_replicacao_oficial/PREREGISTRO.md): confirmatory replication
in the official AdaAggRL, seeds 105-114, `fixed` vs. `td3`, plus the
instrumentation for the B2.2 mechanistic hypotheses.

Reuses `run_oficial.py` (Step 2) without changing it: same official environment, same
Gymnasium adapter, same conditions, same TD3 hyperparameters and same
seed control. It only adds logging:

  1. Observed states: the state the policy receives before each action
     (10 clients x 4 cues) is saved to `obs/<tag>.npy`, with shape (steps, 10, 4).
  2. TD3 actor checkpoints: `model.actor.state_dict()` at the start of
     training (step 0, initial policy) and every 50 environment steps, in
     `actors/<tag>_stepNNN.pt`. With this, B2.2 evaluates the deterministic policy
     (without exploration noise) on swapped states.

Usage (always with the isolated venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b21.py \
      --attack EB --condition td3 --seed 105
"""

import argparse
import contextlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_oficial as R  # noqa: E402  (also sets sys.path for the shim and the official code)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

ACTOR_EVERY = 50
DEFAULT_OUT = os.path.join(R.REPO, "results", "b21_replicacao_oficial", "raw")


class ObsRecordingAdapter(R.OfficialEnvAdapter):
    """Same as the Step 2 adapter, plus recording the state the policy
    sees before each action (the observation returned by the previous reset or step)."""

    def __init__(self, inner, log_stream):
        super().__init__(inner, log_stream)
        self.obs_in = []
        self._cur_obs = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._cur_obs = obs.copy()
        return obs, info

    def step(self, action):
        self.obs_in.append(self._cur_obs.copy())
        obs, reward, term, trunc, info = super().step(action)
        self._cur_obs = obs.copy()
        return obs, reward, term, trunc, info


class ActorCheckpoint(BaseCallback):
    """Saves the actor state_dict at the start of training and every ACTOR_EVERY steps."""

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def _save(self, step: int) -> None:
        torch.save(self.model.actor.state_dict(), f"{self.prefix}_step{step:03d}.pt")

    def _on_training_start(self) -> None:
        self._save(0)

    def _on_step(self) -> bool:
        if self.num_timesteps % ACTOR_EVERY == 0:
            self._save(self.num_timesteps)
        return True


def run(attack: str, condition: str, seed: int, rounds: int, q: float, dataset: str, out_dir: str) -> str:
    out_dir = os.path.abspath(out_dir)
    for sub in ("", "obs", "actors"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    tag = f"{dataset}_{attack}_q{q}_{condition}_seed{seed}_R{rounds}"
    out_path = os.path.join(out_dir, f"{tag}.json")
    partial_path = os.path.join(out_dir, f"{tag}.partial.json")
    log_path = os.path.join(out_dir, f"{tag}.stdout.log")
    obs_path = os.path.join(out_dir, "obs", f"{tag}.npy")
    actor_prefix = os.path.join(out_dir, "actors", tag)

    cwd = os.getcwd()
    os.chdir(R.OFICIAL)
    try:
        import exp_environments as E
        E.SummaryWriter = R._NoOpWriter

        t_start = time.perf_counter()
        with open(log_path, "w", buffering=1) as log_stream:
            with contextlib.redirect_stdout(log_stream):
                inner = E.FL_mnist(R._official_args(dataset, attack, q))  # calls random.seed(150)
            set_random_seed(seed, using_cuda=torch.cuda.is_available())
            env = ObsRecordingAdapter(inner, log_stream)
            env.action_space.seed(seed)

            def _checkpoint(t):
                if t % R.CHECKPOINT_EVERY == 0:
                    R._dump(R._record(inner, env, E, attack, condition, seed, rounds, q, dataset,
                                      time.perf_counter() - t_start), partial_path)
            env.on_step = _checkpoint

            if condition == "td3":
                n_actions = env.action_space.shape[-1]
                noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                model = R.TD3(
                    "MlpPolicy", env, buffer_size=1000, policy_kwargs={"net_arch": [256, 128]},
                    verbose=0, gamma=0.99, action_noise=noise, learning_rate=1e-5,
                    train_freq=(3, "step"), batch_size=64, seed=seed,
                )
                model.learn(total_timesteps=rounds, log_interval=None,
                            callback=ActorCheckpoint(actor_prefix))
            elif condition == "fixed":
                env.reset()
                for _ in range(rounds):
                    _obs, _r, term, _trunc, _ = env.step(np.asarray(R.A_FIXED, dtype=np.float32))
                    if term:
                        env.reset()
            else:
                raise ValueError(f"B2.1 only has the td3 and fixed conditions, not '{condition}'")
        elapsed = time.perf_counter() - t_start
    finally:
        os.chdir(cwd)

    np.save(obs_path, np.stack(env.obs_in).astype(np.float32))
    R._dump(R._record(inner, env, E, attack, condition, seed, rounds, q, dataset, elapsed), out_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, choices=["LMP", "EB"])
    p.add_argument("--condition", required=True, choices=["td3", "fixed"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--q", type=float, default=0.5)
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    a = p.parse_args()
    print(run(a.attack, a.condition, a.seed, a.rounds, a.q, a.dataset, a.out_dir))


if __name__ == "__main__":
    main()
