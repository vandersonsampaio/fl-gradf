"""
B2.3b (references/roadmap_tese_gradf_v3.md §4.1): ÚLTIMO steelman do TD3 no
AdaAggRL oficial. Pergunta: com a recompensa bem condicionada, o TD3 aprende
uma política dependente do estado e supera a ação fixa?

Motivo: no B2.3 (lr 1e-3, learning_starts 10) a política saturou num canto
constante do Box. A hipótese de condicionamento é a escala da recompensa
oficial (SOMA da loss sobre ~156 batches do teste, dezenas a centenas por
rodada, sem normalização).

Mudanças em relação ao TD3 oficial (main.py):
  recompensa      normalizada pelo VecNormalize do SB3 (norm_reward=True,
                  gamma=0,99, clip 10); observações NÃO normalizadas, então o
                  ator continua avaliável com as ferramentas do B2.2
  learning_rate   1e-5 -> 1e-4   (10x; ator e crítico)
  learning_starts 100  -> 10     (igual ao B2.3)
Todo o resto é idêntico ao oficial e ao B2.3. A recompensa registrada no JSON
continua sendo a bruta (o adaptador registra antes do wrapper).

Reusa `run_b21.run` sem alterá-lo (log de estados e checkpoints do ator),
trocando só o construtor do TD3 neste processo.

Uso (sempre com o venv isolado):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b23b.py --attack EB --seed 100
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b21 as B21  # noqa: E402

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

STEELMAN = {"learning_rate": 1e-4, "learning_starts": 10}
VECNORM = {"norm_obs": False, "norm_reward": True, "gamma": 0.99, "clip_reward": 10.0}
DEFAULT_OUT = os.path.join(B21.R.REPO, "results", "b23b_steelman_normalizado", "raw")

_official_td3 = B21.R.TD3
_used = {}


def _steelman_td3(policy, env, **kwargs):
    """Troca lr e learning_starts e envolve o ambiente num VecNormalize só de recompensa."""
    kwargs.update(STEELMAN)
    venv = VecNormalize(DummyVecEnv([lambda: env]), **VECNORM)
    _used.clear()
    _used.update({k: v for k, v in kwargs.items() if k != "action_noise"})
    _used["vecnormalize"] = dict(VECNORM)
    return _official_td3(policy, venv, **kwargs)


def run(attack: str, seed: int, rounds: int, q: float, dataset: str, out_dir: str) -> str:
    B21.R.TD3 = _steelman_td3  # vale só neste processo
    path = B21.run(attack, "td3", seed, rounds, q, dataset, out_dir)
    d = json.load(open(path))
    d["variant"] = "td3_steelman_normalizado"
    d["td3_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v))
                       for k, v in _used.items()}
    B21.R._dump(d, path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, choices=["LMP", "EB"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--q", type=float, default=0.5)
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    a = p.parse_args()
    print(run(a.attack, a.seed, a.rounds, a.q, a.dataset, a.out_dir))


if __name__ == "__main__":
    main()
