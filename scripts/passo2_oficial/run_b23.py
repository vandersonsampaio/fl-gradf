"""
B2.3 (references/roadmap_tese_gradf_v2.md §4): steelman do TD3 no AdaAggRL
oficial. Pergunta: dando ao TD3 condições muito mais favoráveis de aprender,
ele passa a superar a ação fixa?

Única mudança em relação ao TD3 oficial (main.py):
  learning_rate   1e-5 -> 1e-3   (100x; ator e crítico, Adam do SB3)
  learning_starts 100  -> 10     (aquecimento com ação aleatória 10x menor)
Todo o resto é idêntico: MlpPolicy [256,128], buffer 1000, batch 64,
train_freq 3, ruído N(0, 0,1), gamma 0,99, mesmo ambiente e mesmas sementes.

Exploratório, em sementes gastas (100-104), pareado com os runs `fixed` e `td3`
do Passo 2 (results/frente1_passo2_oficial/raw/). Reusa `run_b21.run` sem
alterá-lo (inclusive o log de estados e os checkpoints do ator), trocando só o
construtor do TD3 neste processo.

Uso (sempre com o venv isolado):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b23.py --attack EB --seed 100
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b21 as B21  # noqa: E402

STEELMAN = {"learning_rate": 1e-3, "learning_starts": 10}
DEFAULT_OUT = os.path.join(B21.R.REPO, "results", "b23_steelman_oficial", "raw")

_official_td3 = B21.R.TD3
_used_kwargs = {}


def _steelman_td3(*args, **kwargs):
    """Substitui só lr e learning_starts; registra os kwargs efetivos."""
    kwargs.update(STEELMAN)
    _used_kwargs.clear()
    _used_kwargs.update({k: v for k, v in kwargs.items() if k not in ("action_noise",)})
    return _official_td3(*args, **kwargs)


def run(attack: str, seed: int, rounds: int, q: float, dataset: str, out_dir: str) -> str:
    B21.R.TD3 = _steelman_td3  # vale só neste processo
    path = B21.run(attack, "td3", seed, rounds, q, dataset, out_dir)
    d = json.load(open(path))
    d["variant"] = "td3_steelman"
    d["td3_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v))
                       for k, v in _used_kwargs.items()}
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
