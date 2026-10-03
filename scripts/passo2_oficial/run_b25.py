"""
B2.5 (references/roadmap_tese_gradf_v4.md §4.1): ataque IPM no AdaAggRL oficial.
Plano em results/b25_ipm_oficial/PLANO.md.

O IPM do código oficial é um update NULO: IPM_attack chama
craft(old_weights, get_parameters(net), 5, -1) com a rede já posta em
old_weights, então weight_diff = 0 e o atacante envia o próprio modelo global
(auditoria §2.2). Aqui:
  IPM_oficial   o ataque do repositório, sem mudança (para documentar a nulidade)
  IPM_real      IPM de Xie et al.: delta forjado = −ε × média dos deltas honestos
                da rodada (atacante onisciente); peso enviado = old + delta forjado
  none          sem ataque (mesma partição: os atacantes são sorteados como no
                oficial e depois esvaziados, para preservar o RNG)

Implementação, sem editar o código oficial: o `IPM_attack` oficial não recebe os
updates honestos, mas o `LMP_attack` recebe (a lista de pesos honestos já
calculados na rodada). Para o IPM_real, o ambiente roda no caminho do LMP e, neste
processo, `exp_environments.LMP_attack` é trocado pela função de IPM real. O JSON
registra o ataque efetivo (`attack_label`) e o ε.

Condições:
  td3      TD3 como no main.py oficial (via run_b21: estados e checkpoints do ator)
  fixed    ação [0,475]*5 (centro do Box), como no B2.1
  fedavg   agregação uniforme (sem o filtro oficial; só para o sanity check)

Uso (venv oficial):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b25.py --attack IPM_real --eps 10 --condition fixed --seed 100 --rounds 100
"""

import argparse
import contextlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b21 as B21  # noqa: E402

R = B21.R
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

DEFAULT_OUT = os.path.join(R.REPO, "results", "b25_ipm_oficial", "raw")
ENV_ATTACK = {"IPM_real": "LMP", "IPM_oficial": "IPM", "none": "LMP"}


def ipm_real_attack(eps: float):
    """Devolve uma função com a assinatura do LMP_attack oficial que implementa o IPM real."""
    def _attack(net, old_weights, cids, att_ids, trainloaders, weights_lis):
        print(f"----------IPM real (B2.5, eps={eps:g}) Attack--------------")
        n_layers = len(old_weights)
        mean_delta = [np.mean([np.asarray(w[i], dtype=np.float64) - old_weights[i] for w in weights_lis], axis=0)
                      for i in range(n_layers)]
        crafted = [(old_weights[i] - eps * mean_delta[i]).astype(old_weights[i].dtype) for i in range(n_layers)]
        return {cid: [c.copy() for c in crafted] for cid in att_ids}
    return _attack


def _uniform_preview(env, action):
    n = len(env.cids)
    return np.ones(n) / n


def run(attack_label: str, eps: float, condition: str, seed: int, rounds: int, out_dir: str,
        q: float = 0.5, dataset: str = "MNIST") -> str:
    out_dir = os.path.abspath(out_dir)
    for sub in ("", "obs", "actors"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    alabel = f"IPMr{eps:g}" if attack_label == "IPM_real" else attack_label
    tag = f"{dataset}_{alabel}_q{q}_{condition}_seed{seed}_R{rounds}"
    out_path = os.path.join(out_dir, f"{tag}.json")
    partial_path = os.path.join(out_dir, f"{tag}.partial.json")
    log_path = os.path.join(out_dir, f"{tag}.stdout.log")

    cwd = os.getcwd()
    os.chdir(R.OFICIAL)
    try:
        import exp_environments as E
        E.SummaryWriter = R._NoOpWriter
        if attack_label == "IPM_real":
            E.LMP_attack = ipm_real_attack(eps)
        if condition == "fedavg":
            E.aggeregate = lambda new_weights, fractions: E.average(new_weights)
            R._preview_weights = _uniform_preview

        t_start = time.perf_counter()
        with open(log_path, "w", buffering=1) as log_stream:
            with contextlib.redirect_stdout(log_stream):
                inner = E.FL_mnist(R._official_args(dataset, ENV_ATTACK[attack_label], q))  # random.seed(150)
            if attack_label == "none":
                inner.att_ids = []  # sorteio feito (RNG preservado), atacantes esvaziados
            set_random_seed(seed, using_cuda=torch.cuda.is_available())
            env = B21.ObsRecordingAdapter(inner, log_stream)
            env.action_space.seed(seed)

            def _checkpoint(t):
                if t % R.CHECKPOINT_EVERY == 0:
                    R._dump(R._record(inner, env, E, alabel, condition, seed, rounds, q, dataset,
                                      time.perf_counter() - t_start), partial_path)
            env.on_step = _checkpoint

            if condition == "td3":
                n_actions = env.action_space.shape[-1]
                noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                model = R.TD3("MlpPolicy", env, buffer_size=1000, policy_kwargs={"net_arch": [256, 128]},
                              verbose=0, gamma=0.99, action_noise=noise, learning_rate=1e-5,
                              train_freq=(3, "step"), batch_size=64, seed=seed)
                model.learn(total_timesteps=rounds, log_interval=None,
                            callback=B21.ActorCheckpoint(os.path.join(out_dir, "actors", tag)))
            else:
                env.reset()
                for _ in range(rounds):
                    _o, _r, term, _t, _ = env.step(np.asarray(R.A_FIXED, dtype=np.float32))
                    if term:
                        env.reset()
        elapsed = time.perf_counter() - t_start
    finally:
        os.chdir(cwd)

    np.save(os.path.join(out_dir, "obs", f"{tag}.npy"), np.stack(env.obs_in).astype(np.float32))
    rec = R._record(inner, env, E, alabel, condition, seed, rounds, q, dataset, elapsed)
    rec.update({"attack_label": attack_label, "eps": eps if attack_label == "IPM_real" else None,
                "env_attack_path": ENV_ATTACK[attack_label]})
    R._dump(rec, out_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, choices=list(ENV_ATTACK))
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--condition", required=True, choices=["td3", "fixed", "fedavg"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    a = p.parse_args()
    if a.attack == "IPM_real" and a.eps is None:
        p.error("--eps é obrigatório para IPM_real")
    print(run(a.attack, a.eps, a.condition, a.seed, a.rounds, a.out_dir))


if __name__ == "__main__":
    main()
