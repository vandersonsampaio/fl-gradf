"""
B3.0: trains the BloodMNIST feature extractor for the official AdaAggRL
(results/b31_medmnist_oficial/PREREGISTRO.md §2). The official `extract_feature.pt` is
an MNISTClassifier trained on MNIST (1 channel); the repository does not ship the training
script. Here: the same architecture (BloodClassifier = MNISTClassifier with 3 channels and
8 classes), centralized training on the BloodMNIST training split, Adam lr 1e-3,
batch 128, 15 epochs, seed 0, with the same transforms as the environment.
Saves the full state_dict (the environment drops the last layer when loading).
Does not touch the MNIST extractor or data.

Usage (official venv): external/.venv_adaaggrl/bin/python scripts/passo2_oficial/treinar_extrator_bloodmnist.py
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_oficial as R  # noqa: E402,F401  (sets sys.path for the official code and the shim)
import bloodmnist_shim as B  # noqa: E402

import torch  # noqa: E402

EPOCHS, LR, BATCH, SEED = 15, 1e-3, 128, 0


def main():
    sys.path.insert(0, R.OFICIAL)
    from utilities import MNISTClassifier
    torch.manual_seed(SEED)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train, test = B.build_bloodmnist(augmentations=True)
    tl = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True, drop_last=True)
    vl = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)
    net = B.make_classifier_cls(MNISTClassifier)().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    crit = torch.nn.CrossEntropyLoss()
    hist = []
    for ep in range(EPOCHS):
        net.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            crit(net(x), y).backward()
            opt.step()
        net.eval()
        c = n = 0
        with torch.no_grad():
            for x, y in vl:
                c += (net(x.to(dev)).argmax(1).cpu() == y).sum().item()
                n += len(y)
        hist.append(c / n)
        print(f"epoch {ep + 1}: test accuracy {c / n:.4f}", flush=True)
    os.makedirs(os.path.dirname(B.EXTRACTOR_PATH), exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, B.EXTRACTOR_PATH)
    h = hashlib.sha256(open(B.EXTRACTOR_PATH, "rb").read()).hexdigest()
    meta = {"epochs": EPOCHS, "lr": LR, "batch": BATCH, "seed": SEED, "test_acc_por_epoca": hist,
            "sha256": h, "device": str(dev), "torch": torch.__version__}
    json.dump(meta, open(B.EXTRACTOR_PATH + ".json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
