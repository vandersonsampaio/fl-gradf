"""
Minimal adaptation of the official AdaAggRL to BloodMNIST (B3.0/B3.1;
results/b31_medmnist_oficial/PREREGISTRO.md §2). It is an ADDITION: the official MNIST
path stays intact. Nothing in external/AdaAggRL is edited; the monkeypatches are only
applied by `install(E, dataset)` when dataset == "BloodMNIST" (for
"MNIST" the function does nothing), after `import exp_environments as E`.

Pieces:
  - BloodMNIST read from the official MedMNIST v2 .npz (Zenodo, record 10519652),
    expected at data/raw/medmnist/bloodmnist.npz (downloaded, not versioned); official
    train and test splits. Transforms as in the official MNIST branch: ToTensor + Normalize
    (per-channel mean/std of the training split) and, for training, RandomCrop(28, padding=4) + RandomHorizontalFlip.
  - `BloodClassifier`: the official MNISTClassifier with a 3-channel conv1 and an 8-output
    fc1 (same architecture and initialization).
  - Extractor: the environment __init__'s `torch.load('extract_feature.pt')` is
    redirected to the extractor trained in B3.0 (EXTRACTOR_PATH).
  - Regime: 8 classes → the official code creates int(num_clients / num_class) clients
    per class group; with 100 clients this gives 12 × 8 = 96, and ids sampled from
    range(100) would overflow. Hence num_clients = 96 (12 per group) and
    subsample_rate = 10/96 (10 clients per round, as in the official code); 20 attackers.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NPZ = os.path.join(REPO, "data", "raw", "medmnist", "bloodmnist.npz")
EXTRACTOR_PATH = os.path.join(REPO, "data", "models", "extract_feature_bloodmnist.pt")
NUM_CLASS = 8
NUM_CLIENTS = 96
SUBSAMPLE = 10 / 96


class BloodMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        self.images = images  # (N, 28, 28, 3) uint8
        self.labels = labels.astype(np.int64).ravel()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])


def build_bloodmnist(augmentations=True, normalize=True):
    d = np.load(NPZ)
    tr = d["train_images"].astype(np.float32) / 255.0
    mean = tuple(float(x) for x in tr.mean(axis=(0, 1, 2)))
    std = tuple(float(x) for x in tr.std(axis=(0, 1, 2)))
    base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x)])
    train_tf = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(28, padding=4),
                                   transforms.RandomHorizontalFlip(), base]) if augmentations else \
        transforms.Compose([transforms.ToPILImage(), base])
    test_tf = transforms.Compose([transforms.ToPILImage(), base])
    return (BloodMNISTDataset(d["train_images"], d["train_labels"], train_tf),
            BloodMNISTDataset(d["test_images"], d["test_labels"], test_tf))


def make_classifier_cls(MNISTClassifier):
    class BloodClassifier(MNISTClassifier):
        def __init__(self, nb_filters=64, activation="relu"):
            super().__init__(nb_filters, activation)
            self.conv1 = nn.Conv2d(3, nb_filters, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3))
            nn.init.xavier_uniform_(self.conv1.weight)
            self.fc1 = nn.Linear(nb_filters * 2, NUM_CLASS)
            nn.init.xavier_uniform_(self.fc1.weight)
    return BloodClassifier


def install(E, dataset):
    """Applies the monkeypatches to the already imported exp_environments module, ONLY for
    BloodMNIST. For any other dataset (MNIST included) it changes nothing."""
    if dataset != "BloodMNIST":
        return E
    orig_cd = E.construct_dataloaders

    def construct_dataloaders(dataset, *a, **k):
        if dataset == "BloodMNIST":
            return build_bloodmnist(k.get("augmentations", True), k.get("normalize", True))
        return orig_cd(dataset, *a, **k)
    E.construct_dataloaders = construct_dataloaders

    E.MNISTClassifier = make_classifier_cls(E.MNISTClassifier)

    orig_load = E.torch.load

    def load(path, *a, **k):
        if path == "extract_feature.pt":
            k.setdefault("map_location", "cpu")
            return orig_load(EXTRACTOR_PATH, *a, **k)
        return orig_load(path, *a, **k)
    E.torch.load = load  # applies to this process only
    return E


def adapt_args(args):
    if args.dataset != "BloodMNIST":
        return args
    args.num_class = NUM_CLASS
    args.num_clients = NUM_CLIENTS
    args.subsample_rate = SUBSAMPLE
    return args
