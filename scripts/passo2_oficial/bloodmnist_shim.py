"""
Adaptação mínima do AdaAggRL oficial ao BloodMNIST (B3.0/B3.1;
results/b31_medmnist_oficial/PREREGISTRO.md §2). É um ACRÉSCIMO: o caminho MNIST
oficial continua intacto. Nada em external/AdaAggRL é editado; os monkeypatches só
são aplicados por `install(E, dataset)` quando dataset == "BloodMNIST" (para
"MNIST" a função não faz nada), depois de `import exp_environments as E`.

Peças:
  - BloodMNIST lido do .npz oficial do MedMNIST v2 (Zenodo, record 10519652),
    data/raw/medmnist/bloodmnist.npz; splits oficiais de treino e teste.
    Transformações como no ramo MNIST oficial: ToTensor + Normalize (média/desvio
    por canal do treino) e, no treino, RandomCrop(28, padding=4) + RandomHorizontalFlip.
  - `BloodClassifier`: o MNISTClassifier oficial com conv1 de 3 canais e fc1 de 8
    saídas (mesma arquitetura e inicialização).
  - Extrator: o `torch.load('extract_feature.pt')` do __init__ do ambiente é
    redirecionado para o extrator treinado no B3.0 (EXTRACTOR_PATH).
  - Regime: 8 classes → o código oficial cria int(num_clients / num_class) clientes
    por grupo de classe; com 100 clientes daria 12 × 8 = 96 e os ids sorteados em
    range(100) estourariam. Por isso num_clients = 96 (12 por grupo) e
    subsample_rate = 10/96 (10 clientes por rodada, como no oficial); 20 atacantes.
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
    """Aplica os monkeypatches no módulo exp_environments já importado, SÓ para o
    BloodMNIST. Para qualquer outro dataset (MNIST incluído) não altera nada."""
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
    E.torch.load = load  # vale só neste processo
    return E


def adapt_args(args):
    if args.dataset != "BloodMNIST":
        return args
    args.num_class = NUM_CLASS
    args.num_clients = NUM_CLIENTS
    args.subsample_rate = SUBSAMPLE
    return args
