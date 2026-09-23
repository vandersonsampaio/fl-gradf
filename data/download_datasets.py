"""
Download and partition MNIST and CIFAR-10 for federated learning experiments.

Produces two split flavours:
  - IID:     uniform random partition across clients
  - Non-IID: Dirichlet(α) partition — lower α → more heterogeneous

Directory layout after running:
  data/
    raw/
      mnist/        ← full dataset as numpy arrays
      cifar10/      ← full dataset as numpy arrays
    processed/
      mnist/
        iid/
          server_val/   X.npy  y.npy
          client_0/     X_train.npy  y_train.npy  X_test.npy  y_test.npy
          ...
          client_9/
        non_iid/
          server_val/
          client_0/
          ...
          client_9/
      cifar10/
        iid/   (same structure)
        non_iid/

Matching the TARS paper experimental setup:
  - N = 10 clients, 20% Byzantine (f = 2)
  - Non-IID via Dirichlet(α=0.5)
  - Server holds a small clean validation set (D_val, 5% of train)
  - Shared test set split equally across clients

Usage:
  python data/download_datasets.py
  python data/download_datasets.py --n_clients 5 --alpha 0.3
"""

import argparse
import os

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_mnist(root: str):
    """Return (X_train, y_train, X_test, y_test) normalised to [0, 1]."""
    import torchvision
    import torchvision.transforms as T

    transform = T.ToTensor()
    train_ds = torchvision.datasets.MNIST(root, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)

    X_train = train_ds.data.numpy().astype(np.float32) / 255.0   # (60000, 28, 28)
    y_train = train_ds.targets.numpy()
    X_test  = test_ds.data.numpy().astype(np.float32)  / 255.0   # (10000, 28, 28)
    y_test  = test_ds.targets.numpy()

    return X_train, y_train, X_test, y_test


def _download_cifar10(root: str):
    """Return (X_train, y_train, X_test, y_test) normalised to [0, 1]."""
    import torchvision

    train_ds = torchvision.datasets.CIFAR10(root, train=True,  download=True)
    test_ds  = torchvision.datasets.CIFAR10(root, train=False, download=True)

    X_train = train_ds.data.astype(np.float32) / 255.0   # (50000, 32, 32, 3)
    y_train = np.array(train_ds.targets)
    X_test  = test_ds.data.astype(np.float32)  / 255.0   # (10000, 32, 32, 3)
    y_test  = np.array(test_ds.targets)

    return X_train, y_train, X_test, y_test


def _save_raw(dataset_name: str, X_train, y_train, X_test, y_test, raw_root: str):
    out = os.path.join(raw_root, dataset_name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "X_train.npy"), X_train)
    np.save(os.path.join(out, "y_train.npy"), y_train)
    np.save(os.path.join(out, "X_test.npy"),  X_test)
    np.save(os.path.join(out, "y_test.npy"),  y_test)
    print(f"  saved raw {dataset_name}: train={X_train.shape}  test={X_test.shape}")


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def _iid_partition(X_train, y_train, n_clients: int, rng: np.random.Generator):
    """Uniformly random partition of training data across clients."""
    n = len(X_train)
    perm = rng.permutation(n)
    splits = np.array_split(perm, n_clients)
    return [splits[i] for i in range(n_clients)]


def _dirichlet_partition(
    y_train: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
    min_samples: int = 10,
) -> list:
    """
    Non-IID partition via Dirichlet(α) over class labels.

    Each client receives samples according to a Dirichlet-drawn class
    proportion.  Smaller α → more heterogeneous (each client sees fewer
    classes).  α=100 approximates IID.
    """
    n_classes = int(y_train.max()) + 1
    # Indices per class
    class_idx = [np.where(y_train == c)[0] for c in range(n_classes)]
    for ci in class_idx:
        rng.shuffle(ci)

    client_idx: list = [[] for _ in range(n_clients)]

    for c, idx in enumerate(class_idx):
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        # Distribute samples proportionally, ensuring every client gets ≥ 1
        counts = (proportions * len(idx)).astype(int)
        # Fix rounding shortfall
        shortfall = len(idx) - counts.sum()
        if shortfall > 0:
            top = np.argsort(proportions)[-shortfall:]
            counts[top] += 1

        pos = 0
        for ci, cnt in enumerate(counts):
            client_idx[ci].extend(idx[pos : pos + cnt].tolist())
            pos += cnt

    # Ensure minimum samples (resample from largest client if needed)
    for ci in range(n_clients):
        while len(client_idx[ci]) < min_samples:
            donor = max(range(n_clients), key=lambda x: len(client_idx[x]))
            take = rng.choice(client_idx[donor], min_samples, replace=False).tolist()
            client_idx[ci].extend(take)
            for idx in take:
                client_idx[donor].remove(idx)

    return [np.array(client_idx[ci]) for ci in range(n_clients)]


# ---------------------------------------------------------------------------
# Save splits
# ---------------------------------------------------------------------------

def _save_split(
    split_name: str,        # 'iid' or 'non_iid'
    dataset_name: str,
    client_indices: list,   # list of index arrays, one per client
    X_train, y_train,
    X_test, y_test,
    processed_root: str,
    server_val_frac: float = 0.05,
    rng: np.random.Generator = None,
):
    base = os.path.join(processed_root, dataset_name, split_name)
    os.makedirs(base, exist_ok=True)

    # --- Server validation set (carved from the full training pool) ---
    all_train_idx = np.concatenate(client_indices)
    n_val = max(100, int(len(all_train_idx) * server_val_frac))
    val_idx = rng.choice(all_train_idx, n_val, replace=False)
    val_set = set(val_idx.tolist())

    val_dir = os.path.join(base, "server_val")
    os.makedirs(val_dir, exist_ok=True)
    np.save(os.path.join(val_dir, "X.npy"), X_train[val_idx])
    np.save(os.path.join(val_dir, "y.npy"), y_train[val_idx])

    # --- Test set split equally across clients ---
    n_test_clients = len(X_test)
    test_splits = np.array_split(rng.permutation(n_test_clients), len(client_indices))

    # --- Per-client train split ---
    for ci, idx in enumerate(client_indices):
        # Remove server val samples from client training data
        client_train_idx = np.array([i for i in idx if i not in val_set])
        if len(client_train_idx) == 0:
            client_train_idx = idx  # fallback (shouldn't happen)

        client_dir = os.path.join(base, f"client_{ci}")
        os.makedirs(client_dir, exist_ok=True)

        np.save(os.path.join(client_dir, "X_train.npy"), X_train[client_train_idx])
        np.save(os.path.join(client_dir, "y_train.npy"), y_train[client_train_idx])
        np.save(os.path.join(client_dir, "X_test.npy"),  X_test[test_splits[ci]])
        np.save(os.path.join(client_dir, "y_test.npy"),  y_test[test_splits[ci]])

        class_dist = np.bincount(y_train[client_train_idx], minlength=int(y_train.max()) + 1)
        dominant = class_dist.argmax()
        print(
            f"    client_{ci}: {len(client_train_idx):5d} train samples  "
            f"dominant_class={dominant}  dist={class_dist.tolist()}"
        )

    print(f"    server_val: {n_val} samples")


# ---------------------------------------------------------------------------
# Non-IID coefficient
# ---------------------------------------------------------------------------

def _non_iid_coefficient(client_indices: list, y_train: np.ndarray) -> float:
    """
    Measures heterogeneity as 1 - min_rate / max_rate on dominant class rate.
    Value in [0, 1]; higher → more heterogeneous.
    """
    n_classes = int(y_train.max()) + 1
    dominant_rates = []
    for idx in client_indices:
        counts = np.bincount(y_train[idx], minlength=n_classes)
        dominant_rates.append(counts.max() / len(idx))
    return 1.0 - min(dominant_rates) / max(dominant_rates)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_raw(raw_root: str, name: str):
    """Load the raw arrays already saved by a previous run (without
    downloading again) — used by `--extra_alphas` mode."""
    out = os.path.join(raw_root, name)
    for fname in ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"):
        if not os.path.exists(os.path.join(out, fname)):
            raise FileNotFoundError(
                f"{os.path.join(out, fname)} does not exist — run `python data/download_datasets.py` "
                f"(without --extra_alphas) at least once first to download/save the raw data."
            )
    return (
        np.load(os.path.join(out, "X_train.npy")),
        np.load(os.path.join(out, "y_train.npy")),
        np.load(os.path.join(out, "X_test.npy")),
        np.load(os.path.join(out, "y_test.npy")),
    )


def _add_extra_alpha_splits(data_dir: str, alphas, n_clients: int, seed: int):
    """Generate additional non-IID partitions (see
    references/estrategia_B_recriar_headroom.md: severe non-IID, α=0.1/0.05)
    from the raw arrays ALREADY downloaded, WITHOUT touching `iid/` or
    `non_iid/` (α=0.5) — saves each extra α to `non_iid_a{alpha}/`, same
    structure as `_save_split`. Does not re-download or rewrite the default
    split."""
    raw_root  = os.path.join(data_dir, "raw")
    proc_root = os.path.join(data_dir, "processed")
    rng = np.random.default_rng(seed)

    for name in ("mnist", "cifar10"):
        print(f"\n{'='*60}\n  {name.upper()} — extra alphas {alphas}\n{'='*60}")
        X_train, y_train, X_test, y_test = _load_raw(raw_root, name)
        for alpha in alphas:
            split_name = f"non_iid_a{alpha}"
            print(f"[extra] {split_name} (n_clients={n_clients}) ...")
            idx = _dirichlet_partition(y_train, n_clients, alpha, rng)
            coeff = _non_iid_coefficient(idx, y_train)
            _save_split(split_name, name, idx, X_train, y_train, X_test, y_test, proc_root, rng=rng)
            print(f"    non-IID coefficient: {coeff:.4f}  (α={alpha}, lower α → more heterogeneous)")


def main():
    parser = argparse.ArgumentParser(description="Download and partition MNIST & CIFAR-10 for FL")
    parser.add_argument("--n_clients", type=int,   default=10,   help="Number of FL clients")
    parser.add_argument("--alpha",     type=float, default=0.5,  help="Dirichlet α for non-IID (lower = more heterogeneous)")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    parser.add_argument("--data_dir",  type=str,   default=None, help="Base data directory (default: ./data relative to this script)")
    parser.add_argument(
        "--extra_alphas", type=float, nargs="+", default=None,
        help="Generate ADDITIONAL non-IID partitions at these α values (e.g. 0.1 0.05), "
             "saved to non_iid_a{alpha}/, from the raw data already downloaded — does NOT "
             "touch iid/ or non_iid/ (α=0.5). Requires having run without this flag first.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = args.data_dir or script_dir

    if args.extra_alphas:
        _add_extra_alpha_splits(data_dir, args.extra_alphas, args.n_clients, args.seed)
        print(f"\n{'='*60}\nDone (extra alphas only).")
        return

    raw_root   = os.path.join(data_dir, "raw")
    proc_root  = os.path.join(data_dir, "processed")
    torchvision_cache = os.path.join(raw_root, "_torchvision_cache")

    rng = np.random.default_rng(args.seed)

    datasets = {
        "mnist":   _download_mnist,
        "cifar10": _download_cifar10,
    }

    for name, download_fn in datasets.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}")

        # 1. Download
        print(f"[1/4] Downloading {name} ...")
        X_train, y_train, X_test, y_test = download_fn(torchvision_cache)

        # 2. Save raw
        print(f"[2/4] Saving raw arrays ...")
        _save_raw(name, X_train, y_train, X_test, y_test, raw_root)

        # 3. IID partition
        print(f"[3/4] IID partition (n_clients={args.n_clients}) ...")
        iid_idx = _iid_partition(X_train, y_train, args.n_clients, rng)
        coeff = _non_iid_coefficient(iid_idx, y_train)
        _save_split("iid", name, iid_idx, X_train, y_train, X_test, y_test, proc_root, rng=rng)
        print(f"    non-IID coefficient: {coeff:.4f}  (expected ≈ 0.0 for IID)")

        # 4. Non-IID partition
        print(f"[4/4] Non-IID partition (α={args.alpha}) ...")
        noniid_idx = _dirichlet_partition(y_train, args.n_clients, args.alpha, rng)
        coeff = _non_iid_coefficient(noniid_idx, y_train)
        _save_split("non_iid", name, noniid_idx, X_train, y_train, X_test, y_test, proc_root, rng=rng)
        print(f"    non-IID coefficient: {coeff:.4f}  (paper target ≈ 0.7+)")

    print(f"\n{'='*60}")
    print("Done. Directory structure:")
    for root, dirs, files in os.walk(proc_root):
        depth = root.replace(proc_root, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth >= 3:  # don't recurse into client dirs
            dirs.clear()


if __name__ == "__main__":
    main()
