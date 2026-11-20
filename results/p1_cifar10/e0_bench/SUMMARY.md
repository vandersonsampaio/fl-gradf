# E0 cost probe (2026-10-29): device and CPU scaling

CIFAR-10, small CNN (34,522 parameters), 10 clients, 2 local epochs, α = 0.5, root 100, seed 0 (outside the confirmatory block). 3 timed rounds after one warm-up round. Machine: i7-13650HX (6 P-cores + 8 E-cores, 20 threads), 31 GB RAM, RTX 3050 Laptop 6 GB. Raw: `*.json` / `*.log` in this folder (`bench_device.py`, `bench_scaling.sh`, `bench_scaling_all.sh`).

## Device (one process), s/round

| Configuration | Trimmed mean | FLTrust | AdaAggRL |
|---|---|---|---|
| CPU, 1 thread, deterministic ops | 11.0 | 10.9 | 18.0 |
| CPU, all threads, deterministic ops | 9.5 | 9.7 | 28.3 |
| GPU, deterministic ops | 17.9 | 18.4 | 23.8 |
| GPU, non-deterministic | 30.2 | 31.1 | 37.0 |

The GPU is slower than the CPU for this model (batch 32, tiny CNN: overhead-bound). GPU runs needed `LD_LIBRARY_PATH` set to the pip CUDA 12 libraries (`libcusolver.so.11` is not found otherwise).

Final accuracy differs slightly across device and thread count (trimmed mean: 0.2564 CPU 1 thread, 0.2560 CPU all threads, 0.2574 GPU), while repeated runs with the same configuration are identical. Device and threads per process must therefore be fixed per paired comparison.

## CPU scaling (k simultaneous processes × t threads), deterministic ops

| System | k × t | s/round per process | Throughput (rounds/s) | vs one process |
|---|---|---|---|---|
| Trimmed mean | 4 × 1 | 14.3 | 0.280 | 3.1× |
| Trimmed mean | 8 × 1 | 21.7 | 0.368 | 4.0× |
| Trimmed mean | 8 × 2 | 21.7 | 0.369 | 4.0× |
| Trimmed mean | 12 × 1 | 26.1 | 0.460 | 5.0× |
| AdaAggRL | 4 × 1 | 23.9 | 0.168 | 3.0× |
| AdaAggRL | 4 × 2 | 25.5 | 0.157 | 2.8× |
| AdaAggRL | 4 × 4 | 26.4 | 0.152 | 2.7× |
| AdaAggRL | 6 × 2 | 37.3 | 0.161 | 2.9× |
| AdaAggRL | 8 × 1 | 34.2 | 0.234 | 4.2× |
| AdaAggRL | 12 × 1 | 43.7 | 0.274 | 4.9× |

- One thread per process is best for every system; extra threads per process add nothing (fixed rules) or lower throughput (AdaAggRL).
- Throughput still grows from 8 to 12 processes (+25% fixed rules, +17% AdaAggRL), with diminishing returns. 16 processes were not tested: peak RSS is about 2 GB per process.
- Honest client update L2 norms (trimmed mean, no attack, rounds 1–3): median 0.73–0.81, range 0.49–1.08, below GRADF's absolute Layer-1 threshold of 5.0 (D5 in `../DEVIATIONS.md`; still to be compared with MNIST's ratio).

## Memory (after the 2026-10-29 OOM)

One process, 50 rounds, FLTrust under trim attack, α = 0.05 (`e0_memdiag.py`):

| `_CNNModel` | RSS round 1 | RSS round 25 | RSS round 50 | Growth | s/round |
|---|---|---|---|---|---|
| new Keras model per use (before) | 1.87 GB | 2.51 GB | 3.09 GB | 26 MB/round | 11.7 |
| one shared Keras model per process (after) | 1.82 GB | 1.84 GB | 1.85 GB | 0.6 MB/round | 9.5 |

The 12-process E0 grid ran out of RAM because of this growth. Fixed in `src/fl/federated_learner.py` (`_shared_cnn`): every `_CNNModel` keeps its own flat parameters and loads them into one compiled Keras model per process; the initial weights (Glorot-uniform kernels, zero biases, as Keras' defaults) are drawn from NumPy's global RNG. The per-round times measured above (device and scaling tables) predate the fix and are about 20% pessimistic. `run_e0.sh` now waits for at least 4 GB of available RAM before starting each job.
