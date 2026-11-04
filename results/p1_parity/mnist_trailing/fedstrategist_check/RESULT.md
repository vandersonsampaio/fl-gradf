# Result — FedStrategist's mismatch on MNIST

**Reading of `PLAN.md`: the arm is not deterministic on its own.** `alone_1` and `alone_2` differ in 8 of 21 final-round rows, by up to 6.34 p.p. So carry-over between systems is not needed to explain the mismatch.

| pair, seed 45, final round | rows that differ, of 21 | largest difference, p.p. |
|---|---|---|
| `alone_1` − `alone_2` | 8 | 6.34 |
| `alone_1` − `after_gradf` | 8 | 5.60 |
| `alone_2` − `after_gradf` | 5 | 6.34 |
| earlier execution 1 − execution 2 | 7 | 5.65 |

Running after GRADF differs from running alone no more than two runs alone differ from each other. One seed: this can support a cause, not prove it.

## Likely cause, found while reading the code

This goes beyond the reading fixed in `PLAN.md`.

- The arm that differs is variant b. Variant b feeds the bandit with `RLAttackClassifier`, a Keras network. Variant a uses its own diagnostic and has no network, and it matched in 210 of 210 rows.
- `_seeded` in `exp10_selector_comparison.py` seeds TensorFlow only when `model_type == "cnn"`. On the logistic model, the classifier is unseeded.
- Measured in separate processes, with `np.random.seed(7)` only: two different hashes of the classifier's initial weights. Adding `tf.keras.utils.set_random_seed(7)`: two equal hashes.

Only the initial weights were tested. A rerun of the arm with TensorFlow seeded was not made, so the cause is supported, not established. GRADF receives the same classifier, so its differences may not all come from the DQN; the two sources were not separated.

## Variant labels are inverted in the MNIST analyses

In the code, variant a is FedStrategist's own diagnostic and variant b uses GRADF's detector (`FedStrategistGridLearner`). The following call variant a "with GRADF's detector" and variant b plain "FedStrategist":

- `results/p1_parity/PLAN.md`, item 2, and `scripts/p1/run_parity.sh`;
- `scripts/p1/audit_mnist.py`, `run_mnist_inference.py`, `analyze_parity.py`, and `analyze_mnist_trailing.py` until this was corrected;
- `results/p1_parity/RESULTS.md`, Sec. 4 sensitivity line, and `../RESULTS.md` until this was corrected.

The CIFAR-10 campaign is labelled as in the code: `fedstrategist` is variant a, `fedstrategist_detector` is variant b. So the row called "FedStrategist" is variant b on MNIST and variant a on CIFAR-10. **Author's decision, 2026-11-04:** variant a is the primary arm on MNIST, in `../RESULTS.md` and `scripts/p1/analyze_mnist_trailing.py`.

**First submission:** Table 4 reports FedStrat. − Random as −0.044. The published runs give −0.0438 for variant b and −0.0407 for variant a, so that row is variant b, GRADF's detector. Table 3's FedStrat. column also matches variant b on the 7 rows of α = 0.5, and differs from variant a on 6 of them. The other α and the remaining tables and figures were not compared.
