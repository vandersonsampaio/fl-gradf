# Check — FedStrategist's mismatch on MNIST

**Date:** 2026-11-04, before these runs. **Decided by the author.** Diagnostic of `../RESULTS.md`, Sec. 1: FedStrategist differs between two executions in 46 of 210 rows, with no explanation. It decides nothing in the campaign.

**Runs (`scripts/p1/run_fedstrategist_check.sh`):** seed 45, 21 cells, settings of `../../mnist_rq3`, FedStrategist alone (`alone_1`, `alone_2`), and after GRADF in the same process (`after_gradf`). Seed 45 has 7 differing rows of 21.

**Reading, fixed now,** on the final-round column:
- `alone_1` and `alone_2` differ: the arm is not deterministic on its own.
- They match each other, and `after_gradf` differs from them or the earlier executions differ from them: this supports carry-over from systems earlier in the process.
- They match each other and also the earlier executions in every row: no explanation found, and the mismatch is reported as unexplained.

One seed. It can support a cause, not prove it.
