# Execution log — B2.3

**Scheduling** changes (not design changes) relative to `PLANO.md`. The plan, the 10 runs, the runner (`run_b23.py`), the steelman configuration and the analysis remain the same, with the hashes in `PLANO.sha256`.

## 2026-09-29 21:01 — 2 runs started early

- **Reason:** B2.1 entered its last seed with only 4 processes, leaving 2 GPU slots idle for ~8 h. Using them shortens B2.3.
- **What changed:**
  - the `run_grid_b23.sh --after-b21` launcher (armed on 2026-09-28 08:19) was stopped before starting any run;
  - new scheduler `scripts/passo2_oficial/run_b23_escalonado.sh` (new file; `run_grid_b23.sh` was not changed): runs **LMP and EB for seed 100 immediately** and, after the B2.1 GRID_END, the **remaining 8 with 4 in parallel**.
- **Why it does not affect the result:** the execution order is not part of the design; each run is determined by the official environment and the seed. Parallelism only changes wall-clock time.
- **Declared side effect:** during the overlap, the last 4 B2.1 runs are a bit slower (GPU shared by 6 processes, as in the rest of the grid).
- **No result was looked at** before this change (B2.1: health and progress only; B2.3: no run had started yet).
