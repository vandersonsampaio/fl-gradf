# Addendum 2 — W2: validity gate (PLANO §4)

**Date:** 2026-10-24, before any run on the confirmation seeds 180–189. Gate run on the development seeds 92–96 (log `gate1.log`, raw files in `raw_gate/`). Code unchanged since addendum 1.

**Chosen rung: 1.** The gate passed at the first rung, so rungs 2 and 3 were not run.

## Gate output (`gate_rung1.txt`)

```
W2 gate, rung 1 {'trigger_size': 4, 'poison_fraction': 0.5}: 30/30 runs
  alpha 0.05: ASR fedavg − clean = +98.00 p.p. (≥ 50.0); clean acc drop = +0.11 p.p. (≤ 5.0) -> PASS
  alpha 0.1: ASR fedavg − clean = +98.94 p.p. (≥ 50.0); clean acc drop = +0.17 p.p. (≤ 5.0) -> PASS
  alpha 0.5: ASR fedavg − clean = +99.44 p.p. (≥ 50.0); clean acc drop = -0.10 p.p. (≤ 5.0) -> PASS
VERDICT rung 1: VALID (use this rung)
```

## Next step

Confirmation grid on seeds 180–189 with rung 1 (180 runs): `bash scripts/run_grid_w2.sh confirm 1 <n_procs>`, then `analisar --rung 1` only once the grid is complete.

## Hash (sha256)

```
dbf17f7466f7f6688e99ed5c813133fd5dca7a7481d7bebea85ce0f8b687783f  gate_rung1.txt
```
