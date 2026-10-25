# Addendum 2 — W3: validity gate (PLANO §4)

**Date:** 2026-10-24, before any run on the confirmation seeds 190–199. Gate run on the development seeds 125–129 (log `gate1.log`, raw files in `raw_gate/`). Code unchanged since addendum 1.

**Chosen rung: 1.** The gate passed at the first rung, so rungs 2 and 3 were not run.

## Gate output (`gate_rung1.txt`)

```
W3 gate, rung 1 {'c': 2.0, 'step': 0.1}: 60/60 runs
  alpha 0.05: fixed_center acc ipm2 − adaptive = +4.10 p.p. (≥ 2.0); acceptance 53.40% (≥ 30%) -> PASS | fedavg acc: adaptive 83.94, ipm2 83.54
  alpha 0.1: fixed_center acc ipm2 − adaptive = +32.56 p.p. (≥ 2.0); acceptance 63.47% (≥ 30%) -> PASS | fedavg acc: adaptive 74.48, ipm2 74.30
  alpha 0.5: fixed_center acc ipm2 − adaptive = -0.01 p.p. (≥ 2.0); acceptance 66.40% (≥ 30%) -> FAIL | fedavg acc: adaptive 90.83, ipm2 90.81
VERDICT rung 1: 2/3 cells -> VALID (use this rung)
```

**Note:** the α = 0.5 cell fails the gate (the adaptive attack does not lower `fixed_center` accuracy below ipm2's there). The plan's rule is ≥ 2/3 cells, so rung 1 is VALID. The confirmation still runs all three α cells as planned; the α = 0.5 cell should be read knowing the adaptive attack was not stronger than ipm2 in development.

## Next step

Confirmation grid on seeds 190–199 with rung 1 (120 runs): `bash scripts/run_grid_w3.sh confirm 1 <n_procs>`, then `analisar --rung 1` only once the grid is complete.

## Hash (sha256)

```
ce07e580c789d7054fb37a43d5e7436d115b8e07ed71d9859c225418d831add2  gate_rung1.txt
```
