# Addendum 1 — E0b: analysis code and launch

**Date:** 2026-10-30 11:00 -0300, while the E0b runs are in progress and before any of them has finished (0/27 done, `raw/` empty). The plan (`PLAN.md`, hash in `PLAN.sha256`, commit `8c9b28c`) does not change.

## Analysis code

`scripts/p1/analyze_e0b.py` implements PLAN Secs. 1–6 as written:

- Sec. 1: stops (no rule applied) if any of the 27 jobs is missing or has the wrong length, or if any of the 21 jobs that repeat an E0 configuration differs from E0 in any of the first 50 rounds.
- Sec. 2: candidates 20, 30, …, 150; S_α(r) = seed mean of the 5-round trailing mean; both gain conditions below 1.0 p.p. for all three α; R = 150 and "not stabilized" otherwise.
- Sec. 3: τ_CNN = 1.5 × the largest honest norm over all 27 jobs and 180 rounds (block A: all clients; blocks B and C: clients not in `byzantine_ids`).
- Sec. 4: σ_E0^E1 and σ_E0^E3 as the SD (ddof = 1) over seeds 0–2 of the per-seed means over the 2 cells, final-round accuracy at R.
- Sec. 5: C(R, k) as in the plan, with the part D costs from `costs.json` when it exists, otherwise the plan's provisional costs, the level then marked provisional. If the budget truncates R, Secs. 4 and 6 are read at the truncated R.
- Sec. 6: descriptive outputs.

Tested on synthetic data in a scratch directory (not in the repository): complete grid, one regression mismatch (stops), one missing job (stops).

**Note on the provisional cost coefficient (no change to the rule):** with the provisional per-round costs listed in PLAN Sec. 5, the formula gives C(R, 18) ≈ 1.59·R + 20 h (the plan's text rounds it to "≈ 1.65·R + 20", carried over from an earlier estimate with a larger sensitivity term), so level 0 fits up to R ≈ 136 instead of ≈ 130. The decision uses the formula with the measured costs, not either rounded figure.

## Launch

- 2026-10-30 10:49, PGID 468615, `bash scripts/p1/run_e0.sh e0b 6`, W1 paused by its window controller (logged in `fl-fedmed/results/janela_execucao.log`, with the author's authorization to run inside the window).
- **Deviation: 6 processes instead of the plan's 8.** Available RAM was 16.3 GB (other programs were open), and each process needs ~2 GB with a 4 GB reserve. The runs are seeded and use 1 thread each, so the number of concurrent processes should not change any result; the Sec. 1 regression check against E0 (run with 8 processes) tests exactly this. Expected wall time ≈ 4.5–5 h instead of ≈ 4 h.
- Smoke test before the launch: 2 rounds, seed 3, FLTrust root 2,500 under `trim_attack` α 0.05; accuracy and the 2 × 10 update norms were written; the file was moved out of `raw/` at once.

## Code hashes (sha256)

```
ac6fcf0cda8ca52bbba6e86dd3f0fcccfb11578712dc259d0b4bb060972000f1  scripts/p1/analyze_e0b.py
a11d7bf6065b9e3769b4cfe89fd4891485f5418bff790230e84b352a18d81313  scripts/p1/e0_pilot.py
17ee346602e281412501db6b03bc88ae7e2e4d789bd1b65e3c5fb1cff81e7a11  scripts/p1/run_e0.sh
```

`e0_pilot.py` and `run_e0.sh` are unchanged since commit `8c9b28c`.
