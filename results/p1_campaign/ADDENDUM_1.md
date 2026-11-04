# Addendum 1 — one repeat of a job that loses its connection

**Date:** 2026-11-04, during the campaign, with 28 of 4,128 jobs finished. **Decision:** the author's, after the freeze. No accuracy of seeds 42–51 had been read; the decision does not depend on any result.

**Change to PREREGISTRATION.md, Sec. 8.** A job whose connection to its VM is lost is repeated **once**, on any machine. If it loses the connection again, it is logged as `INFRA dropped`, reported and waits for the author. Any other failed job stays as Sec. 8 states: reported and not rerun without the author's request.

**How it runs (`scripts/p1/dispatch.py`):**
- Once the machine answers again, what is left of the job there is killed, so that it does not run twice.
- If the first attempt had already written its JSON, the repeat finds it and skips, and that JSON stands. A repeat never replaces a result.
- Every repeat is logged as `INFRA <job> <machine> requeued`, and the analysis reports how many there were.

**Why it is safe.** The trigger is the connection, not the job's outcome. Runs on deterministic arms give the same parameters on any machine (`checks/SUMMARY.md`). On arms with GRADF's DQN the repeat is a new draw, but the lost attempt produced no result to compare with, so nothing is selected.

**Unchanged:** the frozen code hash, every setting and every analysis. `dispatch.py` is outside that hash.
