> English translation of `ADENDO2.md`. The Portuguese original is the frozen record (its hash is in `ADENDO2.sha256`); if the two ever disagree, the original prevails.

# Addendum 2 — B3.1: reset threshold and reward scale

**Date:** 2026-10-05, ~17:40. **Declared timeline:** the B3.1 grid was launched at 17:27 (Addendum 1); this addendum was written ~15 min later, **before any B3.1 data was inspected** (only the first rounds of the first batch existed, not looked at). It **changes nothing** in the design, the criteria or the code; it only declares a property of the official environment observed in B3.0.

## Declaration

**The reset threshold and the reward scale were kept from the official code.**
- Threshold: the episode restarts when the reward < −80 (`exp_environments.py`, l. 271), an official constant, not changed.
- Reward: difference of the **summed** loss over the `testloader` of the whole test set (batch 64, `drop_last=True`): **53 batches** on BloodMNIST (3,392 images evaluated) vs. **156** on MNIST.

**On BloodMNIST, the smaller test set reduces the reward scale by ~3×,** and the reset may not fire during a collapse. This was observed in **B3.0, EB, seed 131**: 7.1% accuracy, below chance for 8 classes (12.5%), i.e. a collapse onto a minority class, with only 14 resets, vs. ~113 in the other two seeds.

## Consequences for the reading

- The effect applies **equally to both conditions** (fixed and td3): same environment, same threshold.
- It **changes the dynamics relative to MNIST**: fewer resets per collapse, and collapses that may persist. It also changes **the reward scale that TD3 sees** (~⅓ of MNIST's). This must be kept in mind when comparing B3.1 with B2.1/B2.2.
- **The threshold was not adjusted:** adjusting it after seeing B3.0 would be a data-driven decision.
- The AUC sensitivity (pre-registration §5) and the reset count per condition still apply and cover this behavior.
