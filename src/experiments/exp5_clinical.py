"""
Experiment 5: clinical validation of the XAI explanations.

This module does NOT synthesize clinician responses — human data collection
is out of code's scope. It only delivers the tooling around it:

  1. `export_explanations_for_survey()` — exports a sample of explanations
     from the audit trail in survey format (CSV with empty Likert columns,
     ready for real clinicians to fill in).
  2. `analyze_clinical_survey()` — computes statistics (mean, 95% CI,
     t-test vs. baseline) once real responses have been filled in.

Usage:
    python -m src.experiments.exp5_clinical export
    # ... clinicians fill in the exported CSV ...
    python -m src.experiments.exp5_clinical analyze filled_responses.csv
"""

import argparse
import json
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from src.xai.audit_trail import AuditTrail

LIKERT_DIMENSIONS = ["comprehensibility", "trustworthiness", "actionability"]


def export_explanations_for_survey(
    audit_trail_path: str = "results/audit_trail.jsonl",
    output_path: str = "results/tables/exp5_survey_template.csv",
    n_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Exports a sample of real explanations from the audit trail in survey
    format. Requires that some experiment with GRADFFederatedLearner has
    already run and generated `audit_trail_path` (e.g. exp1_baseline.py)."""
    records = AuditTrail(path=audit_trail_path).load()
    if not records:
        raise ValueError(
            f"No explanations found in {audit_trail_path}. "
            "Run an experiment with GRADFFederatedLearner first (e.g. exp1_baseline.py)."
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n_samples, len(records)), replace=False)

    rows = []
    for i in idx:
        r = records[int(i)]
        rows.append({
            "explanation_id": int(i),
            "hospital_id": r["hospital_id"],
            "decision": r["decision"],
            "predicted_attack_type": r.get("predicted_attack_type"),
            "narrative": r["narrative"],
            "counterfactual_if_accepted": r["counterfactual"]["if_accepted"],
            "counterfactual_if_rejected": r["counterfactual"]["if_rejected"],
            "clinician_id": "",
            "comprehensibility": "",  # Likert 1-5 — filled in by a real clinician
            "trustworthiness": "",    # Likert 1-5
            "actionability": "",      # Likert 1-5
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def analyze_clinical_survey(
    responses_path: str,
    baseline_score: float = 2.1,
    output_path: str = "results/clinical_validation.json",
) -> Dict:
    """Computes statistics from a responses CSV filled in the format of
    `export_explanations_for_survey` (Likert columns with numeric 1-5 values
    filled in by real clinicians)."""
    df = pd.read_csv(responses_path)
    results: Dict = {}

    for dimension in LIKERT_DIMENSIONS:
        scores = pd.to_numeric(df.get(dimension), errors="coerce").dropna()
        if len(scores) < 2:
            results[dimension] = {"error": "insufficient sample (minimum 2 responses)"}
            continue

        mean = float(scores.mean())
        sem = stats.sem(scores)
        ci_low, ci_high = stats.t.interval(0.95, len(scores) - 1, loc=mean, scale=sem)
        t_stat, p_value = stats.ttest_1samp(scores, baseline_score)

        results[dimension] = {
            "mean": mean,
            "std": float(scores.std()),
            "n": int(len(scores)),
            "ci_95": [float(ci_low), float(ci_high)],
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant_improvement_vs_baseline": bool(p_value < 0.05 and mean > baseline_score),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Exports explanations to the survey format")
    export_parser.add_argument("--audit_trail_path", default="results/audit_trail.jsonl")
    export_parser.add_argument("--output_path", default="results/tables/exp5_survey_template.csv")
    export_parser.add_argument("--n_samples", type=int, default=100)
    export_parser.add_argument("--seed", type=int, default=42)

    analyze_parser = subparsers.add_parser("analyze", help="Computes statistics from real responses")
    analyze_parser.add_argument("responses_path")
    analyze_parser.add_argument("--baseline_score", type=float, default=2.1)
    analyze_parser.add_argument("--output_path", default="results/clinical_validation.json")

    args = parser.parse_args()

    if args.command == "export":
        df = export_explanations_for_survey(
            args.audit_trail_path, args.output_path, args.n_samples, args.seed
        )
        print(f"{len(df)} explanations exported to {args.output_path}")
        print("Blocked on human data collection: fill in comprehensibility/trustworthiness/"
              "actionability (Likert 1-5) with real clinician responses before running 'analyze'.")
    else:
        results = analyze_clinical_survey(args.responses_path, args.baseline_score, args.output_path)
        print(json.dumps(results, indent=2, ensure_ascii=False))
