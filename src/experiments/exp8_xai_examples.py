"""
Experiment 8: qualitative XAI illustration — not clinical validation.

Reads the per-run audit trails that `exp2_robustness.py` already saves to
`results/audit_trail/exp2_*.jsonl` (one file per attack_type/
byzantine_fraction/seed combination), extracts one real ACCEPTED decision
example and one REJECTED example (narrative + counterfactual computed by
`XAIExplainer`, not fixed text), and aggregates the decision rate by
predicted attack type across all saved runs.

Prerequisite: run `python -m src.experiments.exp2_robustness` first (that's
where the files read here come from). Actual clinical validation (clinicians
rating these explanations) is deferred to the journal version — see
`src.experiments.exp5_clinical` and the README.
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger
from src.xai.audit_trail import AuditTrail

logger = get_logger(__name__)

_FILENAME_RE = re.compile(r"^exp2_(?P<attack_type>.+)_(?P<frac>[0-9.]+)_(?P<seed>\d+)$")


def find_audit_trail_files(audit_dir: str = "results/audit_trail", prefix: str = "exp2_") -> List[str]:
    return sorted(glob.glob(os.path.join(audit_dir, f"{prefix}*.jsonl")))


def _parse_run_metadata(path: str) -> Dict[str, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _FILENAME_RE.match(stem)
    if not match:
        return {"attack_type": "unknown", "byzantine_fraction": "unknown", "seed": "unknown"}
    return {
        "attack_type": match.group("attack_type"),
        "byzantine_fraction": match.group("frac"),
        "seed": match.group("seed"),
    }


def load_all_records(files: List[str]) -> pd.DataFrame:
    rows = []
    for path in files:
        meta = _parse_run_metadata(path)
        for record in AuditTrail(path=path).load():
            rows.append({
                "run_attack_type": meta["attack_type"],
                "byzantine_fraction": meta["byzantine_fraction"],
                "seed": meta["seed"],
                "hospital_id": record.get("hospital_id"),
                "decision": record.get("decision"),
                "predicted_attack_type": record.get("predicted_attack_type"),
                "confidence": record.get("confidence"),
                "narrative": record.get("narrative"),
                "counterfactual": record.get("counterfactual"),
            })
    return pd.DataFrame(rows)


def pick_examples(files: List[str]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Walks the files and returns the first real ACCEPTED and REJECTED
    examples found (they don't need to come from the same file)."""
    accepted_example, rejected_example = None, None
    for path in files:
        meta = _parse_run_metadata(path)
        for record in AuditTrail(path=path).load():
            if record.get("decision") == "ACCEPTED" and accepted_example is None:
                accepted_example = {**record, **meta}
            if record.get("decision") == "REJECTED" and rejected_example is None:
                rejected_example = {**record, **meta}
        if accepted_example and rejected_example:
            break
    return accepted_example, rejected_example


def _format_example(title: str, example: Optional[Dict]) -> str:
    if example is None:
        return f"## {title}\n\n(no example found in the available audit trails)\n"
    return f"""## {title}

- Run: attack_type={example['attack_type']}, byzantine_fraction={example['byzantine_fraction']}, seed={example['seed']}
- Hospital: `{example['hospital_id']}`
- Decision: **{example['decision']}**
- Predicted attack type: `{example.get('predicted_attack_type')}`
- Classifier confidence: {example.get('confidence', 0):.2%}

**Narrative:**
```
{(example.get('narrative') or '').strip()}
```

**Counterfactual:**
```
{example.get('counterfactual')}
```
"""


def build_markdown(accepted_example: Optional[Dict], rejected_example: Optional[Dict], decision_rate_table: pd.DataFrame) -> str:
    table_md = decision_rate_table.to_markdown(index=False) if not decision_rate_table.empty else "(no data)"
    return f"""# Qualitative XAI explanation examples (GRADF)

Generated from `exp2_robustness.py`'s real audit trails
(`results/audit_trail/exp2_*.jsonl`) — narrative and counterfactual computed
by `XAIExplainer`, not fixed text. **This is not clinical validation**: no
human has rated these explanations; that is deferred to the journal version
(see the README's scope section).

{_format_example("Example — ACCEPTED", accepted_example)}

{_format_example("Example — REJECTED", rejected_example)}

## Decision rate by predicted attack type (aggregated across all available exp2 runs)

{table_md}
"""


def run_xai_examples(audit_dir: str = "results/audit_trail") -> Tuple[str, pd.DataFrame]:
    files = find_audit_trail_files(audit_dir)
    if not files:
        raise FileNotFoundError(
            f"No audit trail found in {audit_dir}/exp2_*.jsonl — "
            "run `python -m src.experiments.exp2_robustness` first."
        )
    logger.info("Reading %d audit trail(s) from %s...", len(files), audit_dir)

    all_records = load_all_records(files)
    decision_rate_table = (
        all_records.groupby("predicted_attack_type")["decision"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reset_index()
    )

    accepted_example, rejected_example = pick_examples(files)
    markdown = build_markdown(accepted_example, rejected_example, decision_rate_table)
    return markdown, decision_rate_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_dir", default="results/audit_trail")
    parser.add_argument("--output", default="results/tables/xai_examples.md")
    args = parser.parse_args()

    markdown, decision_rate_table = run_xai_examples(args.audit_dir)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(markdown)
    print(f"\nSaved to {args.output}")
