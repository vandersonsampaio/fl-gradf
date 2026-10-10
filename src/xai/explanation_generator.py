from datetime import datetime
from typing import Dict, Optional

from src.xai.audit_trail import AuditTrail


class XAIExplainer:
    def __init__(self, audit_trail: Optional[AuditTrail] = None):
        self.audit_trail = audit_trail or AuditTrail()

    def generate_explanation(
        self,
        hospital_id: str,
        decision: str,
        evidence: Dict,
        old_accuracy: Optional[float] = None,
        new_accuracy: Optional[float] = None,
    ) -> Dict:
        """Generates a structured explanation for an ACCEPT/REJECT decision and
        persists it to the audit trail.

        `old_accuracy`/`new_accuracy` (validation accuracy before/after
        applying this hospital's update, in isolation) feed the
        counterfactual — when absent, the counterfactual is reported as not
        evaluated instead of a fixed placeholder text.
        """
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'hospital_id': hospital_id,
            'decision': decision,  # ACCEPTED or REJECTED
            'confidence': evidence['confidence'],
            'predicted_attack_type': evidence.get('attack_type'),
            'evidence': {
                'gradient_score': evidence['gradient_magnitude'],
                'accuracy_impact': evidence['accuracy_drop'],
                'clustering_distance': evidence['similarity'],
                'temporal_consistency': evidence['temporal'],
            },
            'narrative': self._create_narrative(hospital_id, evidence, decision),
            'counterfactual': self._compute_counterfactual(decision, old_accuracy, new_accuracy),
            'recommendation': 'Reject and investigate' if decision == 'REJECTED' else 'Accept',
        }

        self.audit_trail.add(explanation)

        return explanation

    def _compute_counterfactual(
        self, decision: str, old_accuracy: Optional[float], new_accuracy: Optional[float]
    ) -> Dict[str, str]:
        if old_accuracy is None or new_accuracy is None:
            return {"if_accepted": "not evaluated", "if_rejected": "not evaluated"}

        if decision == 'REJECTED':
            return {
                'if_accepted': (
                    f"Accuracy would drop from {old_accuracy:.2%} to {new_accuracy:.2%} "
                    f"({new_accuracy - old_accuracy:+.2%})"
                ),
                'if_rejected': f"Accuracy kept at {old_accuracy:.2%}",
            }
        return {
            'if_accepted': f"Accuracy at {new_accuracy:.2%}",
            'if_rejected': f"Accuracy would stay at {old_accuracy:.2%} (update discarded)",
        }

    def _create_narrative(self, hospital_id: str, evidence: Dict, decision: str) -> str:
        """Generates natural-language explanatory text"""
        if decision == 'REJECTED':
            return f"""
            This update was rejected because:
            1. Gradients {evidence['gradient_magnitude']:.1f}x larger than normal
            2. It would degrade accuracy by {evidence['accuracy_drop']:.1%}
            3. Pattern similar to the {evidence.get('attack_type', 'unknown')} attack
            4. Behavior inconsistent with history

            Recommendation: Investigate {hospital_id} immediately.
            """
        return f"""
            This update was accepted:
            1. Gradients within normal limits
            2. Does not degrade model accuracy
            3. Pattern consistent with history
            4. Confidence: {evidence['confidence']:.1%}
            """


if __name__ == '__main__':
    explainer = XAIExplainer(audit_trail=AuditTrail(path="results/xai_demo.jsonl"))
    explainer.audit_trail.clear()

    explanation = explainer.generate_explanation(
        hospital_id='hospital_c',
        decision='REJECTED',
        evidence={
            'gradient_magnitude': 48.5,
            'accuracy_drop': 0.18,
            'similarity': 0.12,
            'temporal': 44.0,
            'confidence': 0.97,
            'attack_type': 'MODEL_POISONING',
        },
        old_accuracy=0.91,
        new_accuracy=0.73,
    )

    print(explanation['narrative'])
    print(explanation['counterfactual'])
