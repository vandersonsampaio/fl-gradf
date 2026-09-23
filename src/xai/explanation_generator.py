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
            return {"if_accepted": "não avaliado", "if_rejected": "não avaliado"}

        if decision == 'REJECTED':
            return {
                'if_accepted': (
                    f"Acurácia cairia de {old_accuracy:.2%} para {new_accuracy:.2%} "
                    f"({new_accuracy - old_accuracy:+.2%})"
                ),
                'if_rejected': f"Acurácia mantida em {old_accuracy:.2%}",
            }
        return {
            'if_accepted': f"Acurácia em {new_accuracy:.2%}",
            'if_rejected': f"Acurácia permaneceria em {old_accuracy:.2%} (update descartado)",
        }

    def _create_narrative(self, hospital_id: str, evidence: Dict, decision: str) -> str:
        """Generates natural-language explanatory text"""
        if decision == 'REJECTED':
            return f"""
            Este update foi rejeitado porque:
            1. Gradientes {evidence['gradient_magnitude']:.1f}x maiores que normal
            2. Causaria degradação de acurácia de {evidence['accuracy_drop']:.1%}
            3. Padrão similar ao ataque {evidence.get('attack_type', 'desconhecido')}
            4. Comportamento inconsistente com histórico

            Recomendação: Investigar {hospital_id} imediatamente.
            """
        return f"""
            Este update foi aceito:
            1. Gradientes dentro de limites normais
            2. Não degrada acurácia do modelo
            3. Padrão consistente com histórico
            4. Confiança: {evidence['confidence']:.1%}
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
