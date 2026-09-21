"""Persistence for GRADF's decision audit trail (JSON Lines)."""

import json
import os
from typing import Dict, List, Optional


class AuditTrail:
    def __init__(self, path: str = "results/audit_trail.jsonl") -> None:
        self.path = path

    def add(self, explanation: Dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(explanation, default=str) + "\n")

    def load(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)
