from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

import pandas as pd

from ann import ANN
from packet_preprocessing import PacketPreprocessor
from random_forest import RandomForest


class ModelService:
    def __init__(self, artifacts_dir):
        artifacts_dir = Path(artifacts_dir)
        self.model_name = (artifacts_dir / "selected_model.txt").read_text(
            encoding="utf-8"
        ).strip()
        self.preprocessor = PacketPreprocessor.load(artifacts_dir / "preprocessor.pkl")
        if self.model_name == "ANN":
            self.model = ANN.load(artifacts_dir / "ann_weights.pkl")
        elif self.model_name == "Random Forest":
            self.model = RandomForest.load(artifacts_dir / "random_forest.pkl")
        else:
            raise ValueError(f"Unknown selected model: {self.model_name}")
        self.lock = Lock()

    def predict(self, fields, origin, endpoints=None):
        frame = pd.DataFrame([fields])
        features, _ = self.preprocessor.transform(frame)
        with self.lock:
            score = float(self.model.predict_proba(features).reshape(-1)[0])
        attack = score >= 0.5
        return {
            "id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "origin": origin,
            "experimental": origin in {"pcap", "live"},
            "label": "Attack" if attack else "Normal",
            "attack": attack,
            "score": round(score, 6),
            "severity": "critical" if score >= 0.9 else "high" if attack else "normal",
            "fields": fields,
            "endpoints": endpoints or {},
        }
