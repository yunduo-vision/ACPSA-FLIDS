import numpy as np
import pandas as pd
from ..federated.checkpoint import load_checkpoint
from ..data.preprocessing import FeaturePreprocessor
from ..evaluation.detection_metrics import predict_probabilities


class Predictor:
    def __init__(self, checkpoint, device="auto"):
        self.model, self.state = load_checkpoint(checkpoint, device)
        if self.state["preprocessor"] is None:
            raise ValueError("Controlled-scenario checkpoint has no network-flow preprocessor")
        self.preprocessor = FeaturePreprocessor.from_state(self.state["preprocessor"])

    def predict(self, frame, batch_size=4096):
        x = self.preprocessor.transform_frame(frame)
        probability = predict_probabilities(self.model, x, batch_size=batch_size)
        indices = probability.argmax(axis=1)
        result = pd.DataFrame({"prediction": [self.state["classes"][i] for i in indices],
                               "confidence": probability.max(axis=1), "attack_probability": 1-probability[:, self.state["classes"].index("Benign")]})
        for c, label in enumerate(self.state["classes"]):
            result[f"probability_{label}"] = probability[:, c]
        return result
