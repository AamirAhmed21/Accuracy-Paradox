import numpy as np
import bentoml
import os
from pydantic import BaseModel


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@bentoml.service(name="accuracy_paradox_service")
class AccuracyParadoxService:
    def __init__(self):
        self.model_tag = os.getenv("BENTO_MODEL_TAG", "accuracy_paradox_model:latest")
        self.model = bentoml.sklearn.load_model(self.model_tag)

    @bentoml.api
    def model_info(self) -> dict:
        return {"model_tag": self.model_tag}

    @bentoml.api
    def predict(self, features: list[float]) -> PredictResponse:
        x = np.array(features, dtype=float).reshape(1, -1)
        pred = int(self.model.predict(x)[0])

        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(x)[0, 1])
        else:
            proba = float(pred)

        return PredictResponse(prediction=pred, probability=proba)