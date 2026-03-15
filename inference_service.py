import numpy as np
import bentoml
from pydantic import BaseModel


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@bentoml.service(name="accuracy_paradox_service")
class AccuracyParadoxService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("accuracy_paradox_model:latest")

    @bentoml.api
    def predict(self, features: list[float]) -> PredictResponse:
        x = np.array(features, dtype=float).reshape(1, -1)
        pred = int(self.model.predict(x)[0])

        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(x)[0, 1])
        else:
            proba = float(pred)

        return PredictResponse(prediction=pred, probability=proba)