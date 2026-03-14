import os
import sys
import json
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from numpy.ma import log
import bentoml

from Accuracyparadox.exception.exception import CustomException
from Accuracyparadox.logging.logging import logging
from Accuracyparadox.entity.config_entity import ModelTrainerConfig
from Accuracyparadox.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)

MLFLOW_EXPERIMENT_NAME = 'AccuracyParadox_Experiment'

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
            ## Set MLflow experiment
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        except Exception as e:
            raise CustomException(e, sys) from e
    @staticmethod
    def _split_xy(arr: np.ndarray):
        X = arr[:, :-1]
        y = arr[:, -1]
        return X, y
    @staticmethod
    def _evaluate(model, X_test, y_test) -> dict:
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

        return metrics
    def _get_models(self) -> dict:
        return {
            "DummyClassifier": DummyClassifier(strategy="most_frequent"),
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                random_state=42,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                scale_pos_weight=99,
                eval_metric="logloss",
                random_state=42,
            ),
        }
    def _log_to_mlflow(self, name: str, model, metrics: dict, params: dict = {}):
        """Log each model run to MLflow"""
        with mlflow.start_run(run_name=name, nested=True):

            # log parameters
            mlflow.log_param("model_name", name)
            for key, val in params.items():
                mlflow.log_param(key, val)

            # log metrics
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("balanced_accuracy", metrics["balanced_accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1_score", metrics["f1_score"])

            if metrics["roc_auc"] is not None:
                mlflow.log_metric("roc_auc", metrics["roc_auc"])
            if metrics["pr_auc"] is not None:
                mlflow.log_metric("pr_auc", metrics["pr_auc"])

            # log model
            mlflow.sklearn.log_model(model, artifact_path="model")

            logging.info(f"MLflow logged: {name}")
    def _save_to_bentoml(self, best_model, best_model_name: str, best_f1: float):
        """Save best model to BentoML model store"""
        saved_model = bentoml.sklearn.save_model(
            "accuracy_paradox_model",
            best_model,
            signatures={
                "predict": {"batchable": True},
                "predict_proba": {"batchable": True},
            },
            metadata={
                "best_model_name": best_model_name,
                "f1_score": best_f1,
                "project": "Accuracy Paradox",
                "description": (
                    "Model trained to demonstrate that high accuracy "
                    "does not mean good model performance on imbalanced data"
                ),
            },
        )
        logging.info(f"BentoML model saved: {saved_model.tag}")
        return saved_model.tag
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("─────────────────────────────────────────")
            logging.info("Starting Model Training")
            logging.info("─────────────────────────────────────────")
            
            # Load transformed data
            train_arr = np.load(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = np.load(
                self.data_transformation_artifact.transformed_test_file_path
            )
            X_train, y_train = self._split_xy(train_arr)
            X_test, y_test = self._split_xy(test_arr)
            
            logging.info(f"Train shape: {X_train.shape}")
            logging.info(f"Test shape : {X_test.shape}")
            
            models = self._get_models()
            all_metrics = {}
            best_model_name = None
            best_model = None
            best_f1 = -1.0
            
            # start parent mlflow run 
            with mlflow.start_run(run_name="AccuracyParadox_Training"):

                for name, model in models.items():
                    logging.info(f"Training: {name}")
                    model.fit(X_train, y_train)
                    metrics = self._evaluate(model, X_test, y_test)
                    all_metrics[name] = metrics
                    
                    # log to MLflow
                    self._log_to_mlflow(name, model, metrics)
                    logging.info(
                        f"{name} → "
                        f"Accuracy: {metrics['accuracy']:.4f} | "
                        f"Precision: {metrics['precision']:.4f} | "
                        f"Recall: {metrics['recall']:.4f} | "
                        f"F1: {metrics['f1_score']:.4f} | "
                        f"ROC-AUC: {metrics['roc_auc']} | "
                        f"PR-AUC: {metrics['pr_auc']}"
                    )
                    if metrics["f1_score"] > best_f1:
                        best_f1 = metrics["f1_score"]
                        best_model_name = name
                        best_model = model
                      # log best model summary to parent run
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_f1", best_f1)
              # save best model locally
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)
            logging.info(f"Best model saved locally: {self.model_trainer_config.trained_model_file_path}")
             # save best model to bentoml
            bento_tag = self._save_to_bentoml(best_model, best_model_name, best_f1)
            
             # save full metrics report
            report = {
                "best_model_name": best_model_name,
                "best_model_f1": best_f1,
                "bento_model_tag": str(bento_tag),
                "accuracy_paradox_demo": {
                    "baseline_accuracy": all_metrics["DummyClassifier"]["accuracy"],
                    "baseline_recall": all_metrics["DummyClassifier"]["recall"],
                    "baseline_f1": all_metrics["DummyClassifier"]["f1_score"],
                    "message": (
                        f"DummyClassifier achieves "
                        f"{all_metrics['DummyClassifier']['accuracy']*100:.2f}% accuracy "
                        f"but detects ZERO minority cases (recall=0). "
                        f"This is the Accuracy Paradox!"
                    ),
                },
                "all_model_metrics": all_metrics,
            }
            
            with open(self.model_trainer_config.model_metrics_file_path, "w") as f:
                json.dump(report, f, indent=4)

            logging.info(f"Metrics report: {self.model_trainer_config.model_metrics_file_path}")
            logging.info(f"Best model    : {best_model_name}")
            logging.info(f"Best F1       : {best_f1:.4f}")
            logging.info("─────────────────────────────────────────")
            logging.info("Model Training Completed")
            logging.info("─────────────────────────────────────────")
            
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                model_metrics_file_path=self.model_trainer_config.model_metrics_file_path,
                best_model_name=best_model_name,
                best_model_f1=float(best_f1),
                mlflow_run_id="",
                bentoml_model_tag=str(bento_tag),
            )

        except Exception as e:
            raise CustomException(e, sys) from e