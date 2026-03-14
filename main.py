from Accuracyparadox.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    ing_art, val_art, trans_art, model_art = TrainingPipeline().run_pipeline()

    print("\n✅ Pipeline completed")
    print(f"Train CSV: {ing_art.train_file_path}")
    print(f"Validation report: {val_art.validation_report_file_path}")
    print(f"Transformed train: {trans_art.transformed_train_file_path}")
    print(f"Best model: {model_art.best_model_name}")
    print(f"Metrics file: {model_art.model_metrics_file_path}")