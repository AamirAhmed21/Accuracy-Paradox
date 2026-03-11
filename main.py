from Accuracyparadox.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    artifact = TrainingPipeline().run_pipeline()
    print("\nPipeline completed.")
    print(f"Raw   : {artifact.raw_data_path}")
    print(f"Train : {artifact.train_file_path}")
    print(f"Test  : {artifact.test_file_path}")