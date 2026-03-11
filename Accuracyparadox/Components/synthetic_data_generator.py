import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from Accuracyparadox.logging.logging import logging
from Accuracyparadox.exception.exception import CustomException
import sys
import os 

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
class SyntheticDataGenerator:
    def __init__(self):
        self.output_file = os.path.join(ROOT_DIR, "Data", "raw") ## output directory for synthetic data
        self.file_name = "disease_data.csv" ## output file name
        self.n_samples = 10000
        self.n_features = 20
        self.n_classes = 2
        self.n_redundant = 4
        self.prevalence = 0.01  # 1% positive class
        self.random_state = 42
    def generate_data(self) -> str:
        try:
            logging.info("Generating synthetic data with extreme class imbalance")
            
            ## creating ouytput directory if not exists
            os.makedirs(self.output_file, exist_ok=True)
            
            ## generating synthetic data
            X, y = make_classification(n_samples=self.n_samples,
                                       n_features=self.n_features,
                                       n_classes=self.n_classes,
                                       n_redundant=self.n_redundant,
                                       n_clusters_per_class= 2,
                                       weights=[1 - self.prevalence, self.prevalence],
                                       flip_y=0.001,
                                       class_sep=1.0,
                                       random_state=self.random_state)
                      # create dataframe
            data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(self.n_features)])
            data["target"] = y.astype(int)
            
            ## save to csv 
            
            output_path = os.path.join(self.output_file, self.file_name)
            data.to_csv(output_path, index=False)
            
            ## log class distribution
            class_counts = data["target"].value_counts()
            class_ratio = data['target'].value_counts(normalize=True)
            
            logging.info(f"Data saved to: {output_path}")
            logging.info(f"Total samples : {len(data)}")
            logging.info(f"Class counts  : {class_counts.to_dict()}")
            logging.info(f"Class ratio   : {class_ratio.to_dict()}")
            logging.info(f"Features      : {self.n_features}")

            # print to console
            print(f"\n✅ Synthetic data generated successfully")
            print(f"📁 Saved to     : {output_path}")
            print(f"📊 Total samples: {len(data)}")
            print(f"🔢 Class counts :\n{class_counts}")
            print(f"📉 Class ratio  :\n{class_ratio}")
            print(f"\n⚠️  Accuracy Paradox Demo:")
            print(f"   A model that always predicts 0 (no disease)")
            print(f"   would achieve {class_ratio[0]*100:.2f}% accuracy")
            print(f"   but NEVER detects any disease case!\n")

            return output_path

        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     generator = SyntheticDataGenerator()
#     generator.generate_data()



