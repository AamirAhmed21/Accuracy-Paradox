from setuptools import setup, find_packages

setup(
    name="Accuracyparadox",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "pandas",
        "numpy",
        "pymongo[srv]==3.12",
        "certifi",
        "scikit-learn",
        "dill",
        "pyaml",
        "mlflow",
        "dagshub",
        "fastapi",
        "uvicorn",
        "bentoml",
        "xgboost",
        "streamlit",
        "imbalanced-learn",
    ],
)