"""Streamlit application for predicting individual medical costs billed by health insurance."""

import os
import logging
import socket
import sys
import json
import pandas as pd
import streamlit as st
from src.prediction import ModelPredictor
from src.model import ModelTrainer
from src.visualise import DataVisualizer
from src.results_visualise import ModelMetricsVisualizer

_LOG_PATH = "./logs/"
_MODEL_PATH = "./models/"
_RESULT_PATH = "./results/"
_DATA_PATH = "./data/"

os.makedirs(_LOG_PATH, exist_ok=True)
os.makedirs(_MODEL_PATH, exist_ok=True)
os.makedirs(_RESULT_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s - Host: "
    + socket.gethostname(),
    handlers=[
        logging.FileHandler(os.path.join(_LOG_PATH, "logs.txt")),
        logging.StreamHandler(sys.stdout),
    ],
)


class HealthInsuranceApp:
    """Streamlit application for predicting individual medical costs billed by health insurance."""

    def __init__(self):
        """Initialize the HealthInsuranceApp."""
        self.logger = logging.getLogger(__name__)
        self.model_names = ["RandomForest", "KNeighbors", "Linear", "Ridge", "SVR"]
        self.trainer = ModelTrainer(
            model_names=self.model_names,
            logger=self.logger,
            train_parameters={"test_size": 0.3, "random_state": 42},
            dataset={
                "dataset_path": os.path.join(_DATA_PATH, "data.csv"),
                "target_column": "predict",
            },
        )
        self.predictor = ModelPredictor(
            trainer=self.trainer,
            model_names=self.model_names,
            logger=self.logger,
            model_path=_MODEL_PATH,
        )
        self.visualiser = DataVisualizer(os.path.join(_DATA_PATH, "data.csv"))
        with open(os.path.join(_RESULT_PATH, "results.json"), "r") as f:
            data = json.load(f)
        self.results_visualise = ModelMetricsVisualizer(data)

    def input_data(self):
        """Prompt user to input data."""
        st.sidebar.title("Enter Patient Information")
        input_fields = {
            "age": st.sidebar.number_input("Age", value=30),
            "sex": st.sidebar.radio("Sex", ["Male", "Female"]),
            "bmi": st.sidebar.number_input("BMI", value=25.0),
            "children": st.sidebar.number_input("Number of Children", value=0),
            "smoker": st.sidebar.radio("Smoker", ["Yes", "No"]),
            "region": st.sidebar.selectbox(
                "Region", ["Northeast", "Northwest", "Southeast", "Southwest"], index=0
            ),
        }
        return pd.DataFrame(input_fields, index=[0])

    def predict_medical_costs(self, input_data, model):
        """Predict medical costs based on input data."""
        try:
            prediction = self.predictor.predict(input_data, model)
            return prediction
        except Exception as e:
            self.logger.error("Error in prediction: %s", e, exc_info=True)
            st.error("Error occurred during prediction. Please check your input data.")
            return None

    def train_models(self):
        """Train all models on the training data."""
        try:
            self.logger.info("Training Models...")
            self.trainer.train_save_models()
            st.sidebar.text("Models Trained Successfully!")
        except Exception as e:
            self.logger.error("Error training models: %s", e, exc_info=True)
            st.error("Error occurred during training. Please check the dataset.")
            return None

    def visualize_data(self):
        """Visualize the dataset."""
        return self.visualiser.visualize_data()

    def visualize_results(self):
        """Visualize the results of the models."""
        self.logger.info("Visualizing Model Results...")
        return [
            self.results_visualise.visualize_bar_plots(),
            self.results_visualise.visualize_box_plots(),
        ]

    def data(self):
        """Return the dataset."""
        data = pd.read_csv(os.path.join(_DATA_PATH, "data.csv"))
        return data
