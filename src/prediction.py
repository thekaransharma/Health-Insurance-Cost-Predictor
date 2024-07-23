"""Prediction module to load saved models and make predictions using them."""

import os
import joblib
from src.preprocess import DatasetPreprocessor


class ModelPredictor:
    """
    ModelPredictor class to load saved models and make predictions using them.
    """

    def __init__(
        self, trainer=None, model_names=None, logger=None, model_path="./models/"
    ):
        """
        Initialize ModelPredictor.

        Args:
            trainer (object): ModelTrainer instance.
            model_names (list): List of model names.
            logger (object): Logger instance.
            model_path (str): Path to the directory containing saved models.
        """
        self.loaded_models = None
        self.model_names = model_names
        self.model_path = model_path
        self.preprocessor = DatasetPreprocessor()
        self.logger = logger
        self.trainer = trainer

    def load_models(self, model_name="Linear"):
        """
        Load saved models from the specified directory.

        Args:
            model_name (str): Name of the model to load.
        """
        self.logger.info("Loading models...")
        model_file = os.path.join(self.model_path, f"{model_name}.pkl")
        if not os.path.exists(model_file):
            self.logger.error(f"Model file '{model_file}' not found.")
            self.trainer.train_save_models()
        try:
            self.loaded_models = joblib.load(model_file)
            self.logger.info(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading model '{model_name}': {str(e)}")

    def preprocess_data(self, data):
        """
        Preprocess the input data using the DatasetPreprocessor class.

        Args:
            data (pandas.DataFrame): Input data.

        Returns:
            pandas.DataFrame: Preprocessed data.
        """
        self.logger.info("Preprocessing data...")
        data = self.preprocessor.preprocess_data(data, to_do="predict")
        return data

    def predict(self, data, model):
        """
        Predict target values using the loaded models for the input data.

        Args:
            data (pandas.DataFrame): Input data.
            model (str): Name of the model to use for prediction.

        Returns:
            float: Predicted target value.
        """
        self.logger.info("Starting prediction...")
        data = self.preprocess_data(data)
        self.load_models(model)
        predictions = self.loaded_models.predict(data)[0]
        self.logger.info(f"Predictions for model '{model}': {predictions}")
        return predictions
