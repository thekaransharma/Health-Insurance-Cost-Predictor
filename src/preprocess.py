"""Preprocessing of the Dataset."""

import logging
from sklearn.impute import SimpleImputer
import pandas as pd


class DatasetPreprocessor:
    """Preprocess the dataset for training."""

    def __init__(self, strategy="mean", logger=None):
        """Initialize the DatasetPreprocessor."""
        self.strategy = strategy
        self._sex_mapping = {"female": 0, "male": 1}
        self._smoker_mapping = {"no": 0, "yes": 1}
        self._region_mapping = {
            "northeast": 0,
            "northwest": 1,
            "southeast": 2,
            "southwest": 3,
        }
        self.logger = logger or logging.getLogger(__name__)

    def encode_categorical_data(self, dataset):
        """Encode categorical columns in the dataset."""
        self.logger.info("Encoding categorical data... [Sex, Smoker, Region]")
        dataset["sex"] = dataset["sex"].map(self._sex_mapping)
        dataset["smoker"] = dataset["smoker"].map(self._smoker_mapping)
        dataset["region"] = dataset["region"].map(self._region_mapping)
        return dataset

    def handle_missing_data(self, dataset):
        """Handle missing data in the dataset."""
        self.logger.info("Handling missing data...")
        imputer = SimpleImputer(strategy=self.strategy)
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
        return dataset

    def preprocess_data(self, dataset, to_do="train"):
        """
        Preprocess the input data.

        Args:
            dataset (pd.DataFrame): Input dataset.
            to_do (str): Specify whether to preprocess for training or prediction.

        Returns:
            pd.DataFrame: Preprocessed dataset.
        """
        if to_do == "predict":
            self.logger.info("Preprocessing data for prediction...")
            return self.encode_categorical_data(dataset)
        
        self.logger.info("Preprocessing data for training...")
        dataset = dataset.drop_duplicates()
        dataset = self.encode_categorical_data(dataset)
        return self.handle_missing_data(dataset)
