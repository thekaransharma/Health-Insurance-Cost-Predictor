""" Train and evaluate multiple regression models namely 
Random Forest, K-Nearest Neighbors, Linear Regression, 
Ridge Regression, and Support Vector Regressor. """

import json
import logging
import joblib
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from src.preprocess import DatasetPreprocessor


class ModelTrainer:
    """Train and evaluate multiple regression models."""

    def __init__(
        self,
        model_names=None,
        logger=None,
        train_parameters=None,
        dataset=None,
    ):
        if model_names is None:
            model_names = ["RandomForest", "KNeighbors", "Linear", "Ridge", "SVR"]
        if train_parameters is None:
            train_parameters = {"test_size": 0.3, "random_state": 42}
        if dataset is None:
            dataset = {"dataset_path": "./data/data.csv", "target_column": "predict"}
        self.model_names = model_names
        self.models = {name: None for name in self.model_names}
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = DatasetPreprocessor()
        self.test_size = train_parameters["test_size"]
        self.random_state = train_parameters["random_state"]
        self.dataset_path = dataset["dataset_path"]
        self.target_column = dataset["target_column"]
        self.evaluation_results = None

    def read_and_split_dataset(self):
        """Read dataset and split into train and test sets."""
        try:
            self.logger.info("Reading and Splitting Dataset...")
            dataset = pd.read_csv(self.dataset_path)
            dataset = self.preprocessor.preprocess_data(dataset, to_do="train")
            features = dataset.drop(columns=[self.target_column])
            to_predict = dataset[self.target_column]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                features,
                to_predict,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        except Exception as e:
            self.logger.error(
                f"Error reading and splitting dataset: {e}", exc_info=True
            )
            raise

    def train_models(self):
        """Train all models on the training data."""
        try:
            self.logger.info("Training Models...")
            results = Parallel(n_jobs=-1)(
                delayed(self._train_model)(name) for name in self.model_names
            )
            for name, model in results:
                self.models[name] = model
        except Exception as e:
            self.logger.error(f"Error training models: {e}", exc_info=True)
            raise

    def _train_model(self, name):
        """Train a single model and return the model object."""
        try:
            self.logger.info(f"Training Regressor {name}...")
            if name == "RandomForest":
                model = self.train_model(
                    RandomForestRegressor(), self.get_rf_param_grid()
                )
            elif name == "KNeighbors":
                model = self.train_model(
                    KNeighborsRegressor(), self.get_knn_param_grid()
                )
            elif name == "Linear":
                model = LinearRegression()
            elif name == "Ridge":
                model = self.train_model(Ridge(), self.get_ridge_param_grid())
            elif name == "SVR":
                model = self.train_model(SVR(), self.get_svr_param_grid())
            else:
                self.logger.error(f"Unsupported training model {name}")
                raise ValueError(f"Unsupported training model {name}")
            model.fit(self.x_train, self.y_train)
            self.logger.info(f"Training {name} Model Complete.")
            return name, model
        except Exception as e:
            self.logger.error(f"Error training model {name}: {e}", exc_info=True)
            raise

    def train_model(self, estimator, param_grid):
        """Train a model with grid search cross-validation and return the best estimator."""
        try:
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="neg_mean_squared_error" if estimator != SVR else "r2",
                cv=5,
                n_jobs=-1,
            )
            grid_search.fit(self.x_train, self.y_train)
            return grid_search.best_estimator_
        except Exception as e:
            self.logger.error(f"Error training model: {e}", exc_info=True)
            raise

    @staticmethod
    def get_rf_param_grid():
        """Return the parameter grid for the Random Forest Regressor."""
        return {
            "n_estimators": [10, 50, 100],
            "max_features": [1, 2, 3, "sqrt", "log2", None],
            "max_depth": [3, 5, None],
        }

    @staticmethod
    def get_knn_param_grid():
        """Return the parameter grid for the K-Nearest Neighbors Regressor."""
        return {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"]}

    @staticmethod
    def get_ridge_param_grid():
        """Return the parameter grid for the Ridge Regressor."""
        return {
            "alpha": [0.01, 0.1, 1.0],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg"],
        }

    @staticmethod
    def get_svr_param_grid():
        """Return the parameter grid for the Support Vector Regressor."""
        return {
            "kernel": ["linear", "rbf"],
            "C": [1.0, 2.0, 3.0],
            "epsilon": [0.1, 0.2, 0.3],
        }

    def evaluate_models(self):
        """Evaluate all models on the test data"""
        try:
            self.logger.info("Evaluating Models...")
            results = {}
            for name, model in self.models.items():
                y_pred = model.predict(self.x_test)
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                rmse = mse**0.5
                results[name] = {
                    "Mean Absolute Error (MAE)": mae,
                    "Mean Squared Error (MSE)": mse,
                    "R-squared (R²) Score": r2,
                    "Root Mean Squared Error (RMSE)": rmse,
                }
                self.logger.info(f"{name} results: {results[name]}")
                self.evaluation_results = results
        except Exception as e:
            self.logger.error(f"Error evaluating models: {e}", exc_info=True)
            raise

    def save_results(self):
        """Save the results of the models to JSON files."""
        try:
            self.logger.info("Saving Results...")
            file_name = f"./results/results.json"
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(self.evaluation_results, f)
        except Exception as e:
            self.logger.error(f"Error saving results: {e}", exc_info=True)
            raise

    def save_models(self):
        """Save all models to disk using joblib."""
        try:
            self.logger.info("Saving Models...")
            for name, model in self.models.items():
                self.logger.info(f"Saving {name} Model...")
                joblib.dump(model, f"./models/{name}.pkl")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}", exc_info=True)
            raise

    def train_save_models(self):
        """Train models and save results and models to disk."""
        self.logger.info("Training Models...")
        self.read_and_split_dataset()
        self.train_models()
        self.evaluate_models()
        self.save_results()
        self.save_models()
        self.logger.info("Models trained and saved successfully.")
