import sys
import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier()
        }

    # -------------------------------
    # Evaluate models
    # -------------------------------
    def evaluate_models(self, X_train, y_train, X_test, y_test):
        try:
            report = {}

            for name, model in self.models.items():
                logging.info(f"Training model: {name}")

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                score = accuracy_score(y_test, y_pred)

                report[name] = score

                logging.info(f"{name} score: {score}")

            return report

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Get best model
    # -------------------------------
    def get_best_model(self, model_report: dict):
        try:
            best_model_name = max(model_report, key=model_report.get)
            best_score = model_report[best_model_name]
            best_model = self.models[best_model_name]

            return best_model_name, best_model, best_score

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Hyperparameter tuning
    # -------------------------------
    def finetune_best_model(self, best_model, best_model_name, X_train, y_train):
        try:
            config = self.utils.read_yaml_file(self.config.model_config_file_path)

            param_grid = config["model_selection"]["model"][best_model_name]["search_param_grid"]

            logging.info(f"Running GridSearch for {best_model_name}")

            grid_search = GridSearchCV(
                estimator=best_model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            logging.info(f"Best params: {grid_search.best_params_}")

            return grid_search.best_estimator_

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------
    # Main training pipeline
    # -------------------------------
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train/test arrays")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Step 1: Evaluate base models
            model_report = self.evaluate_models(X_train, y_train, X_test, y_test)

            # Step 2: Select best model
            best_model_name, best_model, best_score = self.get_best_model(model_report)

            logging.info(f"Best base model: {best_model_name} with score {best_score}")

            # Step 3: Fine-tune best model
            best_model = self.finetune_best_model(
                best_model,
                best_model_name,
                X_train,
                y_train
            )

            # Step 4: Final evaluation
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            final_score = accuracy_score(y_test, y_pred)

            logging.info(f"Final model score: {final_score}")

            if final_score < self.config.expected_accuracy:
                raise Exception("No model meets expected accuracy")

            # Step 5: Save model
            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path=self.config.trained_model_path,
                obj=best_model
            )

            logging.info("Model saved successfully")

            return self.config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)