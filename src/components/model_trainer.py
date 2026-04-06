# we train different differnt algothims 
import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import(
AdaBoostRegressor,
GradientBoostingRegressor,
RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Knn Regressor": KNeighborsRegressor()
             }

            params = {
                "Decision Tree": {
                'criterion': ['squared_error','friedman_mse','absolute_error','poisson']
                },
                "Knn Regressor": {
                'n_neighbors': [3, 5, 7, 9],
                'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                "Linear Regression": {},
                "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "XGBRegressor": {
                'learning_rate': [.1, .01, .05, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Regressor": {
                'learning_rate': [.1, .01, 0.5, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
                }
                }
            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            ## get best model name first
            best_model_name = max(model_report, key=lambda x: model_report[x][0])

            ## then get its score
            best_model_score = model_report[best_model_name][0]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing data sets")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicated = best_model.predict(X_test)
            r2  =r2_score(y_test,predicated)
            mae = mean_absolute_error(y_test,predicated)
            mse = mean_squared_error(y_test,predicated)
            return(
                r2,
                mae,
                mse
            )
        except Exception as e:
         raise CustomException(e, sys)