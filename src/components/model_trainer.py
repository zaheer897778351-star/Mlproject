import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from src.logger import logging
from src.utils import saved_object,evaluate_models
from src.exception import Custom_Execption


@dataclass
class ModelTrainerConfig:
    traine_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random forest":RandomForestRegressor(),
                "Decision tree":DecisionTreeRegressor(),
                "Gradient boost":GradientBoostingRegressor(),
                "linear Regression":LinearRegression(),
                "kNN":KNeighborsRegressor(),
                "catboost":CatBoostRegressor(),
                "ada boost":AdaBoostRegressor(),
                "s v r": SVR(kernel='linear')
            }
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                raise Custom_Execption("No best model found")
            logging.info("best model found in model_training")
            saved_object(
                file_path=self.model_trainer_config.traine_model_file_path,
                 obj = best_model)
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            

            return r2_square
        
        except Exception as e:
            raise Custom_Execption(e,sys)