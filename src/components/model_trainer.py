import os,sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split train and test input data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            ) 

            models = {
                "CatBoost": CatBoostRegressor(allow_const_label=True),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-neighbours": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }
            params={                
                "CatBoost": {
                    'depth': [6,8,10], 
                    "learning_rate": [0.1,0.05,0.01],
                    "iterations": [30,50,100],
                },
                "Random Forest":{
                    'criterion' : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    'n_estimators':[8,16,32,64,128,256],
                    'max_depth': [2,3,4,5],
                    # "max_features": ["sqrt", "log2", None]
                },
                "Decision Tree": {
                    'criterion' : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "splitter": ["best", "random"],
                    # "max_features": ["sqrt", "log2", ],
                    'max_depth': [2,3,4,5],
                },
                'Gradient Boosting': {
                    "loss": ["squared_error", "absolute_error", "huber", "quantile"],
                    "learning_rate": [0.1,0.05,0.01,0.001],
                    'n_estimators':[8,16,32,64,128,256],
                    "subsample": [0.6,0.7,0.75,0.8,0.85,0.9,1],
                    "max_features": ["sqrt", "log2", ],
                    # 'criterion' : ["squared_error", "friedman_mse"],
                },
                "Linear Regression": {},
                "K-neighbours": {
                    "n_neighbors": [5,7,9,10,12],
                    "weights":["uniform", "distance"],
                    # "algorithm":["auto", "ball_tree", "kd_tree", "brute"]
                },
                "XGBoost": {
                    "learning_rate": [0.1,0.05,0.01,0.001],
                    'n_estimators':[8,16,32,64,128,256],
                },
                "AdaBoost":{
                    "learning_rate": [0.1,0.05,0.01,0.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators':[8,16,32,64,128,256]
                }

            }

            report = evaluate_model(X_train, y_train, X_test, y_test, models = models, params = params)
            
            train_report = report[0]
            model_report = report[1]

            report = {}
            for k in train_report.keys():
                report[k] = [train_report[k],model_report[k]]

            report = pd.DataFrame.from_dict(report, orient = 'index', columns= ['Train_score', 'Test_score'])
            print(report)
            report.to_csv('artifacts/model_report.csv')

            ##To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ##To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best Model Found!!')

            logging.info(f"Best model found: {best_model} scoring good {best_model_score} on both training and test dataset")
            save_object(self.model_trainer.trained_model_file_path, best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)

