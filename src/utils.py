import os,sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

            logging.info('Pickle file saved to artifacts')

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train, y_train, X_test, y_test, models,params):
    try:
        train_report={}
        test_report={}
        
        logging.info('Training Stated....')
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = list(params.values())[i]

            print(model, " is training ")

            gs = GridSearchCV(model,para,cv=5)

            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            train_report[list(models.keys())[i]] = train_model_score
            test_report[list(models.keys())[i]] = test_model_score
            # logging.info(f"{model} gives the Scores, train: {train_model_score}, test: {test_model_score}")
        return train_report,test_report

    except Exception as e:
        raise CustomException(e,sys)


