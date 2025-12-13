import os
import sys
import dill
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import Custom_Execption
from sklearn.model_selection import GridSearchCV

def saved_object(file_path,obj):
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise Custom_Execption(e,sys)
    
##for model training
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv = 3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            r2_score1 = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = r2_score1

        return report
    except Exception as e:
        raise Custom_Execption(e,sys)
