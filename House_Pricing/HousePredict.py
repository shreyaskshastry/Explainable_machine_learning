#installing required library
import subprocess

subprocess.call(['pip', 'install', '-U' , 'dblue-mlwatch'])

#importing libraries
import joblib
import numpy as np
import pandas as pd
import uuid
from dblue_mlwatch import MLWatch

#Python file for Pre-Processnig
import house_pre_process as hpp

#setting MLWatch envi-variables and agruments
account = "dblue"
api = "c3FIOG9vSGV4VHo4QzAyg5T1JvNnJoZ3ExaVNyQWw6WjRsanRKZG5lQk9qUE1BVQ"
watcher = MLWatch(
    account = account,
    api_key = api,
    model_id = "house-price-prediction",
    model_version="1-0-0"
)


class HousePredict:
    def __init__(self):
        self.svr = joblib.load('filename.pkl')        
    
    def predict(self, features, feature_names, **kwargs):        
        X_transform = hpp.master_func(features)
        prediction = np.floor(np.expm1(self.svr.predict(X_transform))).tolist()
        
        unique_id = uuid.uuid4().hex
        
        predict_data = {
                  'unique_id': unique_id,
                  'features': features,
                  'prediction' : float(prediction[0])
                }
                
        pred_ls = []
        
        pred_ls.append(prediction[0])
        pred_ls.append(unique_id)
        
        watcher.capture_prediction(predict_data)
        
        return pred_ls