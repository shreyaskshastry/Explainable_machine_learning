#!/usr/bin/env python
# coding: utf-8

#installing required libraries
import subprocess
subprocess.call(['pip', 'install', '-U' , 'dblue-mlwatch'])

import os
import joblib
import uuid
from dblue_mlwatch import MLWatch

#importing pre-processing file
import pre_process as pp

path = os.getcwd()
in_path = os.path.join(path,'data')
model_path = os.path.join(in_path,'nbmodel1.pkl')

account = "dblue"
api = "c3FIOG9vSGV4VHo4QzAyg5T1JvNnJoZ3ExaVNyQWw6WjRsanRKZG5lQk9qUE1BVQ"
watcher = MLWatch(
    account = account,
    api_key = api,
    model_id = "amazon-food-review-sentiment-analysis",
    model_version="1-0-0"
)


class AmazonPredict:
    def __init__(self):
        self.nb_model = joblib.load(model_path)
    
    def predict(self, features, feature_names, **kwargs):        
        X_transform = pp.pred(features)
        prediction = self.nb_model.predict(X_transform).tolist()
        probability = self.nb_model.predict_proba(X_transform).tolist()
        
        unique_id = uuid.uuid4().hex
        
        predict_data = {
                        'unique_id': unique_id,
                        'features': features,
                       'prediction' : prediction[0],
                        'prediction_probs' : {
                            'negative' : probability[0][0],
                            'neutral' : probability[0][1],
                            'positive' : probability[0][2]
                        }
                }
        
        watcher.capture_prediction(predict_data)
        
        prediction.append(probability)
        prediction.append(unique_id)
        return (prediction)