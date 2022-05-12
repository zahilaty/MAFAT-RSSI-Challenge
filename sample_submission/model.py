
import pickle
import numpy as np
from os.path import isfile
import joblib
from sklearn.ensemble import RandomForestClassifier
from helper_func import preprocess
import os

class model:
    def __init__(self):
        '''
        Init the model
        '''

        self.model  = RandomForestClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=2,
                              n_estimators=350, random_state=0, class_weight="balanced", bootstrap = True)
        self.RSSI_value_selection = 'Average'

    def predict(self, X):
        '''
        Edit this function to fit your model.

        This function should provide predictions of labels on (test) data.
        Make sure that the predicted values are in the correct format for the scoring
        metric.
        preprocess: it our code for add feature to the data before we predict the model.
        :param X: is DataFrame with the columns - 'Time', 'Device_ID', 'Rssi_Left','Rssi_Right'. 
                  X is window of size 360 samples time, shape(360,4).
        :return: a float value of the prediction for class 1 (the room is occupied).
        '''
        # preprocessing should work on a single window, i.e a dataframe with 360 rows and 4 columns
        X = preprocess(X,self.RSSI_value_selection)
        y = self.model.predict_proba(X)[:,1][0]
        
        '''
        Track 2 - for track 2 we naively assume that the model from track-1 predicts 0/1 correctly. 
        We use that assumption in the following way:
        when the room is occupied (1,2,3 - model predicted 1) we assign the majorty class (2) as prediction.       
        '''
        y = 0 if y<0.5 else 2
        return y

    def load(self, dir_path):
        '''
        Edit this function to fit your model.

        This function should load the model that you trained on the train set.
        :param dir_path: A path for the folder the model is submitted 
        '''
        model_name = 'model_track_2.sav' 
        model_file = os.path.join(dir_path, model_name)
        self.model = joblib.load(model_file)