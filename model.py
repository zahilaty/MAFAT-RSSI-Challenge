import torch
import pickle
import numpy as np
from os.path import isfile
import joblib
from sklearn.ensemble import RandomForestClassifier
from helper_func import MyResNet18,ExtractFeaturesFromVecs,DealWithOutputs,GetResNet101,EnsamblePred,EnsamblePredForAUC,GetNumOfUniqValues
import os

### Checklist for manual changes before submission:
# 1) resnet 18 or 101 in line 20
# 2) Track 1 of 2 in line 26
# 3) Ensamble

class model:
    def __init__(self):
        '''
        Init the model
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #net = MyResNet18(InputChannelNum=4,IsSqueezed=0,LastSeqParamList=[512,32,4],pretrained=False) #should we convert to cude?
        net  = GetResNet101(InputChannelNum=4,LastSeqParamList=[2048,512,32,4],pretrained=False)
        #net = torch.load('CompleteModel.pt')
        net = net.to(self.device)
        net.eval()
        self.model = net
        self.WhichTrack = 1
        self.ensemble = False
        
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
        Rssi_Left = (X['RSSI_Left'].to_numpy()).reshape(1,-1) #1x360
        Rssi_Right = (X['RSSI_Right'].to_numpy()).reshape(1,-1) #1x360
        #NumOfUniqValues = GetNumOfUniqValues(np.concatenate((Rssi_Left,Rssi_Right),axis=1))
        Xnew = np.concatenate((Rssi_Left,Rssi_Right),axis=0) #2x360
        signal = ExtractFeaturesFromVecs(Xnew) #3x360
        signal = torch.tensor(signal,dtype=torch.float32)
        signal = signal.to(self.device)
        signal = torch.unsqueeze(signal, 2) #2x360x1
        signal =  torch.unsqueeze(signal, 0) #1x2x360x1 (the batch dim)
        
        if self.WhichTrack == 2:
            if self.ensemble == True:
                predicted_method_2,_ = EnsamblePred(self.model,signal)
                y = int(predicted_method_2)
            else:
                outputs = self.model(signal) #1x4 
                outputs = DealWithOutputs('classification',outputs) #1x1
                #y = outputs.item() # Error: Prediction values  should be of type int.
                # round to nearest int
                predicted_method_1 = torch.round(outputs).reshape(-1,) #1,
                y = int(predicted_method_1.item())
        
        if self.WhichTrack == 1:
            if self.ensemble == True: 
                prob_preds,_ = EnsamblePredForAUC(self.model,signal)
                prob_preds = prob_preds.item()
            else:
                outputs = self.model(signal) #1x4 
                Probs = outputs/outputs.sum(axis=1,keepdims=True)
                prob_preds = torch.sum(Probs[:,1:],axis=1).item()
            if prob_preds>1.0:
                prob_preds = 1.0
            if prob_preds<0.0:
                prob_preds = 0.0
            y = prob_preds
               
        return y

    def load(self, dir_path):
        '''
        Edit this function to fit your model.

        This function should load the model that you trained on the train set.
        :param dir_path: A path for the folder the model is submitted 
        ''' 
        model_name = 'NetWeights.pth' 
        model_file = os.path.join(dir_path, model_name)
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()
        
        #model_name = 'CompleteModel.pt' 
        #model_file = os.path.join(dir_path, model_name)
        #net = torch.load(model_file)
        #net = net.to(self.device)
        #net.eval()
        #self.model = net