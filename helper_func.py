# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 22:58:41 2021

@author: zahil
"""

import torch
import torchvision
import torch.nn as nn
#import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np

class MyResNet18(ResNet):
    def __init__(self,InputChannelNum=2,IsSqueezed=0,LastSeqParamList=[512,32],pretrained=True):
        
        super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2]) #one cant just call resnet18 because it is a function and not a network
        
        self.InputChannelNum = InputChannelNum       
        self.IsSqueezed = IsSqueezed
        
        self.conv1 = nn.Conv2d(InputChannelNum, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #changed number of channels from 3 to InputChannelNum. The rest is the same

        # cascading layers with add_module
        LastLayers = nn.Sequential()
        for ind,val in enumerate(LastSeqParamList[:-1]):
            val_next = LastSeqParamList[ind+1]
            if ind == len(LastSeqParamList)-2:
                LastLayers.add_module('layer'+str(ind),nn.Sequential(nn.Dropout(p=0.0),nn.Linear(in_features=val,out_features=val_next,bias=True),nn.Sigmoid()))
            else:
                LastLayers.add_module('layer'+str(ind),nn.Sequential(nn.Dropout(p=0.25),nn.Linear(in_features=val,out_features=val_next,bias=True),nn.ReLU()))
        self.fc = LastLayers


    def forward(self, x):
        if self.InputChannelNum == 1 & self.IsSqueezed == 1:
            #return self._forward_impl(x[:,[0],:,:]) #[0] instead of 0 to keep the dim
            x = torch.unsqueeze(x,1)
            return self._forward_impl(x) 
        else:
            return self._forward_impl(x)
        
# https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/4    
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py          
# https://discuss.pytorch.org/t/add-sequential-model-to-sequential/71765 

def DealWithOutputs(regression_or_classification,outputs):
    if regression_or_classification == 'regression':
        outputs = outputs*3 #scale the sigmoid to [0,3]
        return outputs
    if regression_or_classification == 'classification':
        # Probabilties calculation is better than softmax, because the metric is MAE(same as L1?)
        Probs = outputs/outputs.sum(axis=1,keepdims=True)
        Expectation = (Probs[:,0]*0 + Probs[:,1]*1 + Probs[:,2]*2 + Probs[:,3]*3).reshape(-1,1)
        return Expectation

def ExtractFeaturesFromVecs(X):
    #assert(X.shape[0] == 2)
    #assert(X.shape[1] == 360)
    c1 = X[1,:] - X[0,:]
    c2 = np.diff(c1,prepend=0)
    c3 = 10*np.log10(10**(X[1,:]/10) + 10**(X[0,:]/10))
    signal = np.concatenate((c1.reshape(1,-1),c2.reshape(1,-1),c3.reshape(1,-1)),axis=0)
    return signal
    
# def preprocess(X, RSSI_value_selection):

    
    # signal = torch.tensor(self.audio_mat[index,:],dtype=torch.float32) #2x360
    # signal = signal.to(self.device)
    # signal = signal - signal.mean() # A sort of augmentation..
    # signal = torch.unsqueeze(signal, 2) #2x360x1
        
        
#     """
#     Calculate the features on the selected RSSI on the test set
#     :param X: Dataset to extract features from.
#     :param RSSI_value_selection: Which signal values to use- - in our case it is Average.
#     :return: Test x dataset with features
#     """
#     if RSSI_value_selection=="RSSI_Left":
#         X["RSSI"] = X.RSSI_Left
#     elif RSSI_value_selection=="RSSI_Right":
#         X["RSSI"] = X.RSSI_Right
#     elif RSSI_value_selection=="Min":
#         X["RSSI"] = X[['RSSI_Left','RSSI_Right']].min(axis=1).values
#     elif RSSI_value_selection=="Max":
#         X["RSSI"] = X[['RSSI_Left','RSSI_Right']].max(axis=1).values
#     else: 
#         X["RSSI"] = np.ceil(X[['RSSI_Left','RSSI_Right']].mean(axis=1).values).astype('int')

#     X, features_name = extract_features(X)
#     X.drop('Device_ID', axis=1, inplace=True)
#     return X[features_name]
    
    
    
    