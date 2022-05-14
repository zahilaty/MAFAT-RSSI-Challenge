# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 22:58:41 2021

@author: zahil
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
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


    # def forward(self, x):
    #     if self.InputChannelNum == 1 & self.IsSqueezed == 1:
    #         #return self._forward_impl(x[:,[0],:,:]) #[0] instead of 0 to keep the dim
    #         x = torch.unsqueeze(x,1)
    #         return self._forward_impl(x) 
    #     else:
    #         return self._forward_impl(x)
        
##############################################################################################################

class MyBasicBlock(nn.Module):
    def __init__(self):
        super(MyBasicBlock,self).__init__()
        
        
        self.BlockOne = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=5,stride=3),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Dropout(p=0.25))
        
        self.BlockTwo = nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=5,stride=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        dummi_input = torch.zeros((1,1,360))
        self.tmp_num = num_flat_features(self.BlockTwo(self.BlockOne(dummi_input)))
        print('during constructor of this object the FC size was calculated to be: ' ,self.tmp_num)

    def forward(self,x):
        x = self.BlockOne(x)
        x = self.BlockTwo(x)
        x = x.view(-1,self.tmp_num)
        return x
    
class Net1D(nn.Module):
    def __init__(self):
        super(Net1D,self).__init__()
        self.CH1 = MyBasicBlock()
        self.CH2 = MyBasicBlock()
        self.CH3 = MyBasicBlock()

        self.fc_final = nn.Sequential(nn.Linear(3*39*32,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(p=0.5),
                                      nn.Linear(256,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(p=0.25),
                                      nn.Linear(32,4),nn.Sigmoid())
                                
    def forward(self,x):
        ch1 = self.CH1(torch.unsqueeze(x[:,0,:],1))
        ch2 = self.CH1(torch.unsqueeze(x[:,1,:],1))
        ch3 = self.CH1(torch.unsqueeze(x[:,2,:],1))
        x = torch.concat((ch1,ch2,ch3),dim=1)
        x = self.fc_final(x)
        return x

##debug:
#dummi_input = torch.zeros((100,3,360))
#net = Net1D()
#net(dummi_input)

##############################################################################################################
def num_flat_features(x):
    size = x.size()[1:] #all dimensions except the batch dimention
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
    
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
    #c1 = (X[1,:]-np.mean(X[1,:])) - (X[0,:]-np.mean(X[0,:]))
    c1 = X[1,:] - X[0,:]
    c2 = np.diff(c1,prepend=0)
    c3 = 10*np.log10(10**(X[1,:]/10) + 10**(X[0,:]/10))
    signal = np.concatenate((c1.reshape(1,-1),c2.reshape(1,-1),c3.reshape(1,-1)),axis=0)
    return signal
    
# def preprocess(X, RSSI_value_selection):
    
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