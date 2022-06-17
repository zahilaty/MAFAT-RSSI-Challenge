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

def GetResNet101(InputChannelNum=4,LastSeqParamList=[2048,32],pretrained=True):
    net = torchvision.models.resnet101(pretrained=pretrained)
    net.conv1 = nn.Conv2d(InputChannelNum, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #changed number of channels from 3 to InputChannelNum. The rest is the same
    # cascading layers with add_module
    LastLayers = nn.Sequential()
    for ind,val in enumerate(LastSeqParamList[:-1]):
        val_next = LastSeqParamList[ind+1]
        if ind == len(LastSeqParamList)-2:
            LastLayers.add_module('layer'+str(ind),nn.Sequential(nn.Dropout(p=0.0),nn.Linear(in_features=val,out_features=val_next,bias=True),nn.Sigmoid()))
        else:
            LastLayers.add_module('layer'+str(ind),nn.Sequential(nn.Dropout(p=0.75),nn.Linear(in_features=val,out_features=val_next,bias=True),nn.ReLU()))
    net.fc = LastLayers
    return net
        

    
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x     
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
    
def CalcExpectation(outputs):
    # Probabilties calculation is better than softmax, because the metric is MAE(same as L1?)
    Probs = outputs/outputs.sum(axis=1,keepdims=True)
    Expectation = (Probs[:,0]*0 + Probs[:,1]*1 + Probs[:,2]*2 + Probs[:,3]*3).reshape(-1,1)
    return Expectation

def ConvertTargetsToOrdinal(targets,K=4):
    # n = len(targets)
    # for i in range(n):
    #   if int(tmp_y[i])   == 0: tmp_y[i] = 0.125
    #   elif int(tmp_y[i]) == 1: tmp_y[i] = 0.375
    #   elif int(tmp_y[i]) == 2: tmp_y[i] = 0.625
    #   elif int(tmp_y[i]) == 3: tmp_y[i] = 0.875
    #   else: print("Fatal logic error ")
    targets = targets*(1/K)+1/(2*K)
    return targets

def float_oupt_to_class(oupt, k):
  # end_pts = np.zeros(k+1, dtype=np.float32) 
  # delta = 1.0 / k
  # for i in range(k):
  #   end_pts[i] = i * delta
  # end_pts[k] = 1.0
  # if k=4, [0.0, 0.25, 0.50, 0.75, 1.0] 
  end_pts = [0.0, 0.25, 0.50, 0.75, 1.0]
  for i in range(k):
    if oupt >= end_pts[i] and oupt <= end_pts[i+1]:
      return i

def Myfloat_oupt_to_class(oupt, K=4):
    classes = torch.round((oupt-1/(2*K))*K).int()
    classes[np.where(classes.detach().cpu().numpy()==4)] = 3
    return classes
    
def ExtractFeaturesFromVecs(X):
    #assert(X.shape[0] == 2)
    #assert(X.shape[1] == 360)
    #c1 = (X[1,:]-np.mean(X[1,:])) - (X[0,:]-np.mean(X[0,:]))
    #c3 = 10*np.log10(10**(X[1,:]/10) + 10**(X[0,:]/10))
    x1 = X[0,:]
    x2 = X[1,:]
    
    #x1,x2 = NormlizeX(x1,x2)
    
    # x1_ZeroMean = x1 - np.mean(x1)
    # x2_ZeroMean = x2 - np.mean(x2)
    # x1_std = np.std(x1)
    # x2_std = np.std(x2)
    Avarage = (x1 + x2)/2.0 
    diff = np.abs(x2 - x1)
    
    c1 = Avarage
    c2 = np.diff(c1,prepend=c1[0])
    c3 = diff
    c4 = np.diff(c3,prepend=c3[0])
    #c5 = np.correlate(x1_ZeroMean,x2_ZeroMean,mode='same')/(x1_std+1e-15)/(x2_std+1e-15) #can be clipped for scaling
    
    #signal = np.concatenate((c1.reshape(1,-1),c2.reshape(1,-1),c3.reshape(1,-1),c4.reshape(1,-1),c5.reshape(1,-1)),axis=0)
    signal = np.concatenate((c1.reshape(1,-1),c2.reshape(1,-1),c3.reshape(1,-1),c4.reshape(1,-1)),axis=0)
    return signal

def EnsamblePred(net,samples):
    ShiftList = [-150,-90,-30,0,30,90,150]
    Ensemble_out = torch.zeros((len(samples),len(ShiftList)),device=('cuda'))
    for ind,T in enumerate(ShiftList):
        shifted_input = torch.roll(samples, T, dims=2) #I could have done it in numpy on x1,x2
        tmp_outputs = net(shifted_input)
        tmp_outputs = CalcExpectation(tmp_outputs)
        Ensemble_out[:,ind] = tmp_outputs.reshape(-1,)
    #predicted_method_2 = torch.mean(Ensemble_out,dim=1)
    predicted_method_2 = torch.median(Ensemble_out,dim=1)[0]
    predicted_method_2 = torch.round(predicted_method_2).reshape(-1,)
    return predicted_method_2,Ensemble_out

def EnsamblePredForAUC(net,samples):
    ShiftList = [-150,-90,-30,0,30,90,150]
    Ensemble_out = torch.zeros((len(samples),len(ShiftList)),device=('cuda'))
    for ind,T in enumerate(ShiftList):
        shifted_input = torch.roll(samples, T, dims=2) #I could have done it in numpy on x1,x2
        tmp_outputs = net(shifted_input) #1x4
        tmp_Probs = tmp_outputs/tmp_outputs.sum(axis=1,keepdims=True)
        tmp_prob_preds = torch.sum(tmp_Probs[:,1:],axis=1)
        Ensemble_out[:,ind] = tmp_prob_preds.reshape(-1,)
        prob_preds = torch.median(Ensemble_out,dim=1)[0]
    return prob_preds,Ensemble_out

def GetNumOfUniqValues(combined):
    uniq = np.unique(combined,axis=1)
    return uniq.shape[1]

def GetNumOfUniqValues_torch_ver(samples):
    c1 = torch.squeeze(samples[:,0,:,:]) #176x360
    c3 = torch.squeeze(samples[:,2,:,:]) #176x360
    x1 = ((2*c1-c3)/2).cpu().detach().numpy()
    x2 = ((2*c1+c3)/2).cpu().detach().numpy()
    combined = np.hstack((x1,x2))
    NumOfUniqValues = np.zeros((samples.shape[0],1))
    for k in range(samples.shape[0]):
        NumOfUniqValues[k] = len(np.unique(combined[k,:]))
    return NumOfUniqValues

def NormlizeX(x1,x2):
    # we already saw that whitening the signal (remove dc) is not good,
    # so I will only remove outliers and scale
    x1 = np.clip(x1, -70, -20)
    x2 = np.clip(x2, -70, -20)
    x1 = x1 / 70
    x2 = x2 / 70
    return x1,x2