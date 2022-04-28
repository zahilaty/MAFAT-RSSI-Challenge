# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:08:59 2022

@author: zahil
"""

### Imports ###
#import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as sio
from TrainAux import MyDataset
from helper_func import MyResNet18

### HyperParams ###
BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.0003
mat_file = 'Data\DataV1_bin.mat'

### DataSets ###
my_ds = MyDataset(mat_file,'cuda') #calling the after-processed dataset
l1 = np.reshape(sio.loadmat(mat_file)["l1"],(-1,)) # we need to save the indexes so we wont have data contimanation
l2 = np.reshape(sio.loadmat(mat_file)["l2"],(-1,))
assert len(np.intersect1d(l1,l2)) == 0
train_set = torch.utils.data.Subset(my_ds, l1)
val_set = torch.utils.data.Subset(my_ds, l2)
print(f"There are {len(train_set)} samples in the train set and {len(val_set)} in validation set.")
#How to get single sample for testing:   signal, label = demod_ds[0] ; signal.cpu().detach().numpy() ; %varexp --imshow sig

### DataLoader ### 
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE,drop_last=False)
test_dataloader = DataLoader(val_set, batch_size=val_set.dataset.__len__())

### transform ###
# TBD - there is an error here -> transform get the mean and std you want to remove
#MyRandomTransforms = transforms.Compose([transforms.Normalize((0,), (1,)),transforms.RandomAffine(degrees = 0, translate=(0.2,0.15)),AddGaussianNoise(0., 1.)])

######### Part A - train with out unsupervised pretraining
net  = MyResNet18(InputChannelNum=2,IsSqueezed=0,LastSeqParamList=[512,32,1],pretrained=True).cuda()
loss_fn = nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)
Costs = np.array([])
Costs_val = np.array([])

#prepare val data
val_samples, val_labels = val_set[:]
target_val = torch.reshape(torch.tensor(val_labels),(-1,1)).float().cuda()

for Epoch in range(EPOCHS):
    #batch_i, [batch,label] = next(enumerate(train_dataloader))
    for batch_i, [batch,label] in enumerate(train_dataloader):
        optimizer.zero_grad()
        prob = net(batch)
        target = torch.reshape(label,(-1,1)).float().cuda()
        loss = loss_fn(prob,target)
        Costs = np.append(Costs,loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        #check ValAccuracy every epoch
        net.eval()
        with torch.no_grad():   
            p_val = net(val_samples)
            accuracy = torch.sum((p_val>0.5) == torch.reshape(target_val,(-1,1)).cuda())/len(val_set)
            loss_val = loss_fn(p_val,target_val) #TBD - AUC
            Costs_val = np.append(Costs_val,loss_val.cpu().detach().numpy())
        net.train()
        print('[%d, %5d] loss: %.3f  Loss_val: %.3f Accuracy: %.1f' %(Epoch + 1, batch_i + 1, Costs[-1],Costs_val[-1],accuracy*100))
        
#TODO
# 0) Create Y in matlab with person number
# 1) fork into 2 branchs, where one will predict multy label and the other binary
#       Alternative: binary classification will be post-process of regression (avoid unbalanced data)
# 2) For multylabel use CE or L1 (their metric is MAE)
# 3) For Binary use focal loss or use  the multy version with round at the end
# 4) sigmoid*4
# 5) Probabilty calculation vs softmax

