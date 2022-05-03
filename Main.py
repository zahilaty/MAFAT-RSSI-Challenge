# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:08:59 2022

@author: zahil
"""

### Imports ###
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as sio
from TrainAux import MyDataset
from helper_func import MyResNet18,DealWithOutputs

### HyperParams ###
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0003
mat_file = 'Data\DataV2_mul.mat'
regression_or_classification = 'classification' #regression

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

### Net type (anyway the last layer will be sigmoid)
if regression_or_classification == 'regression':
    net  = MyResNet18(InputChannelNum=2,IsSqueezed=0,LastSeqParamList=[512,32,1],pretrained=True).cuda()
if regression_or_classification == 'classification':
    net  = MyResNet18(InputChannelNum=2,IsSqueezed=0,LastSeqParamList=[512,32,4],pretrained=True).cuda()

### Creterion - I dont see any reason to use MSE and not MAE at this moment
loss_fn = nn.L1Loss(reduction='none') 
    
### optimizer declaration - must be after net declaration
optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)

### Prepare val data
val_samples, val_labels ,val_weights = val_set[:]
targets_val = torch.reshape(torch.tensor(val_labels),(-1,1)).float().cuda()
val_weights = torch.reshape(torch.tensor(val_weights),(-1,1)).float().cuda()

### The training loop
Costs = np.array([])
Costs_val = np.array([])
for Epoch in range(EPOCHS):
    for batch_i, [batch,labels,weights] in enumerate(train_dataloader):
        #if batch_i>0: #same as: batch_i, [batch,label] = next(enumerate(train_dataloader)) #for debug
        #    continue
        optimizer.zero_grad()
        outputs = net(batch)        
        targets = torch.reshape(labels,(-1,1)).float().cuda()
        weights = torch.reshape(weights,(-1,1)).float().cuda()
        outputs = DealWithOutputs(regression_or_classification,outputs)
        #torch can convert to "one hot" if needed, but here we give 1D output anyway 
        loss = torch.mean(loss_fn(outputs,targets)*weights)
        Costs = np.append(Costs,loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        #check ValAccuracy every epoch
        net.eval()
        with torch.no_grad():
            outputs_val = net(val_samples)
            outputs_val = DealWithOutputs(regression_or_classification,outputs_val)
            loss_val = torch.mean(loss_fn(outputs_val,targets_val)*val_weights) #should be AUC for bin case, and L1 (as now) for multy label
            accuracy_bin = torch.sum((outputs_val>0.5) == torch.reshape(targets_val>0,(-1,1)).cuda())/len(val_set)
            Costs_val = np.append(Costs_val,loss_val.cpu().detach().numpy())
        net.train()
        print('[%d, %5d] loss: %.3f  Loss_val: %.3f Accuracy: %.1f' %(Epoch + 1, batch_i + 1, Costs[-1],Costs_val[-1],accuracy_bin*100))
        
#TODO
# 1) binary classification will be post-process of regression (avoid unbalanced data)
# 2) For multylabel use CE or L1 (their metric is MAE)
# 3) For Binary use focal loss or use  the multy version with round at the end

# 4) change data\ to os.join
# 5) add if device
# 6) logs

