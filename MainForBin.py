# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:08:59 2022

@author: zahil
"""

#Only in the notebook
# from google.colab import files
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# print(os.getcwd())
### Imports ###
import numpy as np
import torch
from torch.utils.data import DataLoader
from TrainAux import MyDataset
from helper_func import MyResNet18,GetResNet101,EnsamblePredForAUC,GetNumOfUniqValues_torch_ver
import scipy.io as sio
from sklearn.metrics import confusion_matrix,roc_auc_score

### HyperParams ###
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 0.0001
mat_file = 'Data\DataV2_mul.mat' #TODO change data\ to os.join
LastCheckPoint = None ## A manual option to re-train # 05_06\\ResNet_0.473.pth

### DataSets ###
my_ds = MyDataset(mat_file,'cuda',Return1D = False,augmentations = True) #calling the after-processed dataset
my_ds_val = MyDataset(mat_file,'cuda',Return1D = False,augmentations = False) #calling the after-processed dataset
l1 = np.reshape(sio.loadmat(mat_file)["l1"],(-1,)) # we need to save the indexes so we wont have data contimanation
l2 = np.reshape(sio.loadmat(mat_file)["l2"],(-1,))
assert len(np.intersect1d(l1,l2)) == 0
train_set = torch.utils.data.Subset(my_ds, l1[::100]) #sample in the local version
val_set = torch.utils.data.Subset(my_ds_val, l2)
print(f"There are {len(train_set)} samples in the train set and {len(val_set)} in validation set.")
#How to get single sample for testing:   signal, label = demod_ds[0] ; signal.cpu().detach().numpy() ; %varexp --imshow sig

### DataLoader ### 
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE,drop_last=False,shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=val_set.dataset.__len__())

### transform ###
#Constand transforms are defined in the dataset
#RandomTransforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])

### Net type (anyway the last layer will be sigmoid)
net  = MyResNet18(InputChannelNum=4,IsSqueezed=0,LastSeqParamList=[512,32,1],pretrained=True).cuda()
#net  = GetResNet101(InputChannelNum=4,LastSeqParamList=[2048,512,32,1],pretrained=True).cuda()

### A new try
#net.fc[0][0].p = 0.9
#net.fc[0][0].p = 0.75
#net.fc[2][0].p = 0.5

### Creterion - I dont see any reason to use MSE and not MAE at this moment
#loss_fn = torchvision.ops.sigmoid_focal_loss(alpha = (5737/13187), gamma = 2, reduction = 'mean')
#loss_fn_val = torchvision.ops.sigmoid_focal_loss(alpha = 0.5, gamma = 2, reduction = 'mean')
alpha = 5737/(13187+5737)
alpha_val = 0.5 
gamma = 2

### optimizer declaration - must be after net declaration
optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)

### Prepare val data
#val_samples, val_labels ,val_weights = val_set[:] #will not work when I use custum preprocess function 
val_samples, val_labels ,val_weights = next(iter(val_dataloader)) #better practice
targets_val = torch.reshape(val_labels,(-1,1)).float().cuda()
val_weights = torch.reshape(val_weights,(-1,1)).float().cuda()
y_true = targets_val.cpu().detach().numpy()
y_true_bin = y_true>0.9

### The training loop
Costs = np.array([])
Score_val = np.array([])
if LastCheckPoint is not None:
    net.load_state_dict(torch.load(LastCheckPoint)) 
net.train() # In case we re-run this "cell"
max_auc_val = 0.4
#batch_i, [batch,labels,weights] = next(enumerate(train_dataloader)) #for debug
for Epoch in range(EPOCHS):
    for batch_i, [batch,labels,weights] in enumerate(train_dataloader):
        #if batch_i !=3: # needed for augmentations debug
        #    continue
        optimizer.zero_grad()
        #inputs = my_augmentations(batch)
        outputs = net(batch)        
        y = torch.reshape(torch.clip(labels,min=0,max=1),(-1,1)).int().cuda()
        loss = torch.mean(-alpha * ((1 - outputs) ** gamma) * torch.log(outputs+torch.tensor(1e-10)) * y - (1 - alpha) * (outputs ** gamma) * torch.log(1 - outputs+torch.tensor(1e-10)) * (1 - y))
        Costs = np.append(Costs,loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        #check ValAccuracy every epoch
        net.eval()
        with torch.no_grad():
            outputs_val = net(val_samples)       
            AUC_val = roc_auc_score(y_true_bin, outputs_val.cpu().numpy())
            Score_val = np.append(Score_val,AUC_val)
           
            if  (AUC_val>max_auc_val): 
                min_name = 'ResNet_' + str(round(AUC_val,2))
                torch.save(net.state_dict(), min_name  + '.pth' )
                max_auc_val += 0.02
                predicted_method_1 = torch.round(outputs_val).reshape(-1,)
                y_pred_1 = predicted_method_1.cpu().detach().numpy()
                cf_matrix_1 = confusion_matrix(y_true_bin,y_pred_1)
                sio.savemat(min_name + '.mat', {"Costs": Costs, "Score_val": Score_val,'cf_matrix_1':cf_matrix_1})
        net.train()
        print('[%d, %5d] loss: %.3f  AUC: %.2f' %(Epoch + 1, batch_i + 1, Costs[-1],AUC_val))

### Save net weights
print('Finished Training')
#end_name = LastCheckPoint
#torch.save(net.state_dict(),end_name)
# myfile = min_name + '.pth' #only in the notebook
# myfile2 = min_name + '.mat' #only in the notebook

### Starting all kinds of eval
if 'min_name' in locals(): 
    net.load_state_dict(torch.load(min_name + '.pth')) #else, we continue with "end_name"
net.eval()
with torch.no_grad():
    outputs_val = net(val_samples)
    NumOfUniqValues = GetNumOfUniqValues_torch_ver(val_samples)

### AUC Eval
with torch.no_grad():
    y_pred_1 = outputs_val.cpu().detach().numpy()
    AUC_val = roc_auc_score(y_true_bin, y_pred_1)
    print('AUC on val set:',AUC_val)
    
    y_pred_2,_ = EnsamblePredForAUC(net,val_samples)
    y_pred_2 = y_pred_2.cpu().numpy()
    AUC_val_ensamble = roc_auc_score(y_true_bin, y_pred_2)
    print('AUC on val set with ensamble:',AUC_val_ensamble)

    y_pred_3 = np.copy(y_pred_1)
    y_pred_3[np.where(NumOfUniqValues > 19)[0],0] = 0
    Fixed_AUC_val = roc_auc_score(y_true_bin,y_pred_3)
    print('AUC on val set with UniqValues fix:',Fixed_AUC_val)

### Confussion Matrix
cf_matrix_1 = confusion_matrix(y_true_bin,np.round(y_pred_1))
cf_matrix_2 = confusion_matrix(y_true_bin,np.round(y_pred_2))
cf_matrix_3 = confusion_matrix(y_true_bin,np.round(y_pred_3))

### Accuracy
Accuracy_1 = np.trace(cf_matrix_1)/len(val_set)
print(f'Accuracy on val set: {100 * Accuracy_1} %')
Accuracy_2 = np.trace(cf_matrix_2)/len(val_set)
print(f'Accuracy on val set: {100 * Accuracy_2} %')
Accuracy_3 = np.trace(cf_matrix_3)/len(val_set)
print(f'Accuracy on val set: {100 * Accuracy_3} %')


sio.savemat('LossVals.mat', {"Costs": Costs, "Score_val": Score_val,
                             'cf_matrix_1':cf_matrix_1,'cf_matrix_2':cf_matrix_2,'cf_matrix_3':cf_matrix_3,
                             'outputs_val':outputs_val.cpu().detach().numpy()})

### Only in notebook
# !cp $myfile /content/drive/MyDrive/ResNet101/
# !cp $myfile2 /content/drive/MyDrive/ResNet101/
# !cp $end_name /content/drive/MyDrive/ResNet101/
# !cp 'LossVals.mat' /content/drive/MyDrive/ResNet101/
# print(min_name)