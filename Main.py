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
import torch.nn as nn
from torch.utils.data import DataLoader
from TrainAux import MyDataset,my_augmentations
from helper_func import MyResNet18,DealWithOutputs,Net1D,Identity,GetResNet101,EnsamblePred,EnsamblePredForAUC,GetNumOfUniqValues_torch_ver
import scipy.io as sio
from sklearn.metrics import confusion_matrix,roc_auc_score

### HyperParams ###
BATCH_SIZE = 1024
EPOCHS = 0
LEARNING_RATE = 0.0001
mat_file = 'Data\DataV2_mul.mat' #TODO change data\ to os.join
regression_or_classification = 'classification' #regression
#net = Net1D().cuda()
LastCheckPoint = 'Checkpoints\\16_06\\ResNet_0.441_0.93.pth' #None ## A manual option to re-train # 05_06\\ResNet_0.473.pth

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
if regression_or_classification == 'regression':
    net  = MyResNet18(InputChannelNum=3,IsSqueezed=0,LastSeqParamList=[512,32,1],pretrained=True).cuda()
if regression_or_classification == 'classification':
    #net  = MyResNet18(InputChannelNum=4,IsSqueezed=0,LastSeqParamList=[512,32,4],pretrained=True).cuda()
    net  = GetResNet101(InputChannelNum=4,LastSeqParamList=[2048,512,32,4],pretrained=True).cuda()

### A new try
#net.fc[0][0].p = 0.9
#net.fc[0][0].p = 0.75
#net.fc[2][0].p = 0.5

### Creterion - I dont see any reason to use MSE and not MAE at this moment
loss_fn = nn.L1Loss(reduction='none') 
loss_fn_val = nn.L1Loss(reduction='none') 
    
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
Costs_val_weighted  = np.array([])
Costs_val = np.array([])
if LastCheckPoint is not None:
    net.load_state_dict(torch.load(LastCheckPoint)) 
net.train() # In case we re-run this "cell"
min_loss_val = 0.55
max_auc_val = 0.4
#batch_i, [batch,labels,weights] = next(enumerate(train_dataloader)) #for debug
for Epoch in range(EPOCHS):
    for batch_i, [batch,labels,weights] in enumerate(train_dataloader):
        save_flag = 0
        #if batch_i !=3: # needed for augmentations debug
        #    continue
        optimizer.zero_grad()
        #inputs = my_augmentations(batch)
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
            
            Probs = outputs_val/outputs_val.sum(axis=1,keepdims=True)
            prob_preds = np.clip(torch.sum(Probs[:,1:],axis=1).cpu().numpy(),0.0,1.0)
            AUC_val = roc_auc_score(y_true_bin, prob_preds)
                          
            outputs_val = DealWithOutputs(regression_or_classification,outputs_val)
            
            loss_val_weighted = torch.mean(loss_fn_val(outputs_val,targets_val)*val_weights) #should be AUC for bin case, and L1 (as now) for multy label
            loss_val = torch.mean(loss_fn_val(outputs_val,targets_val))
            
            
            Costs_val_weighted = np.append(Costs_val_weighted,loss_val_weighted.cpu().detach().numpy())
            Costs_val = np.append(Costs_val,loss_val.cpu().detach().numpy())
            
            if (loss_val.item() < min_loss_val): 
                min_name = 'ResNet_' + str(round(Costs_val[-1],3)) + '_' + str(round(AUC_val,2))
                torch.save(net.state_dict(), min_name  + '.pth' )
                min_loss_val -= 0.025
                save_flag = 1
            if  (AUC_val>max_auc_val): 
                min_name = 'ResNet_' + str(round(Costs_val[-1],3)) + '_' + str(round(AUC_val,2))
                torch.save(net.state_dict(), min_name  + '.pth' )
                max_auc_val += 0.01
                save_flag = 1
            if save_flag == 1:
                predicted_method_1 = torch.round(outputs_val).reshape(-1,)
                y_pred_1 = predicted_method_1.cpu().detach().numpy()
                cf_matrix_1 = confusion_matrix(y_true,y_pred_1)
                sio.savemat(min_name + '.mat', {"Costs": Costs, "Costs_val": Costs_val,'Costs_val_weighted': Costs_val_weighted,'cf_matrix_1':cf_matrix_1,'AUC':AUC_val})
        net.train()
        print('[%d, %5d] loss: %.3f  Loss_val: %.3f AUC: %.2f' %(Epoch + 1, batch_i + 1, Costs[-1],Costs_val[-1],AUC_val))

### Save net weights
print('Finished Training')
#end_name =  'ResNet_' + str(round(Costs_val[-1],3)) + '.pth' #manualy write: 
end_name = LastCheckPoint
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
    Probs = outputs_val/outputs_val.sum(axis=1,keepdims=True)
    prob_preds = np.clip(torch.sum(Probs[:,1:],axis=1).cpu().numpy(),0.0,1.0)
    AUC_val = roc_auc_score(y_true_bin, prob_preds)
    print('AUC on val set:',AUC_val)
    
    prob_preds_ensamble,_ = EnsamblePredForAUC(net,val_samples)
    prob_preds_ensamble = np.clip(prob_preds_ensamble.cpu().numpy(),0.0,1.0)
    AUC_val_ensamble = roc_auc_score(y_true_bin, prob_preds_ensamble)
    print('AUC on val set with ensamble:',AUC_val_ensamble)

    FixedProb = torch.clone(Probs)
    FixedProb[np.where(NumOfUniqValues > 19)[0],0] = 0
    Fixed_AUC_val = roc_auc_score(y_true_bin,  np.clip(torch.sum(FixedProb[:,1:],axis=1).cpu().numpy(),0.0,1.0))
    print('AUC on val set with UniqValues fix:',Fixed_AUC_val)

### Track 2 Eval
with torch.no_grad(): 
    outputs_val = DealWithOutputs(regression_or_classification,outputs_val)
    predicted_method_1 = torch.round(outputs_val).reshape(-1,)
    correct_method_1 = (predicted_method_1 == targets_val.reshape(-1,)).sum().item()
    print(f'Accuracy on val set: {100 * correct_method_1 // len(val_set)} %')

    predicted_method_2,Ensemble_out = EnsamblePred(net,val_samples)
    correct_method_2 = (predicted_method_2 == targets_val.reshape(-1,)).sum().item()
    print(f'Accuracy on val set with ensamble: {100 * correct_method_2 // len(val_set)} %')

    Fixed_Expectation = (FixedProb[:,0]*0 + FixedProb[:,1]*1 + FixedProb[:,2]*2 + FixedProb[:,3]*3).reshape(-1,1)
    predicted_method_3 = torch.round(Fixed_Expectation).reshape(-1,)
    correct_method_3 = (predicted_method_3 == targets_val.reshape(-1,)).sum().item()
    print(f'Accuracy on val set with UniqValues fix: {100 * correct_method_3 // len(val_set)} %')

### Confussion Matrix
y_pred_1 = predicted_method_1.cpu().detach().numpy()
y_pred_2 = predicted_method_2.cpu().detach().numpy()
y_pred_3 = predicted_method_3.cpu().detach().numpy()
cf_matrix_1 = confusion_matrix(y_true,y_pred_1)
cf_matrix_2 = confusion_matrix(y_true,y_pred_2)
cf_matrix_3 = confusion_matrix(y_true,y_pred_3)

### MAE score, after quantization
DistsMat = np.array([[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]])
TheRealThingINeed_1 =  np.trace(cf_matrix_1@DistsMat) #smaller is better
TheRealThingINeed_2 =  np.trace(cf_matrix_2@DistsMat) #smaller is better
TheRealThingINeed_3 =  np.trace(cf_matrix_3@DistsMat) #smaller is better
print('L1 Distance on val set is:',TheRealThingINeed_1,TheRealThingINeed_2,TheRealThingINeed_3)

sio.savemat('LossVals.mat', {"Costs": Costs, "Costs_val": Costs_val,'Costs_val_weighted': Costs_val_weighted,
                             'cf_matrix_1':cf_matrix_1,'cf_matrix_2':cf_matrix_2,'cf_matrix_3':cf_matrix_3,
                             'outputs_val':outputs_val.cpu().detach().numpy(),
                             'Ensemble_out':Ensemble_out.cpu().detach().numpy(),
                             'targets_val':targets_val.cpu().detach().numpy(),
                             'AUC':AUC_val})
### Only in notebook
# !cp $myfile /content/drive/MyDrive/ResNet101/
# !cp $myfile2 /content/drive/MyDrive/ResNet101/
# !cp $end_name /content/drive/MyDrive/ResNet101/
# !cp 'LossVals.mat' /content/drive/MyDrive/ResNet101/
# print(min_name)