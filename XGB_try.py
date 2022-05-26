# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:45:13 2022

@author: zahil
"""

###################################################################
import numpy as np
import torch
from torch.utils.data import DataLoader
from helper_func import MyResNet18,Identity
from TrainAux import MyDataset,my_augmentations
import scipy.io as sio
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
#from xgboost.sklearn import XGBRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import graphviz
import xgboost as xgb

###################################################################
### Init
DataFromResNetOrMAFAT = 1
n_estimators = 300
max_depth = 5
learning_rate = 0.1

###################################################################
# Data From Resnet code (remove the Fc)
if DataFromResNetOrMAFAT == 0:
    mat_file = 'Data\DataV2_mul.mat'
    my_ds = MyDataset(mat_file,'cuda',Return1D = False,augmentations = True) #calling the after-processed dataset
    my_ds_val = MyDataset(mat_file,'cuda',Return1D = False,augmentations = False) #calling the after-processed dataset
    l1 = np.reshape(sio.loadmat(mat_file)["l1"],(-1,)) # we need to save the indexes so we wont have data contimanation
    l2 = np.reshape(sio.loadmat(mat_file)["l2"],(-1,))
    train_set = torch.utils.data.Subset(my_ds, l1)
    val_set = torch.utils.data.Subset(my_ds_val, l2)
    train_dataloader = DataLoader(train_set, batch_size=100,drop_last=False,shuffle=False)
    val_dataloader = DataLoader(val_set, batch_size=val_set.dataset.__len__())
    
    net  = MyResNet18(InputChannelNum=4,IsSqueezed=0,LastSeqParamList=[512,32,4],pretrained=True).cuda()
    LastCheckPoint = 'Checkpoints\\16_05_C\\long\\ResNet_0.547.pth' #None ## A manual option to re-train
    net.load_state_dict(torch.load(LastCheckPoint)) 
    net.fc = Identity()
    
    X_train = np.empty((0,512))
    X_test = np.empty((0,512))
    y_train = np.empty((0,1),int)
    y_test = np.empty((0,1),int)
    
    # lines if I want static memory
    # np.zeros(len(train_set),512)
    # np.zeros(len(val_set),512)
    # start_ind = 0
    # codes[start_ind:(start_ind+batch_size),:] = outputs to numpy
    # start_ind = (start_ind+batch_size) validate +-1
    
    with torch.no_grad():
        for batch_i, [batch,labels,weights] in enumerate(train_dataloader):
            if not(batch_i%10): print(batch_i)
            outputs = net(batch)
            assert(outputs.dim()==2)
            assert(outputs.shape[0]==batch.shape[0])
            assert(outputs.shape[1]==512)
            assert(labels.shape[0] == batch.shape[0])
            X_train = np.vstack((X_train, outputs.cpu().detach().numpy()))
            y_train = np.vstack((y_train, labels.reshape(-1,1)))
            
        for batch_i, [batch,labels,weights] in enumerate(val_dataloader):
            outputs = net(batch)
            assert(outputs.dim()==2)
            assert(outputs.shape[0]==batch.shape[0])
            assert(outputs.shape[1]==512)
            X_test = np.vstack((X_test, outputs.cpu().detach().numpy()))
            y_test = np.vstack((y_test, labels.reshape(-1,1)))
    
    print('size of X in MB:',X_train.nbytes/1e6)

###################################################################
if DataFromResNetOrMAFAT == 1:
    MAT = sio.loadmat('Data\FromMafat.mat')
    X = MAT['X']
    Y = MAT['Y'].transpose()
    train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)

###################################################################
y_train = y_train.ravel()
y_test = y_test.ravel()

####################################################################
### Sikit gradient boosting
model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                  loss = 'absolute_error',random_state=0,verbose=0)
#model = MLPRegressor((20,5),verbose=True)
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds)) 
mae = mean_absolute_error(y_test, preds)
print('RMSE:',rmse,'  MAE:',mae)

test_score = np.zeros((n_estimators,), dtype=np.float64)
for i, y_pred in enumerate(model.staged_predict(X_test)):
    test_score[i] = model.loss_(y_test, y_pred)
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.plot(np.arange(n_estimators) + 1,model.train_score_,"b-",label="Training Set Deviance")
plt.plot(np.arange(n_estimators) + 1,test_score,"r-", label="Test Set Deviance")
plt.title("Train and Val loss - Sikit:" + str(mae) );plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations");plt.ylabel("Deviance")
fig.tight_layout();plt.show()

feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos,sorted_idx)
plt.title("Feature Importance (MDI)")


# result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
# sorted_idx = result.importances_mean.argsort()
# plt.subplot(1, 2, 2)
# plt.boxplot(
#     result.importances[sorted_idx].T,
#     vert=False,labels=sorted_idx,)
# plt.title("Permutation Importance (test set)")
# fig.tight_layout()
# plt.show()

###################################################################
### XGB boosting
# It does not have MAE! https://stackoverflow.com/questions/38556983/xgboost-increasing-training-error-mae
evalset = [(X_train, y_train), (X_test,y_test)]
xg_reg = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = learning_rate,
                      max_depth = max_depth, alpha = 10, n_estimators = n_estimators,verbosity=0,
                      eval_metric=mean_absolute_error,random_state=0)
xg_reg.fit(X_train,y_train,eval_set=evalset) #eval_metric=mean_absolute_error
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds)) #TODO: see if it is the same: mean_absolute_error
mae = mean_absolute_error(y_test, preds)
print('RMSE:',rmse,'  MAE:',mae)

# plot learning curves
results = xg_reg.evals_result()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.plot(np.arange(n_estimators) + 1,results['validation_0']['rmse'],"b-",label='Training Set Deviance')
plt.plot(np.arange(n_estimators) + 1,results['validation_1']['rmse'],"r-",label='Test Set Deviance')
plt.title("Train and Val loss - XGB: " + str(mae));plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations");plt.ylabel("Deviance")
fig.tight_layout()
plt.show()

feature_importance = xg_reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos,sorted_idx)
plt.title("Feature Importance (MDI)")

# xgb.plot_tree(xg_reg)
# plt.rcParams['figure.figsize'] = [50, 10]
# plt.show()