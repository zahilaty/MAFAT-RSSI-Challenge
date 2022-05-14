# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:15:04 2022

@author: zahil
"""
#import os
import numpy as np
import torch
from torch.utils.data import Dataset
from helper_func import ExtractFeaturesFromVecs
import scipy.io as sio

##########################################################
   
class MyDataset(Dataset):

    def __init__(self,mat_file = 'Data\DataV2.mat',device = "cuda",Return1D = False):
        self.audio_mat = sio.loadmat(mat_file)["X"]
        self.annotations = sio.loadmat(mat_file)["Y"]
        self.weights = sio.loadmat(mat_file)["W"]        
        self.device = device
        self.Return1D = Return1D
        #self.transformation = transformation.to(self.device)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label = self.annotations[index]
        weight = self.weights[index]
        X = self.audio_mat[index,:,:] #2x360
        X = my_augmentations(X)
        signal = ExtractFeaturesFromVecs(X) #3x360
        signal = torch.tensor(signal,dtype=torch.float32)
        signal = signal.to(self.device)
        if not self.Return1D:
            signal = torch.unsqueeze(signal, 2) #2x360x1
        #signal = self.transformation(signal)
        return signal, label , weight
    
if __name__ == "__main__":
    MAT_FILE = 'Data\DataV2_mul.mat'

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    demod_ds = MyDataset(MAT_FILE,device,Return1D = False)
    print(f"There are {len(demod_ds)} samples in the dataset.")
    signal,label,weight = demod_ds[9]

##########################################################
import random

def my_augmentations(X):
    # 1) Anntenas flip 
    if random.random() > 0.5:
        X = np.flip(X,0)
    # 2) Time flip
    if random.random() > 0.5:
        X = np.flip(X,1)
    # 2) Add -2 to +2 dB bias for each channel
    X = X + np.random.randint(-2,3,(2,1))
    
    
    return X
