# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:15:04 2022

@author: zahil
"""
#import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio

##########################################################
# Random Augmentations:
    
# 1) AWGN to all spectogram 
# 2) Random shift in the time axis of about ~0.3 sec (done with RandomAffine)
# 3) Random shift in the frequency axis of about 1000Hz (done with RandomAffine)
#https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78

# class AddGaussianNoise(object):
#     def __init__(self, mean=0., std=3.): #This WGN is added to the spectogram (dB units) rather than the signal. It has nothing to do with the termal noise!!
#         self.std = std
#         self.mean = mean
        
#     def __call__(self, tensor):
#         device = tensor.device
#         return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class MyDataset(Dataset):

    def __init__(self,mat_file = 'Data\DataV1.mat',device = "cuda"):
        self.annotations = sio.loadmat(mat_file)["Y"]
        self.audio_mat = sio.loadmat(mat_file)["X"]
        self.device = device
        #self.transformation = transformation.to(self.device)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label = self.annotations[index]
        signal = torch.tensor(self.audio_mat[index,:],dtype=torch.float32) #2x360
        signal = signal.to(self.device)
        signal = signal - signal.mean() # A sort of augmentation..
        signal = torch.unsqueeze(signal, 2) #2x360x1
        #signal = self.transformation(signal)
        return signal, label
    
if __name__ == "__main__":
    MAT_FILE = 'Data\DataV1_mul.mat'

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    demod_ds = MyDataset(MAT_FILE,device)
    print(f"There are {len(demod_ds)} samples in the dataset.")
    signal,label = demod_ds[9]