# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:22:27 2022

@author: zahil
"""
from zipfile import ZipFile

# create a ZipFile object
zipObj = ZipFile('submission.zip', 'w')
# Add multiple files to the zip
zipObj.write('model.py')
zipObj.write('helper_func.py')
#zipObj.write('CompleteModel.pt')
zipObj.write('sample_submission\metadata',arcname='metadata')
zipObj.write('Checkpoints\\14_05\\ResNet_0.685.pth',arcname='NetWeights.pth')
# close the Zip File
zipObj.close()

#######################################################################
# Create a ZipFile Object and load sample.zip in it
with ZipFile('submission.zip', 'r') as zipObj:
   # Extract all the contents of zip file in different directory
   zipObj.extractall('temp')
   zipObj.close()
   
#######################################################################
print('Starting tests')
import os
os.chdir('temp')
import pandas as pd
X = pd.read_csv('..\Data\one_window_for_demo.csv')
from model import *
M = model()
M.load('')
Y_test=[]
unique_windows = list(set(X.Num_Window))
for window in unique_windows:
    X_test_window = X.loc[X['Num_Window'] == window]
    X_test_window.drop('Num_Window', axis=1, inplace=True)
    Y_test.append(M.predict(X_test_window))

print(f'Occupancy prediction: {round(Y_test[0],3)}')

##############################
#For the times when we are using old torchvision without adaptive avargepooling
#net  = MyResNet18(InputChannelNum=3,IsSqueezed=0,LastSeqParamList=[512,32,4],pretrained=True).cuda()
#net.load_state_dict(torch.load('Checkpoints/ResNet_0.515.pth'))
#torch.save(net.cpu(),'CompleteModel.pt')
#net2.fc.layer0[1].weight #make sure no cuda