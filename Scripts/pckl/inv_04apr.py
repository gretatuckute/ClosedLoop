# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:19:05 2019

@author: Greta
"""

import pickle
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import os
import numpy as np
    
#%% # Extracted 550 samples

os.chdir('C:\\Users\\Greta\\Documents\\GitLab\\project\\Python_Scripts\\18Mar_RT_files\\')

with open('18Mar_subj_07.pkl', "rb") as fin:
    sub07_550 = (pickle.load(fin))[0]
    
with open('18Mar_subj_08.pkl', "rb") as fin:
    sub08_550 = (pickle.load(fin))[0]
    
with open('18Mar_subj_11.pkl', "rb") as fin:
    sub11_550 = (pickle.load(fin))[0]

#%% 
os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\pckl\\pckl_V1\\')
    
with open('04April_subj_07.pkl', "rb") as fin:
    sub07new = (pickle.load(fin))[0] 
    
# Sub07 and sub07new are identical. Sub07 uses the new, updated scripts.

#%%
os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\pckl\\')

with open('04April_V2_subj_07.pkl', "rb") as fin: # Not matching samples
    sub07 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_08.pkl', "rb") as fin: # Not matching samples
    sub08 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_11.pkl', "rb") as fin: # Not matching samples
    sub11 = (pickle.load(fin))[0] 

with open('04April_V2_subj_13.pkl', "rb") as fin:
    sub13 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_14.pkl', "rb") as fin:
    sub14 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_15.pkl', "rb") as fin:
    sub15 = (pickle.load(fin))[0] 
    
# V3 files

with open('04April_V3_subj_16.pkl', "rb") as fin: # Not matching samples
    sub16 = (pickle.load(fin))[0] 
    
with open('04April_V3_subj_17.pkl', "rb") as fin: # Not matching samples
    sub17 = (pickle.load(fin))[0] 

with open('04April_V3_subj_18.pkl', "rb") as fin:
    sub18 = (pickle.load(fin))[0] 
    
with open('04April_V3_subj_19.pkl', "rb") as fin:
    sub19 = (pickle.load(fin))[0] 
    
with open('04April_V3_subj_21.pkl', "rb") as fin:
    sub21 = (pickle.load(fin))[0] 
    
# Not yet running: (Ran Monday 8th night)
    
with open('04April_V3_subj_22.pkl', "rb") as fin: 
    sub22 = (pickle.load(fin))[0] 
    
with open('04April_V3_subj_23.pkl', "rb") as fin: 
    sub23 = (pickle.load(fin))[0] 

with open('04April_V3_subj_25.pkl', "rb") as fin:
    sub25 = (pickle.load(fin))[0] 
    
with open('04April_V3_subj_27.pkl', "rb") as fin:
    sub27 = (pickle.load(fin))[0] 
    
with open('04April_V3_subj_30.pkl', "rb") as fin:
    sub30 = (pickle.load(fin))[0] 


#%% Extract RT offline accuracy and training accuracy
    
# I want RT_test_acc_corr, RT_test_acc_corr_run, EVT: conf_corr (or RT_conf_corr for newer versions), scene_acc (RT_scene_acc), face_acc (RT_face_acc)
# For training accuracy: train_acc_stable_corr
    
# Extract values from dict
#dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
#wanted_keysV2 = ("RT_test_acc_corr","RT_test_acc_corr_run","train_acc_stable_corr","train_acc_stable_NF","train_acc_stable_test")
#
#result = dictfilt(sub07new2, wanted_keysV2)
#
#g = list(result.values())    

#%% 
wanted_keysV2 = ("RT_test_acc_corr","RT_test_acc_corr_run","train_acc_stable_corr","train_acc_stable_NF","train_acc_stable_test")

# Figure out three first subs!!!!

subLst = [sub07, sub08, sub11, sub13,sub14,sub15,sub16,sub17,sub18,sub19,\
          sub21,sub22,sub23,sub25,sub27,sub30]
    
# subLstt = [sub13,sub14] #test

allSubs = []

for sub in subLst:
    result = dictfilt(sub, wanted_keysV2)
    #print(result)
    g = list(result.values())    
    allSubs.append(g)
    

#%% Analyze allSubs
    
# RT
allSubsRT = []
allSubsRTrun = []

# Offline
allSubsOFF = []
allSubsOFF_allblocks = []
allSubsOFF_stabletest = []

for sub in allSubs:
    RT_acc = sub[0]
    RT_acc_run = sub[1]
    OFF_acc = sub[2]
    OFF_acc_all = sub[3]
    OFF_acc_stabletest = sub[4]
    allSubsRT.append(RT_acc)
    allSubsRTrun.append(RT_acc_run)
    allSubsOFF.append(OFF_acc)
    allSubsOFF_allblocks.append(OFF_acc_all)
    allSubsOFF_stabletest.append(OFF_acc_stabletest)
    
allSubsRT_mean = np.mean(allSubsRT)     #0.59, 0.605, 0.61
allSubsRT_min = np.min(allSubsRTrun)
allSubsRT_max = np.max(allSubsRTrun)

# Reality check
np.mean(allSubsRTrun) # same

# Offline train acc
allSubsOFF_mean = np.mean(allSubsOFF)   #0.578, 0.625, 0.627
allSubsOFF_min = np.min(allSubsOFF)
allSubsOFF_max = np.max(allSubsOFF) #0.66



np.mean(allSubsOFF_allblocks) # 0.60, 0.64, 0.64
np.mean(allSubsOFF_stabletest) # 0.597, 0.63, 0.63

#%%
# Plot showing where the correct predictions are located pr. run.

#n_it=5
#
#pred_run = np.reshape(sub11['correct_NFtest_pred'],[n_it,200])
#
#for run in range(n_it):
#    plt.figure(run)
#    plt.bar(np.arange(200),pred_run[run,:])