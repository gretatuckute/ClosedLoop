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

with open('04April_V3_subj_24.pkl', "rb") as fin:
    sub24 = (pickle.load(fin))[0]

with open('04April_V3_subj_25.pkl', "rb") as fin:
    sub25 = (pickle.load(fin))[0]
    
with open('04April_V3_subj_26.pkl', "rb") as fin:
    sub26 = (pickle.load(fin))[0]
    
with open('04April_V3_subj_27.pkl', "rb") as fin:
    sub27 = (pickle.load(fin))[0] 
    
with open('04April_V3_subj_30.pkl', "rb") as fin:
    sub30 = (pickle.load(fin))[0] 
    
with open('04April_V3_subj_33.pkl', "rb") as fin:
    sub33 = (pickle.load(fin))[0]

#%% 
wanted_keysV1 = ("RT_test_acc_corr","RT_test_acc_corr_run","train_acc_stable")#,"train_acc_stable_NF","train_acc_stable_test")
wanted_keysV2 = ("RT_test_acc_corr","RT_test_acc_corr_run","train_acc_stable_corr")#,"train_acc_stable_NF","train_acc_stable_test")

# Figure out three first subs!!!!

subLst = [sub13,sub14,sub15,sub16,sub17,sub18,sub19,\
          sub21,sub22,sub23,sub24,sub25,sub26,sub27,sub30,sub33]

subs_550 = [sub07_550, sub08_550, sub11_550]

allSubs = []
for sub in subLst:
    result = dictfilt(sub, wanted_keysV2)
    #print(result)
    g = list(result.values())    
    allSubs.append(g)
    
allSubs_550 = []

for sub in subs_550:
    result = dictfilt(sub, wanted_keysV1)
    #print(result)
    g = list(result.values())    
    allSubs_550.append(g)
    
#%%
    
# RT
allSubsRT_550 = []
allSubsRTrun_550 = []

# Offline
allSubsOFF_550 = []


for sub in allSubs_550:
    RT_acc = sub[0]
    RT_acc_run = sub[1]
    OFF_acc = sub[2]

    allSubsRT_550.append(RT_acc)
    allSubsRTrun_550.append(RT_acc_run)
    allSubsOFF_550.append(OFF_acc)

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
#    OFF_acc_all = sub[3]
#    OFF_acc_stabletest = sub[4]
    allSubsRT.append(RT_acc)
    allSubsRTrun.append(RT_acc_run)
    allSubsOFF.append(OFF_acc)
#    allSubsOFF_allblocks.append(OFF_acc_all)
#    allSubsOFF_stabletest.append(OFF_acc_stabletest)
 
#%% Add 550 subs
allSubsRT2 = allSubsRT+allSubsRT_550
allSubsRTrun2 = allSubsRTrun + allSubsRTrun_550
allSubsOFF2 = allSubsOFF + allSubsOFF_550

np.mean(allSubsRT2)
np.min(allSubsRTrun2)
np.max(allSubsRTrun2)
    
np.mean(allSubsRTrun2) # same
np.mean(allSubsOFF2)   #0.578, 0.625, 0.627

#%%
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

