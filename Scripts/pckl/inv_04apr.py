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

os.chdir('C:\\Users\\Greta\\Documents\\GitLab\\project\\Python_Scripts\\18Mar_RT_files\\')



# Extract values from dict
#dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
#
#wanted_keys = ("alpha_fromfile_overall","alpha_fromfile_run")
#
#result = dictfilt(pkl_arr[0], wanted_keys)
#
#g = list(result.values())
    
#%%

with open('18Mar_subj_07.pkl', "rb") as fin:
    sub07 = (pickle.load(fin))[0]

with open('18Mar_subj_08.pkl', "rb") as fin:
    sub08 = (pickle.load(fin))[0]

with open('18Mar_subj_11.pkl', "rb") as fin:
    sub11 = (pickle.load(fin))[0]
    
with open('18Mar_subj_92.pkl', "rb") as fin:
    sub92 = (pickle.load(fin))[0]

with open('18Mar_subj_95.pkl', "rb") as fin:
    sub95 = (pickle.load(fin))[0]

with open('18Mar_subj_97.pkl', "rb") as fin:
    sub97 = (pickle.load(fin))[0]
    
#%%
# Plot showing where the correct predictions are located pr. run.

n_it=5

pred_run = np.reshape(sub11['correct_NFtest_pred'],[n_it,200])

for run in range(n_it):
    plt.figure(run)
    plt.bar(np.arange(200),pred_run[run,:])