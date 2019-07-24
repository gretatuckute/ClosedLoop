# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:05:43 2019

@author: Greta
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
from scipy import stats
from scipy.signal import detrend
import pandas as pd
import os 

os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Documents\\')
fileName = 'event_feedback.xlsx'

colnames = ['Timestamp', 'subjID', 'age', 'gender', 'control', 'strategy', 'attention', 'cross', 'comments']
data = pd.read_excel(fileName, names=colnames)
subjIDs = data.subjID.tolist()
control = data.control.tolist()
age = data.age.tolist()
attention = data.attention.tolist()
cross = data.cross.tolist()



sub_fb = [7,8,26,27,30,11,13,14,16,19,22]
sub_c = [17,18,23,31,34,15,21,24,25,32,33]

fb_control = []
c_control = []

fb_attention = []
c_attention = []

fb_cross = []
c_cross = []

for idx,sub in enumerate(subjIDs):
    if sub in sub_fb:
        fb_control.append(control[idx])
        fb_attention.append(attention[idx])
        fb_cross.append(cross[idx])
    else:
        c_control.append(control[idx])
        c_attention.append(attention[idx])
        c_cross.append(cross[idx])

stats.ttest_ind(fb_control,c_control)

np.mean(c_attention)
np.mean(fb_attention)

# Age
age_m = np.mean(age)
np.min(age),np.max(age)