# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:12:38 2019

@author: Greta
"""

import pickle
from matplotlib import pyplot as plt
import os

saveDir = 'P:\\closed_loop_data\\beh_analysis\\'
os.chdir(saveDir)


# Extract values from dict
#dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
#
#wanted_keys = ("alpha_fromfile_overall","alpha_fromfile_run")
#
#result = dictfilt(pkl_arr[0], wanted_keys)
#
#g = list(result.values())
    
#%%

with open('Beh_subjID_02.pkl', "rb") as fin:
    sub02 = (pickle.load(fin))[0]

with open('Beh_subjID_03.pkl', "rb") as fin:
    sub03 = (pickle.load(fin))[0]

with open('Beh_subjID_15.pkl', "rb") as fin:
    sub15 = (pickle.load(fin))[0]

with open('Beh_subjID_07.pkl', "rb") as fin:
    sub07 = (pickle.load(fin))[0]
    
with open('Beh_subjID_08.pkl', "rb") as fin:
    sub08 = (pickle.load(fin))[0]
