# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:09:06 2019

@author: Greta
"""
from sklearn.linear_model import LinearRegression
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pickle
import random

#%%
matplotlib.rcParams['figure.constrained_layout.use'] = True
matplotlib.rcParams['figure.constrained_layout.use'] = False

#%% Directories
saveDir = 'P:\\closed_loop_data\\beh_analysis\\'
# saveDir = 'P:\\closed_loop_data\\beh_analysis\\V3_200_1200\\'

figDir = 'C:\\Users\\Greta\\Desktop\\closed_loop\\RESULTS\\EEG\\'
EEGDir = 'P:\\closed_loop_data\\offline_analysis_pckl\\' 
scriptsDir = 'C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\'
npyDir = 'C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Documents\\npy\\'

#%% Variables
subjID_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']
subjID_NF = ['07','08','11','13','14','16','19','22','26','27','30']
subjID_C = ['15','17','18','21','23','24','25','31','32','33','34']

# Subj 11 and 15 removed
subjID_20 = ['07','08','13','14','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

# For reorganizing
idxLst = [2,0,1,4,9,3,6,10,7,5,8]

# For matching
NF_group = ['07','08','11','13','14','16','19','22','26','27','30']
C_group = ['17','18','15','24','21','33','25','32','34','23','31']

# Model fits
lm = LinearRegression()

n_it = 5

#%% Variables for EEG decoding plots, comparing NF and C
color = ['tomato']*11 + ['dodgerblue']*11
color_uncor = ['brown']*11 + ['navy']*11
sub_axis = subjID_NF + subjID_C
sub_axis_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

#%% Load EEG accuracy arrays

# Load RT accuracy np array
subsAll_RT_acc = np.load(npyDir+'subsAll_RT_acc.npy') # From EEG inv 18April
subsNF_RT_acc = np.load(npyDir+'subsNF_RT_acc.npy').flatten()
subsC_RT_acc = np.load(npyDir+'subsC_RT_acc.npy').flatten()

# LOBO and LORO
subsAll_LOBO = np.load(npyDir+'subsAll_LOBO.npy')
subsAll_LORO = np.load(npyDir+'subsAll_LORO_09May.npy')

subsNF_LORO = np.load(npyDir+'subsNF_LORO_09May.npy') # omit 09 May if old one
subsC_LORO = np.load(npyDir+'subsC_LORO_09May.npy')

subsNF_LOBO = np.load(npyDir+'subsNF_LOBO.npy') 
subsC_LOBO = np.load(npyDir+'subsC_LOBO.npy')

# Alphas
subsNF_meanAlphas = np.load(npyDir+'subsNF_meanAlphas.npy')

# Load d_all2
with open(npyDir+'d_all2.pkl', "rb") as fin:
    d_all2 = (pickle.load(fin))

# d_all3
with open(npyDir+'d_all3.pkl', "rb") as fin:
    d_all3 = (pickle.load(fin))

#%% Load stats dictionaries for all subjects
# 3005 includes errorrate, RER, A, ARER. Extracted as 150-150 ms.

with open(npyDir+'statsDay_all_3005.pkl', "rb") as fin:
    statsDay_all = (pickle.load(fin))

with open(npyDir+'statsBlock_all_3005.pkl', "rb") as fin:
    statsBlock_all = (pickle.load(fin))

with open(npyDir+'statsBlockDay2_all_3005.pkl', "rb") as fin:
    statsBlockDay2_all = (pickle.load(fin))
    
# Extracted at 200-1200 ms
# with open(npyDir+'statsDay_all_200_1200.pkl', "rb") as fin:
#     statsDay_all = (pickle.load(fin))

# with open(npyDir+'statsBlock_all_200_1200.pkl', "rb") as fin:
#     statsBlock_all = (pickle.load(fin))

# with open(npyDir+'statsBlockDay2_all_200_1200.pkl', "rb") as fin:
#     statsBlockDay2_all = (pickle.load(fin))    


#%% Plot styles
plt.style.use('seaborn-pastel')

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rc('text',usetex=True)
# matplotlib.rc('text',usetex=False)
matplotlib.rc('font',family='serif')
plt.rcParams.update({'font.size':12})
matplotlib.rcParams['grid.alpha'] = 1
matplotlib.rcParams['xtick.labelsize'] = 'medium'
matplotlib.rcParams['ytick.labelsize'] = 'medium'

