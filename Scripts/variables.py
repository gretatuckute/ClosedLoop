# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:09:06 2019

@author: Greta
"""
from sklearn.linear_model import LinearRegression
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

#%% Directories
saveDir = 'P:\\closed_loop_data\\beh_analysis\\'
#saveDir = 'P:\\closed_loop_data\\beh_analysis\\V3_150_1000\\'
EEGDir = 'P:\\closed_loop_data\\offline_analysis_pckl\\' 
scriptsDir = 'C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\'

#%% Variables
subjID_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']
subjID_NF = ['07','08','11','13','14','16','19','22','26','27','30']
subjID_C = ['15','17','18','21','23','24','25','31','32','33','34']

# For matching
NF_group = ['07','08','11','13','14','16','19','22','26','27','30']
C_group = ['17','18','15','24','21','33','25','32','34','23','31']

# Model fits
lm = LinearRegression()

#%% Load EEG accuracy arrays

# Load RT accuracy np array
subsAll_RT_acc = np.load(scriptsDir+'subsAll_RT_acc.npy') # From EEG inv 18April
subsNF_RT_acc = np.load(scriptsDir+'subsNF_RT_acc.npy').flatten()
subsC_RT_acc = np.load(scriptsDir+'subsC_RT_acc.npy').flatten()

# LOBO and LORO
subsAll_LOBO = np.load(scriptsDir+'subsAll_LOBO.npy')
subsAll_LORO = np.load(scriptsDir+'subsAll_LORO_09May.npy')

subsNF_LORO = np.load(scriptsDir+'subsNF_LORO_09May.npy') # omit 09 May if old one
subsC_LORO = np.load(scriptsDir+'subsC_LORO_09May.npy')

subsNF_LOBO = np.load(scriptsDir+'subsNF_LOBO.npy') 
subsC_LOBO = np.load(scriptsDir+'subsC_LOBO.npy')

# Alphas
subsNF_meanAlphas = np.load(scriptsDir+'subsNF_meanAlphas.npy')

#%% Plot styles
plt.style.use('seaborn-pastel')

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rc('text',usetex=True)
matplotlib.rc('font',family='serif')
plt.rcParams.update({'font.size':12})
matplotlib.rcParams['grid.alpha'] = 1
matplotlib.rcParams['xtick.labelsize'] = 'medium'
matplotlib.rcParams['ytick.labelsize'] = 'medium'