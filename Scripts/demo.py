# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:25:47 2019

@author: Greta
"""
#%% Validate that these correspond to the GPU RT analysis scripts
#os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts')

from EEG_classification import sigmoid, testEpoch,trainLogReg_cross_offline,trainLogReg_cross,trainLogReg_cross2,trainLogReg
from EEG_analysis_RT import preproc1epoch, create_info_mne, applySSP,average_stable
from EEG_analysis_offline import extractEpochs_tmin,extractCat,extractAlpha,applySSP_forplot

#%%

os.chdir('P:\\closed_loop_data\\90_demo\\')

EEGfile = 'subject_90_EEG_04-23-19_10-24.csv'
markerfile = 'subject_90_EEG_04-23-19_10-24.csv'
idxFile = 'createIndices_90_day_2.csv'
alphaFile = 'alpha_subjID_90.csv' 
n_it = 2

#%% Test RT alpha per run
alpha,marker = extractAlpha(alphaFile)
above_chance = len(np.where((np.array(alpha)>0.5))[0])/len(alpha) 

alpha_per_run = np.zeros((n_it))
j = 0

for ii in range(n_it):
    alpha_per_run[ii] = len(np.where((np.array(alpha[j:j+200])>0.5))[0])/200 
    j += 200


clf_output = np.load('clf_out.npy')

