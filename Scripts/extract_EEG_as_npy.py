# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:02:30 2019

Script for extracting epochs and category for saving as .npy files

@author: Greta
"""

#%% Imports
from sklearn import metrics
import numpy as np
from scipy.stats import zscore
import mne
import pickle
import argparse
import os

#%% Validate that these correspond to the GPU RT analysis scripts
#os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts')

# from EEG_classification import sigmoid, testEpoch,trainLogReg_cross_offline,trainLogReg_cross,trainLogReg_cross2,trainLogReg
# from EEG_analysis_RT import preproc1epoch, create_info_mne, applySSP,average_stable

os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts')


from EEG_analysis_offline import extractEpochs_tmin,extractCat

#%%

def findSubjectFiles(subjID,manual_n_it=False):
    ''' Loads an EEG file, a marker file, a category file, and an alpha value file (real-time classifier output values) from a chosen subject.
    
    # Input
    - subjID: string, subject ID containing two digits.
    - manual_n_it: int, manual entry for number of iterations
    
    # Output
    - EEGfile, markerFile, idxFile, alphaFile: strings of the names of the files.
    - n_it: int, number of iterations. Default is 5 (corresponds to 5 neurofeedback runs)
    
    '''
    
    if subjID == '07': 
        EEGfile = 'subject_07EEG_.csv'
        markerFile = 'subject_07marker_.csv'
    
    if subjID == '08': 
        EEGfile = 'subject_08EEG_.csv'
        markerFile = 'subject_08marker_.csv'
    
    if subjID == '11': 
        EEGfile = 'subject_11EEG_.csv'
        markerFile = 'subject_11marker_.csv'

    if subjID == '13': 
        EEGfile = 'subject_13_EEG_03-21-19_11-02.csv'
        markerFile = 'subject_13_marker_03-21-19_11-02.csv'
    
    if subjID == '14': 
        EEGfile = 'subject_14_EEG_03-21-19_09-48.csv'
        markerFile = 'subject_14_marker_03-21-19_09-48.csv'
        
    if subjID == '15': 
        EEGfile = 'subject_15_EEG_03-21-19_12-43.csv'
        markerFile = 'subject_15_marker_03-21-19_12-43.csv'

    if subjID == '16': 
        EEGfile = 'subject_16_EEG_03-28-19_08-43.csv'
        markerFile = 'subject_16_marker_03-28-19_08-43.csv'
        
    if subjID == '17': 
        EEGfile = 'subject_17_EEG_03-28-19_15-37.csv'
        markerFile = 'subject_17_marker_03-28-19_15-37.csv'
    
    if subjID == '18': 
        EEGfile = 'subject_18_EEG_03-28-19_14-08.csv'
        markerFile = 'subject_18_marker_03-28-19_14-08.csv'
        
    if subjID == '19': 
        EEGfile = 'subject_19_EEG_03-21-19_14-34.csv'
        markerFile = 'subject_19_marker_03-21-19_14-34.csv'
        
    if subjID == '21': 
        EEGfile = 'subject_21_EEG_04-02-19_08-45.csv'
        markerFile = 'subject_21_marker_04-02-19_08-45.csv'

    if subjID == '22': 
        EEGfile = 'subject_22_EEG_04-04-19_09-15.csv'
        markerFile = 'subject_22_marker_04-04-19_09-15.csv'
        
    if subjID == '23': 
        EEGfile = 'subject_23_EEG_04-02-19_16-48.csv'
        markerFile = 'subject_23_marker_04-02-19_16-48.csv'
        
    if subjID == '24': 
        EEGfile = 'subject_24_EEG_04-09-19_15-45.csv'
        markerFile = 'subject_24_marker_04-09-19_15-45.csv'
    
    if subjID == '25': 
        EEGfile = 'subject_25_EEG_04-02-19_14-09.csv'
        markerFile = 'subject_25_marker_04-02-19_14-09.csv'
        
    if subjID == '26': 
        EEGfile = 'subject_26_EEG_04-09-19_08-48.csv'
        markerFile = 'subject_26_marker_04-09-19_08-48.csv'
        
    if subjID == '27': 
        EEGfile = 'subject_27_EEG_03-28-19_12-45.csv'
        markerFile = 'subject_27_marker_03-28-19_12-45.csv'
        
    if subjID == '30': 
        EEGfile = 'subject_30_EEG_04-04-19_15-58.csv'
        markerFile = 'subject_30_marker_04-04-19_15-58.csv'
        
    if subjID == '31': 
        EEGfile = 'subject_31_EEG_04-11-19_08-40.csv'
        markerFile = 'subject_31_marker_04-11-19_08-40.csv'
        
    if subjID == '32': 
        EEGfile = 'subject_32_EEG_04-11-19_15-50.csv'
        markerFile = 'subject_32_marker_04-11-19_15-50.csv'
        
    if subjID == '33': 
        EEGfile = 'subject_33_EEG_04-09-19_17-06.csv'
        markerFile = 'subject_33_marker_04-09-19_17-06.csv'
        
    if subjID == '34': 
        EEGfile = 'subject_34_EEG_04-11-19_14-29.csv'
        markerFile = 'subject_34_marker_04-11-19_14-29.csv'  
    
    idxFile = 'createIndices_' + subjID + '_day_2.csv'
    alphaFile = 'alpha_subjID_' + subjID + '.csv' 
    
    if manual_n_it is None:
        n_it = 5
    if manual_n_it is not None:
        n_it = manual_n_it
        
    print(EEGfile)
        
    return EEGfile, markerFile, idxFile, alphaFile, n_it



def analyzeOffline(subjID):
    '''
    
    
    '''
    # Locate files
    data_dir = 'P:\\Research2018_2019\\DTU_closed_loop\\closed_loop_data\\'+subjID+'\\'
    os.chdir(data_dir)
    
    print(data_dir)
    
    # Initialize conditions for preprocessing of epochs (preproc1epoch from EEG_analysis_RT)
    reject_ch = 1 # Rejection of nine predefined channels
    reject = None # Rejection of channels, either manually defined or based on MNE analysis
    mne_reject = 0 
    flat = None # Input for MNE rejection
    bad_channels = None # Input for manual rejection of channels
    opt_detrend = 1 # Temporal EEG detrending (linear)
    
    if reject_ch == 1:
        n_channels = 23
    if reject_ch == 0:
        n_channels = 32
    
    EEGfile, markerFile, idxFile, alphaFile, n_it = findSubjectFiles(subjID,manual_n_it=None)

    # Extract epochs from EEG data
    prefilter = 0
    
    if subjID in ['07','08','11']:
        n_samples_fs500 = 550 # Number of samples to extract for each epoch, sampling frequency 500
        n_samples_fs100 = int(550/5) # Number of samples, sampling frequency 100 (resampled)
    else:
        n_samples_fs500 = 450 
        n_samples_fs100 = int(450/5) 
    
    e = extractEpochs_tmin(EEGfile,markerFile,prefilter=prefilter,marker1=0,n_samples=n_samples_fs500)
    cat = extractCat(idxFile,exp_type='fused')
    
    print('about to save')
    np.save('EEG_epochs_subjID_'+str(subjID)+'.npy', e)
    np.save('y_categories_subjID_'+str(subjID)+'.npy', cat)


#%% Iterate through participants
# subs = ['08','11','13','14','15','16','17','18','19',\
#           '21','22','23','24','25','26','27','30','31','32','33','34']   
    
subs = ['17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']   
    
for subjID in subs:
    print(subjID)
    analyzeOffline(subjID)    