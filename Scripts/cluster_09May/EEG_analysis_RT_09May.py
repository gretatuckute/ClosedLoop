# -*- coding: utf-8 -*-

"""
Created on Tue Jan 22 13:26:22 2019

Functions for working with EEG in Python: computing and analyzing variance of SSP projection
vectors, preprocessing of epoched EEG data (filtering, baseline correction, rereferencing, resampling)

@author: Greta & Sofie
"""

#### Imports ####

import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
import pandas as pd
import os 
import mne
from scipy.signal import detrend

LP=40
HP=0
phase='zero-double'
#### FUNCTIONS ####

def indices(a, func):
    '''
    Returns indices of array. Equivalent of MATLAB find.
    
    '''
    
    return [i for (i, val) in enumerate(a) if func(val)]


def analyzeVar(p, threshold):
    '''
    Analyzes SSP projection vectors and only uses projection is variance explained is above a defined threshold.
    
    # Input
    - p: list of variance explained by each SSP projection vector.
    - Threshold: value between 0 and 1.
    
    # Output
    - Indices of SSP projection vectors above the defined threshold.
    
    '''
    
    threshold_idx = []
    for count, val in enumerate(p):
        if val > threshold:
            print(val)
            threshold_idx.append(count)
            
    return threshold_idx

def computeSSP(EEG,info,threshold):
    '''
    Computes SSP projections of epoched EEG data (bandpass filtered and resampled to 100Hz). 
    
    # Input
    - EEG: Epoched and processed EEG
    - info: mne info struct
    - threshold: Value between 0 and 1, only uses the SSP projection vector if it explains more than the pre-defined threshold.
    
    # Output
    - Array of SSP projection vectors above a pre-defined threshold (variance explained).
    
    '''

    print('Computing SSP based on epochs')

    projs = mne.compute_proj_epochs(EEG,n_eeg=10,n_jobs=1, verbose=True)

    p = [projs[i]['explained_var'] for i in range(10)]

    # If variance explained is above a certain threshold, use the SSP vector for projection
    threshold_idx = analyzeVar(p, threshold)

    threshold_projs = [] # List with projections above a chosen threshold
    for idx in threshold_idx:
        threshold_projs.append(projs[idx])
        
    return threshold_projs

def applySSP(EEG,info,threshold=0.1,):
    '''
    Computes and applies SSP projections of epoched EEG data (bandpass filtered and resampled to 100Hz). 
    
    # Input
    - EEG: Epoched and processed EEG
    - info: mne info struct
    - threshold: Value between 0 and 1, only uses the SSP projection vector if it explains more than the pre-defined threshold.
    
    # Output
    - EEG after SSP
    
    '''    
    EEG = mne.EpochsArray(EEG, info,baseline=None)
    projs = computeSSP(EEG,info,threshold)
    EEG.add_proj(projs)
    EEG.apply_proj()
    EEG = EEG.get_data()
    return projs,EEG

def average_stable(stable):
    '''
    Averaging of neighboring epochs in pairs of two, such that a epoch_c is updated to: (epoch_c+epoch_{c+1})/2
    '''
    stable_mid = (stable[:-1]+stable[1:])/2
    stable_2mean = np.zeros(stable.shape)
    stable_2mean[0] = stable[0]
    stable_2mean[1:] = stable_mid
    return stable_2mean

def removeEpochs(EEG,info,interpolate=0):
    '''
    define ranges for bad epochs and apply (currently not used)
    '''
    # estimate common peak-to-peak for channels and trials
    minAmp=np.min(EEG,axis=2)
    maxAmp=np.max(EEG,axis=2)    
    peak2peak=maxAmp-minAmp
    reject_thres=np.mean(peak2peak)+4*np.std(peak2peak)*4
    reject=dict(eeg=reject_thres)
    flat_thres=np.mean(peak2peak)/100
    flat=dict(eeg=flat_thres)
    no_trials=EEG.shape[0]
    
    EEG = mne.EpochsArray(EEG, info,baseline=None)    
    epochs_copy=EEG.copy()
    epochs_copy.drop_bad(reject=reject, flat=flat, verbose=None)
    log=epochs_copy.drop_log
    if any(log):
        not_blank=np.concatenate([x for x in (log) if x])
        bad_channels, bad_counts = np.unique(not_blank, return_counts=True)
    #bad_channels=[bad_channels[a] for a in range(len(bad_channels))]
        p=0.05
        bad_above_thres=[]
        while p==0.05 or len(bad_above_thres)>5:
            thres=p*no_trials
            bad_above_thres=bad_channels[bad_counts>thres]
            p=p+0.05
   
        if len(bad_above_thres):
            EEG.info['bads']=bad_above_thres
            print('Dropping channels:',bad_above_thres)
            if interpolate==1:
                EEG.interpolate_bads(reset_bads=True)
            else:
                EEG.drop_channels(bad_above_thres)
                print('Not interpolating bad channels')
            # repeat looking for bad epochs
            epochs_copy=EEG.copy()
            epochs_copy.drop_bad(reject=reject, flat=flat, verbose=None)
            log=epochs_copy.drop_log
            
        bad_epochs=[i for i,x in enumerate(log) if x]  
        if len(bad_epochs)>0.20*no_trials:
            reject_thres=reject_thres*2
            reject=dict(eeg=reject_thres)
            flat_thres=flat_thres/2
            flat=dict(eeg=flat_thres)
            epochs_copy=EEG.copy()
            epochs_copy.drop_bad(reject=reject, flat=flat, verbose=None)
            log=epochs_copy.drop_log
            bad_epochs=[i for i,x in enumerate(log) if x]  
            if len(bad_epochs)>0.60*no_trials:
                print('More than 60% are bad, keeping all epochs')
                log=[]
            
        EEG.drop(bad_epochs,reason='Reject')
    else:
        bad_epochs=[]
        bad_above_thres=[]
        
    #EEG.set_eeg_reference(verbose=False)
    # Apply baseline after rereference
    #EEG.apply_baseline(baseline=(None,0),verbose=False)
    EEG=EEG.get_data()

    return EEG, reject,flat,bad_epochs,bad_above_thres

def create_info_mne(reject_ch=0,sfreq=100):
    '''
    Creates an MNE info data structure.
    
    # Input:
        reject_ch: bool. Whether to reject predefined channels.
        sfreq: int. Sampling frequency.
        
    # Output:
        info: MNE info data structure.
    '''
    
    if reject_ch == True:
        channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','C4','F4','C3','F3','Oz','PO3','FC5','FC1','CP5','CP1','CP2','CP6','FC2','FC6','PO4']
        channel_types = ['eeg']*23
    else:
        channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']
        channel_types = ['eeg']*32
        
    montage = 'standard_1020' 
    info = mne.create_info(channel_names, sfreq, channel_types, montage)
    
    return info
    
def preproc1epoch(eeg,info,projs=[],SSP=True,reject=None,mne_reject=1,reject_ch=None,flat=None,bad_channels=[],opt_detrend=1):

    '''
    Preprocesses epoched EEG data.
    
    # Input
    - eeg: numPy array. EEG epoch in the following format: (time samples, channels).
    - info: MNE info data structure. Predefined info containing channels etc. Can be generated using create_info_mne function.
    - projs: MNE SSP projector objects. Used if SSP = True. 
    - reject: bool. Whether to reject channels, either manually defined or based on MNE analysis.
    - reject_ch: bool. Whether to reject nine predefined channels.
    - mne_reject: bool. Whether to use MNE rejection based on epochs._is_good. 
    - flat: bool??. Input for MNE rejection
    - bad_channels: list. Manual rejection of channels.
    - opt_detrend: bool. Whether to apply temporal EEG detrending (linear).
    
    # Preprocessing steps - based on inputs 
    - Linear temporal detrending
    - Rejection of initial, predefined channels 
    - Bandpass filter (0-40Hz)
    - Resample to 100Hz
    - SSP correction 
    - Rejection of bad channels
        - Interpolation of bad channels
    - Rereference to average
    - Baseline correction
    
    # Output
    - Epoched preprocessed EEG data in numPy array.
    
    '''
    
    n_samples = eeg.shape[0]
    n_channels = eeg.shape[1]
    eeg = np.reshape(eeg.T,(1,n_channels,n_samples))
    tmin = -0.1 # Baseline start 
    
    # Temporal detrending:
    if opt_detrend == 1:
        eeg = detrend(eeg, axis=2, type='linear')
        
    epoch = mne.EpochsArray(eeg, info, tmin=tmin, baseline=None, verbose=False)
    
    # Drop list of channels known to be problematic:
    if reject_ch == True: 
        bads =  ['Fp1','Fp2','Fz','AF3','AF4','T7','T8','F7','F8']
        epoch.drop_channels(bads)
    
    # Lowpass
    epoch.filter(HP, LP, fir_design='firwin', phase=phase, verbose=False)
    
    # Downsample
    epoch.resample(100, npad='auto',verbose=False)
    
    # Apply baseline correction
    epoch.apply_baseline(baseline=(None,0),verbose=False)
    
    # Apply SSP projectors
    if SSP == True:
        # Apply projection to the epochs already defined
        epoch.add_proj(projs)
        epoch.apply_proj()
        
    if reject is not None: # Rejection of channels, either manually defined or based on MNE analysis. Currently not used.
        if mne_reject == 1: # Use MNE method to reject+interpolate bad channels
            from mne.epochs import _is_good
            from mne.io.pick import channel_indices_by_type    
            #reject=dict(eeg=100)
            idx_by_type = channel_indices_by_type(epoch.info)
            A,bad_channels = _is_good(epoch.get_data()[0], epoch.ch_names, channel_type_idx=idx_by_type,reject=reject, flat=flat, full_report=True)
            print(A)
            if A == False:
                epoch.info['bads']=bad_channels    
                epoch.interpolate_bads(reset_bads=True, verbose=False)
        else: # Predfined bad_channels 
            epoch.drop_channels(bad_channels)
    
    # Rereferencing
    epoch.set_eeg_reference(verbose=False)
    
    # Apply baseline after rereference
    epoch.apply_baseline(baseline=(None,0),verbose=False)
        
    epoch = epoch.get_data()[0]
    
    return epoch


