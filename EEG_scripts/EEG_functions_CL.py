# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:43:12 2019

@author: sofha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:26:22 2019

@author: sofha
"""

# -*- coding: utf-8 -*-
"""
Functions for working with EEG in Python: Extract raw EEG data, align EEG to time-locked stimuli, divide EEG into epochs
based on marker/experimental trial timing, extract experimental categories, computing and analyzing variance of SSP projection
vectors, preprocessing of epoched EEG data (filtering, baseline correction, rereferencing, resampling)

@author: Greta
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

def extractRaw(EEGfile):
    '''
    Extracts raw EEG based on a .csv file of EEG recordings.
    
    # Input
    - csv file with channels as columns and trials as rows.
    
    # Output
    - EEGdata as a numpy array.
    
    '''
    
    le = np.zeros(4500000)
    EEGdata = []
    c = 0
    
    with open(EEGfile,'r') as csvfile:

        csvReader = csv.reader(csvfile)
        for row in csvfile:
            rownum = np.fromstring(row,dtype=float,sep=',')
            le[c] = (len(rownum))
            EEGdata.append(rownum)
            c += 1
    
    EEGdata = EEGdata[2:]
    eegdata = np.array(EEGdata)
    
    # Check whether the EEG arrays have same size, or whether the EEG acquisition was turned off too soon
    if len(eegdata.shape) == 1:
        print('Arrays of size 1 in the end of EEG file. Removed 300 samples.')
        # EEGdata2 = np.copy(EEGdata)
        EEGdata2 = EEGdata[:-300]
        eegdata = np.array(EEGdata2)
            
    return eegdata

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

def computeSSP(EEG,info,threshold,reject_ch=None,SSP_start_end=0):
    '''
    Computes SSP projections of non-epoched EEG data (bandpass filtered and resampled to 100Hz). 
    
    # Input
    - EEG csv file (calls extractRaw function) or epoched and processed EEG
    - info must exist if EEG is not csv file
    - Threshold: Value between 0 and 1, only uses the SSP projection vector if it explains more than the pre-defined threshold.
    - SSP_start_end: Pre-defined "start" and "end" time points (array consisting of two floats) for the length of raw EEG to use for SSP projection vector analysis. Acquired from extractEpochs function.
    - reject_ch: Whether to reject pre-defined channels.
    
    # Output
    - Array of SSP projection vectors above a pre-defined threshold (variance explained).
    
    '''
    
    # INFO
    #channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1',
    #                  'T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']
    #channel_types = ['eeg']*32
    #sfreq = 100  # in Hertz
    #montage = 'standard_1020' # Or 1010
    #info = mne.create_info(channel_names, sfreq, channel_types, montage)
    
    # Load raw data csv file
    if type(EEG)==str:
        print('Computing SSP based on raw file')
        raw = extractRaw(EEGfile)
        
        if reject_ch == True:
            info['bads'] = ['Fp1','Fp2','Fz','AF3','AF4','T7','T8','F7','F8']
    
            # Remove bad channels based on picks idx
            good_indices = mne.pick_types(info,meg=False,eeg=True,stim=False,eog=False,exclude='bads')
        
            # Find indices for reduced info channels
            channel_names = np.asarray(channel_names)
            channel_names_red = channel_names[good_indices]
            channel_types_red = ['eeg']*(len(channel_names_red))
            channel_names_red = channel_names_red.tolist()
            
            info_red = mne.create_info(channel_names_red, sfreq, channel_types_red, montage)
    
            raw_red = raw[:,good_indices]
            raw_red_d = detrend(raw_red, axis=0, type='linear')
        
            custom_raw = mne.io.RawArray(raw_red_d.T, info_red)
        
        if reject_ch == None:
            raw = raw[:,0:32] # Remove marker channel
            raw_d = detrend(raw, axis=0, type='linear')
            custom_raw = mne.io.RawArray(raw_d.T, info)
            
        # Bandpass filtering
        custom_raw.filter(HP, LP, fir_design='firwin',phase=phase)
    
        # Resampling
        custom_raw.resample(100, npad='auto')
        
        projs = mne.compute_proj_raw(custom_raw, start=SSP_start_end[0], stop=SSP_start_end[1], duration=1, n_grad=0, n_mag=0,
                             n_eeg=10, reject=None, flat=None, n_jobs=1, verbose=True)
    else:
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
    EEG = mne.EpochsArray(EEG, info,baseline=None)
    projs=computeSSP(EEG,info,threshold)
    EEG.add_proj(projs)
    EEG.apply_proj()
    EEG=EEG.get_data()
    return projs,EEG

def average_stable(stable):
    stable_mid=(stable[:-1]+stable[1:])/2
    stable_2mean=np.zeros(stable.shape)
    stable_2mean[0]=stable[0]
    stable_2mean[1:]=stable_mid
    return stable_2mean

def removeEpochs(EEG,info,interpolate=0):

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
    if reject_ch == True:
        channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','C4','F4','C3','F3','Oz','PO3','FC5','FC1','CP5','CP1','CP2','CP6','FC2','FC6','PO4']
        channel_types = ['eeg']*23
    else:
        channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']
        channel_types = ['eeg']*32
    montage = 'standard_1020' # Or 1010
    info = mne.create_info(channel_names, sfreq, channel_types, montage)
    #info['description'] = 'Enobio'
    return info
    
def preproc1epoch(eeg,info,projs=[],SSP=True,reject=None,mne_reject=1,reject_ch=None,flat=None,bad_channels=[],opt_detrend=1):

    from mne.epochs import _is_good
    from mne.io.pick import channel_indices_by_type
    
    '''
    Preprocesses epoched EEG data.
    
    # Input
    - eeg: Epoched EEG data in the following format: (trials, time samples, channels).
    - info: predefined info containing channels etc.
    - projs: used if SSP=True. SSP projectors
    
    # Preprocessing
    - EpochsArray format in MNE (with initial baseline correction)
    - Bandpass filter (0-40Hz)
    - Resample to 100Hz
    - SSP (if True)
    - Reject bad channels
        - interpolate bad channels
    - Rereference to average
    - Baseline correction
    
    # Output
    - Epoched preprocessed EEG data in np array.
    
    '''
    
    # INFO

    #info['bads'] = ['P7',]
    #b2=(np.mean(eeg[:50,:],axis=0)+np.mean(eeg[500:,:],axis=0))/2
    #b=np.reshape(np.tile(b2,[1,550]),(550,32))
    #eeg=eeg-b
    n_samples=eeg.shape[0]
    eeg=np.reshape(eeg.T,(1,32,n_samples))
    tmin = -0.1 
    if opt_detrend==1:
        eeg = detrend(eeg, axis=2, type='linear')
        
    epoch = mne.EpochsArray(eeg, info, tmin=tmin,baseline=None,verbose=False)

    if reject_ch == True:
        bads =  ['Fp1','Fp2','Fz','AF3','AF4','T7','T8','F7','F8']
        epoch.drop_channels(bads)
    # Bandpass filtering
    
    # Lowpass
    epoch.filter(HP, LP, fir_design='firwin',phase=phase,verbose=False)#)
    # downsample
    epoch.resample(100, npad='auto',verbose=False)
    
    # Apply SSP (computed based on the raw/not epoched EEG)
    epoch.apply_baseline(baseline=(None,0),verbose=False)

    if SSP == True:
        # Apply projection to the epochs already defined
        epoch.add_proj(projs)
        epoch.apply_proj()
        
    if reject is not None:
        #epochs.drop_bad(reject=reject, flat='existing', verbose=None)
        if mne_reject==1:
            #reject=dict(eeg=100)
            idx_by_type = channel_indices_by_type(epoch.info)
            A,bad_channels=_is_good(epoch.get_data()[0], epoch.ch_names, channel_type_idx=idx_by_type,reject=reject, flat=flat, full_report=True)
            print(A)
            if A==False:
                epoch.info['bads']=bad_channels    
                epoch.interpolate_bads(reset_bads=True, verbose=False)
        else:
            epoch.drop_channels(bad_channels)
            
#        else:
#            data=epoch.get_data()
#            m=np.max(np.abs(data),axis=2).ravel()
#            bad_channels=m>reject
#                
#            if len(bad_channels):
#                epoch.info['bads'] = [epoch.ch_names[c] for c in bad_channels]
#                epoch.interpolate_bads(reset_bads=False, verbose=False)
            
        #if len(epoch.events)<1:
            
    
    # Rereferencing
    if SSP==True or SSP==False:
        epoch.set_eeg_reference(verbose=False)
        # Apply baseline after rereference
        epoch.apply_baseline(baseline=(None,0),verbose=False)
        
    epoch=epoch.get_data()[0]
    return epoch


