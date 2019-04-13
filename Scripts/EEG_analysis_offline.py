# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:43:08 2019
Functions for analysing EEG offline partly using MNE: Extract raw EEG data, align EEG to time-locked stimuli, divide EEG into epochs
based on marker/experimental trial timing, extract experimental categories, computing and analyzing variance of SSP projection
vectors, preprocessing of epoched EEG data (filtering, baseline correction, rereferencing, resampling)

@author: Greta & Sofie
"""

#### Imports ####

import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
from scipy.signal import detrend
import pandas as pd
import os 
import mne
from scipy import stats

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



def extractEpochs_tmin(EEGfile,markerfile,prefilter=0,marker1=1,n_samples=550):
    ''' 
    Extracts time-locked EEG epochs based on stimuli/marker timing in stream 33.
    
    # Input
    - csv file with channels as columns and trials as rows.
    - csv file with marker time points for each experimental event.
    - prefilter: whether to filter data before extracting epochs
    - marker1: whether all markers of interest have value '1'
    - n_samples: length of epoch to extract
    # Output
    - Epoched EEG data (1100 ms. 100 ms before stimulus onset, and 1000 ms after stimulus onset).
    
    GT edit, 04april: added marker_start_end and epochidx0 info.
    
    '''
    n_channels = 32
    le = np.zeros(4500000)
    EEGdata = []
    c = 0
    tmin = -0.1
    
    # Read raw EEG data csv file 
    with open(EEGfile,'r') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvfile:
            rownum = np.fromstring(row,dtype=float,sep=',')
            le[c] = (len(rownum))
            if le[c] == (n_channels+1):
                EEGdata.append(rownum)
            c += 1
    
    EEGdata = EEGdata[2:]
    eegdata = np.array(EEGdata)
    eegdata_time = eegdata[:,n_channels]
    eegdata = eegdata[:,0:n_channels] # Remove marker as the last channel
    
    # Filtering of EEG data before epoch extraction 
    if prefilter == 1:
        channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']
        channel_types = ['eeg']*n_channels
        sfreq = 500  # in Hertz
        montage = 'standard_1020' 
        info = mne.create_info(channel_names, sfreq, channel_types, montage)
        raw = detrend(eegdata, axis=0, type='linear')
        custom_raw = mne.io.RawArray(raw.T, info)
        custom_raw.filter(HP, LP, fir_design='firwin',phase=phase)
        eegdata = custom_raw.get_data().T
    
    marker = []
    marker_time = []
    c = 0
    
    # Read time marker data csv file
    with open(markerfile,'r') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvfile:
            rownum=np.fromstring(row,dtype=float,sep=',')
            le[c]=(len(rownum))
            if len(rownum)>1:
                marker.append(rownum[0])
                marker_time.append(rownum[1])
            c+=1
        
    # Check whether the EEG arrays in the raw EEG data have same size
    # Mismatch can occur if EEG acquisition equipment was turned off unexpectedly
    if len(eegdata.shape) == 1:
        print('Arrays of size 1 in the end of EEG file. Removed 300 samples.')
        # EEGdata2 = np.copy(EEGdata)
        EEGdata2 = EEGdata[:-300]
        eegdata = np.array(EEGdata2)
        
    marker_time2 = np.asarray(marker_time)

    # Find marker events 1 (start of experimental trial)
    if marker1 == 1:
        eventsM = indices(marker, lambda x: x == 1.0)
    else:
        eventsM = [int(m) for m in marker]
    eventsM2 = np.asarray(eventsM)

    # Match EEG timestamp with marker timestamp (stimuli onset) + tmin 
    nEv = len(eventsM)
    eventsEEG0 = np.zeros(nEv)
    for trial in range(nEv):
        eventsEEG0[trial] = np.argmin(abs(marker_time2[eventsM2[trial]]+tmin-eegdata_time)) #gives EEG 0 timestamp

  
    # Take the eegdata variable and extract epochs
    epochs = np.zeros([nEv,n_samples,n_channels])
    
    epoch0_idx = eventsEEG0.tolist()
    epoch0_idx = [int(i) for i in epoch0_idx]

    for count,number in enumerate(epoch0_idx):
        if number+n_samples<len(eegdata[:,1]):
            epochs[count,:,:] = eegdata[number:number+n_samples,0:n_channels]
            # Should it be eegdata[number+tmin:number+n_samples+tmin] ?
    
    marker_start_end = [marker_time2[eventsM2[0]],marker_time2[eventsM2[-1]]]-eegdata_time[0]

    print('Number of trials extracted: ' + str(nEv))
    
    return epochs

def extractEpochs(EEGfile,markerfile,prefilter=0,marker1=1):
    ''' 
    Extracts time-locked EEG epochs based on stimuli/marker timing in channel 33.
    (less flexible than extractEpochs_tmin)
    # Input
    - EEGfile: csv file with channels as columns and trials as rows.
    - markerfile: csv file with marker time points for each experimental event.
    - prefilter: whether to filter data before extracting epochs
    - marker1: whether all markers of interest have value '1'
    
    # Output
    - Epoched EEG data (1100 ms. 100 ms before stimulus onset, and 1000 ms after stimulus onset).
    - marker_start_end: timestamp of the first and last marker
    '''
    # Initialize
    le = np.zeros(4500000)
    EEGdata = []
    c = 0
    
    with open(EEGfile,'r') as csvfile:

        csvReader = csv.reader(csvfile)
        for row in csvfile:
            rownum = np.fromstring(row,dtype=float,sep=',')
            le[c] = (len(rownum))
            if le[c]==33:
                EEGdata.append(rownum)
            c += 1
    
    EEGdata = EEGdata[2:]
    eegdata = np.array(EEGdata)
    eegdata_time=eegdata[:,32] # timestamps of EEG
    eegdata = eegdata[:,0:32] # Remove marker as the last channel
    if prefilter==1:
        channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']
        channel_types = ['eeg']*32
        sfreq = 500  # in Hertz
        montage = 'standard_1020' # Or 1010
        info = mne.create_info(channel_names, sfreq, channel_types, montage)
        raw=detrend(eegdata, axis=0, type='linear')
        custom_raw = mne.io.RawArray(raw.T, info)
        custom_raw.filter(HP, LP, fir_design='firwin',phase=phase)
        eegdata=custom_raw.get_data().T
    
    marker=[]
    marker_time=[]
    c=0
    with open(markerfile,'r') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvfile:
            rownum=np.fromstring(row,dtype=float,sep=',')
            le[c]=(len(rownum))
            if len(rownum)>1:
                marker.append(rownum[0])
                marker_time.append(rownum[1])
            c+=1
        

    # Check whether the EEG arrays have same size, or whether the EEG acquisition was turned off too soon
    if len(eegdata.shape) == 1:
        print('Arrays of size 1 in the end of EEG file. Removed 300 samples.')
        # EEGdata2 = np.copy(EEGdata)
        EEGdata2 = EEGdata[:-300]
        eegdata = np.array(EEGdata2)
        
    marker_time2 = np.asarray(marker_time)

    # Find marker events 1 (start of experimental trial)
    if marker1==1:
        eventsM = indices(marker, lambda x: x == 1.0)
    else:
        eventsM=[int(m) for m in marker]
    eventsM2 = np.asarray(eventsM)

    # match EEG timestamp with marker timestamp (stimuli onset)    
    nEv = len(eventsM)
    eventsEEG0 = np.zeros(nEv)
    for trial in range(nEv):
        eventsEEG0[trial] = np.argmin(abs(eegdata_time - marker_time2[eventsM2[trial]])) #gives EEG 0 timestamp

  
    # Take the eegdata variable and extract epochs
    epochs = np.zeros([nEv,550,32])

    epoch0_idx = eventsEEG0.tolist()
    epoch0_idx = [int(i) for i in epoch0_idx]

    for count,number in enumerate(epoch0_idx):
        
        epochs[count,:,:] = eegdata[number-50:number+500,0:32]
    
    print('Number of trials extracted: ' + str(nEv))
    marker_start_end=[marker_time2[eventsM2[0]],marker_time2[eventsM2[-1]]]-eegdata_time[0]
    
    return epochs,marker_start_end

def extractCat(indicesFile,exp_type='fused'):
    ''' 
    Extracts experimental categories from .csv file.
    
    # Input
    - csv file with file directories of shown experimental trials (two images for each trial).
    
    # Output
    - Binary list with 0 denoting scenes, and 1 denoting faces.
    
    '''
    
    if exp_type == 'fused':
        colnames = ['1', 'att_cat', 'binary_cat', '3', '4']
        data = pd.read_csv(indicesFile, names=colnames)
        categories = data.att_cat.tolist()
        binary_categories = data.binary_cat.tolist()
        del categories[0:1]
        del binary_categories[0:1]
    
            
    if exp_type == 'nonfused':
        colnames = ['cat', 'path']
        data = pd.read_csv(indicesFile,names=colnames)
        categories = data.cat.tolist()
        del categories[0:1]
        
        binary_categories = []
        for c in categories:
            if c == 'scene':
                binary_categories.append(0)
            else:
                binary_categories.append(1)
                
    return binary_categories

def convertBinary(textCategories):
    '''
    Converts string category names to binary list.
    
    # Input
    - NumPy array of categories.
    
    # Output
    - Binary list
    
    '''
    
    catLst = textCategories.tolist()
    
    binary_cat = []
    for c in catLst:
        c = c.strip()
        if c == 'indoor' or c == 'outdoor':
            binary_cat.append(0)
        else:
            binary_cat.append(1)
    
    return binary_cat

def extractAlpha(alphaFile):
    ''' 
    Extracts experimental alpha values from .csv file.
    
    # Input
    - csv file with alpha and marker
    
    # Output
    - alpha and marker.
    
    '''
    
    colnames = ['marker', 'alpha']
    data = pd.read_csv(alphaFile,names=colnames)
    marker = data.marker.tolist()[1:]
    alpha = data.alpha.tolist()[1:]
   
                
    return alpha,marker

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

def computeSSP_offline(EEGfile,threshold,SSP_start_end,reject_ch=None):
    '''
    Computes SSP projections of non-epoched EEG data (bandpass filtered and resampled to 100Hz). 
    (if EEG has been epoched then use computeSSP from EEG_analysis_RT)
    # Input
    - EEG csv file (calls extractRaw function)
    - Threshold: Value between 0 and 1, only uses the SSP projection vector if it explains more than the pre-defined threshold.
    - SSP_start_end: Pre-defined "start" and "end" time points (array consisting of two floats) for the length of raw EEG to use for SSP projection vector analysis. Acquired from extractEpochs function.
    - reject_ch: Whether to reject pre-defined channels.
    
    # Output
    - Array of SSP projection vectors above a pre-defined threshold (variance explained).
    
    '''
    
    # INFO
    channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1',
                     'T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']
    channel_types = ['eeg']*32
    sfreq = 500  # in Hertz
    montage = 'standard_1020' # Or 1010
    info = mne.create_info(channel_names, sfreq, channel_types, montage)
    
    # Load raw data csv file
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
    
    p = [projs[i]['explained_var'] for i in range(10)]

    # If variance explained is above a certain threshold, use the SSP vector for projection
    threshold_idx = analyzeVar(p, threshold)

    threshold_projs = [] # List with projections above a chosen threshold
    for idx in threshold_idx:
        threshold_projs.append(projs[idx])
        
    return threshold_projs

def preprocEEG(eeg,EEGfile,SSP=True, threshold=0.1,SSP_start_end=None,reject=None,events_list=None,reject_ch=None,set_ref=1,prefilter=0,flat=None):
    '''
    Preprocesses epoched EEG data.
    
    # Input
    - eeg: Epoched EEG data in the following format: (trials, time samples, channels).
    - EEGfile: .csv file with raw EEG recordings (for computing SSP projection vectors). If EEGfile == [], the SSP projection vectors are computed based on epochs.
    - SSP: Whether to apply SSP projections.
    - Threshold: Value between 0 and 1, only uses the SSP projection vector if it explains more than the pre-defined threshold.
    - SSP_start_end: Pre-defined "start" and "end" time points (array consisting of two floats) for the length of raw EEG to use for SSP projection vector analysis. Acquired from extractEpochs function.
    - reject_ch: Whether to reject pre-defined channels.
    - set_ref: Use average reference
    - prefilter: Filter before epoch extraction
    - flat: reject channels having close to no variation
    
    # Preprocessing
    - Linear detrending (channel-wise)
    - EpochsArray format in MNE 
    - Reject channels (if reject_ch == True)
    - Bandpass filter 
    - Resample to 100Hz
    - SSP (if ssp == True)
    - Reject bad channels/epochs (if ??)
    - Baseline correction
    - Rereference to average
    - Baseline correction
    
    # Output
    - epochs: Epoched EEG data in MNE EpochsArray
    - projs: SSP projection vectors
    
    '''
    
    # INFO
    channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']
    channel_types = ['eeg']*32
    sfreq = 500  # in Hertz
    montage = 'standard_1020' # Or 1010
    info = mne.create_info(channel_names, sfreq, channel_types, montage)
    info['description'] = 'Enobio'
    
    #if reject_ch == 'true':
    #    info['bads'] = ['Fp1','Fp2','Fz','AF3','AF4','T7','T8','F7','F8']
    
    eeg = detrend(eeg, axis=1, type='linear')
    no_trials = eeg.shape[0]

    axis_move = np.moveaxis(eeg, [1,2], [-1,-2]) # Rearrange into MNE format
    
    tmin = -0.1 
    
    if events_list is not None: # if categories are included
        event_id=dict(scene=0, face=1)
        n_epochs=len(events_list)
        events_list=[int(i) for i in events_list]
        events = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int),
                           events_list]
    else:
        event_id = None
        events=None

    epochs = mne.EpochsArray(axis_move, info, events=events, tmin=tmin, event_id=event_id,baseline=None)
    
    if reject_ch == True:
        bads =  ['Fp1','Fp2','Fz','AF3','AF4','T7','T8','F7','F8']
        epochs.drop_channels(bads)

    # Bandpass filtering
    epochs.filter(HP, LP, fir_design='firwin',phase=phase)

    # Resampling to 100 Hz
    epochs.resample(100, npad='auto')
    
    # Apply SSP (computed based on the raw/not epoched EEG)
    if SSP == True:
        
        if not EEGfile:
            print('Computing SSP based on epochs')
            all_projs = mne.compute_proj_epochs(epochs,n_eeg=10,n_jobs=1, verbose=True)
            p = [all_projs[i]['explained_var'] for i in range(10)]
            # If variance explained is above a certain threshold, use the SSP vector for projection
            threshold_idx = analyzeVar(p, threshold)
            projs = [] # List with projections above a chosen threshold
            for idx in threshold_idx:
                projs.append(all_projs[idx])
            
            
        else:
        
            if reject_ch == None:
                projs = computeSSP(EEGfile,threshold,SSP_start_end,reject_ch=None)
                
            if reject_ch == True:
                projs = computeSSP(EEGfile,threshold,SSP_start_end,reject_ch=True)
            
        # Apply projection to the epochs already defined
        epochs.add_proj(projs)
        epochs.apply_proj()
    else:
        projs = []
        
    if reject is not None:
        epochs_copy=epochs.copy()
        epochs_copy.drop_bad(reject=reject, flat=flat, verbose=None)
        log=epochs_copy.drop_log
        not_blank=np.concatenate([x for x in (log) if x])
        bad_channels, bad_counts = np.unique(not_blank, return_counts=True)
        #bad_channels=[bad_channels[a] for a in range(len(bad_channels))]
        thres=0.05*no_trials
        bad_above_thres=bad_channels[bad_counts>thres]
        if len(bad_above_thres):
            print(1)
            epochs.info['bads']=bad_above_thres
            print('Dropping channels:',bad_above_thres)
            epochs.interpolate_bads(reset_bads=True)
            # repeat looking for bad epochs
            epochs_copy=epochs.copy()
            epochs_copy.drop_bad(reject=reject, flat=flat, verbose=None)
            log=epochs_copy.drop_log
        
        bad_epochs=[i for i,x in enumerate(log) if x]
        epochs.drop(bad_epochs,reason='Reject')
    
    epochs.apply_baseline(baseline=(None,0))
    
    # Rereferencing
    epochs.set_eeg_reference()

    # Apply baseline after filtering
    epochs.apply_baseline(baseline=(None,0))
    
    return epochs,projs

def scaleEpochs(epochs):
    '''
    Scale each epoch/trial in epochs and extract events
    
    Only outputs correct y if individual epochs have been rejected.
    
    '''
    
    epochs_data = epochs.get_data()
    
    no_trials = epochs_data.shape[0]
    no_chs = epochs_data.shape[1]
    no_samples = epochs_data.shape[2]
    
    X = np.reshape(epochs_data,(no_trials,no_chs*no_samples))
    X = stats.zscore(X, axis=1)
    y = epochs.events[:,2]
    
    return X,y

def scaleArray(eeg_array):
    '''
    Input:
    - eeg_array in the following format: (trials, time samples, channels).
    
    '''
    
    no_trials = eeg_array.shape[0]
    no_chs = eeg_array.shape[1]
    no_samples = eeg_array.shape[2]

    X_res = np.reshape(eeg_array,(no_trials,no_chs*no_samples))

    X_z = stats.zscore(X_res, axis=1)

    return X_z


def ERPcomp(eeg_data,sfreq,times,comp='all_ERPs',avg=1): # Not currently used
    
    ERP=['P1','N1','vN1','P2','vN2','P3','N3']
    P1=[[0.115], [0.135]]
    N1=[[0.135],[0.155]]
    vN1=[[0.155],[0.195]]
    P2=[[0.205],[0.235]]
    vN2=[[0.285],[0.325]]
    P3=[[0.335],[0.395]]
    N3=[[0.495],[0.535]]
    no_trials=eeg_data.shape[0]
    if comp=='all_ERPs':
        comps=np.concatenate((P1,N1,vN1,P2,vN2,P3,N3),axis=1)
        comp_comb=np.zeros((no_trials,32,1))
        for c in range(comps.shape[1]):
            tmin = int(round(comps[0,c] * sfreq)) / sfreq - 0.5 / sfreq # from MNE
            tmax = int(round(comps[1,c] * sfreq)) / sfreq + 0.5 / sfreq # from MNE
            mask=(times >= tmin)
            mask &= (times <= tmax)
            ERP=eeg_data[:,:, mask]
            if avg==1:
                ERP=np.reshape(np.mean(ERP,axis=2),(no_trials,32,1))
            comp_comb=np.append(comp_comb,ERP,axis=2)
    elif type(comp)==list:
        tmin = int(round(comp[0] * sfreq)) / sfreq - 0.5 / sfreq # from MNE
        tmax = int(round(comp[1] * sfreq)) / sfreq + 0.5 / sfreq # from MNE
        mask=(times >= tmin)
        mask &= (times <= tmax)
        ERP=eeg_data[:,:, mask]
        if avg==1:
            ERP=np.reshape(np.mean(ERP,axis=2),(no_trials,32,1))
            
            
    
def preproc1epoch_forplots(eeg,info,projs=[],SSP=True,reject=None,mne_reject=1,reject_ch=None,flat=None,bad_channels=[],opt_detrend=1):
    
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
    
    n_samples=eeg.shape[0]
    n_channels=eeg.shape[1]
    eeg=np.reshape(eeg.T,(1,n_channels,n_samples))
    tmin = -0.1 # start baseline at
    
    # Temporal detrending:
    if opt_detrend==1:
        eeg = detrend(eeg, axis=2, type='linear')
        
    epoch = mne.EpochsArray(eeg, info, tmin=tmin,baseline=None,verbose=False)
    
    # Drop list of channels known to be problematic:
    if reject_ch == True: 
        bads =  ['Fp1','Fp2','Fz','AF3','AF4','T7','T8','F7','F8']
        epoch.drop_channels(bads)
    
    # Lowpass
    epoch.filter(HP, LP, fir_design='firwin',phase=phase,verbose=False)
    
    # Downsample
    epoch.resample(100, npad='auto',verbose=False)
    
    # Apply baseline correction
    epoch.apply_baseline(baseline=(None,0),verbose=False)
    
    # Apply SSP prejectors
    if SSP == True:
        # Apply projection to the epochs already defined
        epoch.add_proj(projs)
        epoch.apply_proj()
        
    if reject is not None: # currently not used
        if mne_reject==1: # use mne method to reject+interpolate bad channels
            from mne.epochs import _is_good
            from mne.io.pick import channel_indices_by_type    
            #reject=dict(eeg=100)
            idx_by_type = channel_indices_by_type(epoch.info)
            A,bad_channels=_is_good(epoch.get_data()[0], epoch.ch_names, channel_type_idx=idx_by_type,reject=reject, flat=flat, full_report=True)
            print(A)
            if A==False:
                epoch.info['bads']=bad_channels    
                epoch.interpolate_bads(reset_bads=True, verbose=False)
        else: # bad_channels is predefined
            epoch.drop_channels(bad_channels)
            
            
    
    # Rereferencing
    epoch.set_eeg_reference(verbose=False)
    # Apply baseline after rereference
    epoch.apply_baseline(baseline=(None,0),verbose=False)
        
    # epoch=epoch.get_data()[0]
    return epoch

def applySSP_forplot(EEG,info,threshold=0.1,add_events=None):
    '''
    Computes and applies SSP projections of epoched EEG data (bandpass filtered and resampled to 100Hz). 
    
    # Input
    - EEG: Epoched and processed EEG
    - info: mne info struct
    - threshold: Value between 0 and 1, only uses the SSP projection vector if it explains more than the pre-defined threshold.
    - add_events: numPy array with binary categories.
    
    # Output
    - EEG in MNE EpochsArray after SSP corection. If add_events is not none, categories are added to the MNE EpochsArray.
    
    '''
    
    if add_events is not None:
        events_list = add_events
        event_id_add = dict(scene=0, face=1)
        n_epochs = len(events_list)
        events_list = [int(i) for i in events_list]
        events_add = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int),events_list]
    
    EEG = mne.EpochsArray(EEG, info, baseline=None, tmin=-0.1,events=events_add, event_id=event_id_add)
    projs,p = computeSSP_forplot(EEG,info,threshold)
    EEG.add_proj(projs)
    EEG.apply_proj()
    
    return projs,EEG,p


def computeSSP_forplot(EEG,info,threshold):
    '''
    Computes SSP projections of epoched EEG data (bandpass filtered and resampled to 100Hz). 
    Returns explained variance (p)
    
    # Input
    - EEG: Epoched and processed EEG
    - info: mne info struct
    - threshold: Value between 0 and 1, only uses the SSP projection vector if it explains more than the pre-defined threshold.
    
    # Output
    - Array of SSP projection vectors above a pre-defined threshold (variance explained).
    - p: Array containing explained variance
    
    '''

    print('Computing SSP based on epochs')

    projs = mne.compute_proj_epochs(EEG,n_eeg=10,n_jobs=1, verbose=True)

    p = [projs[i]['explained_var'] for i in range(10)]

    # If variance explained is above a certain threshold, use the SSP vector for projection
    threshold_idx = analyzeVar(p, threshold)

    threshold_projs = [] # List with projections above a chosen threshold
    for idx in threshold_idx:
        threshold_projs.append(projs[idx])
        
    return threshold_projs,p

            
