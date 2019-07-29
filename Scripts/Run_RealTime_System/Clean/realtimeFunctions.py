# -*- coding: utf-8 -*-

"""
Created on Tue Jan 22 13:26:22 2019

Functions for working with EEG in Python: computing and analyzing variance of SSP projection
vectors, preprocessing of epoched EEG data (filtering, baseline correction, rereferencing, resampling)

Contains functions for preprocessing EEG in real-time (first part) and decoding EEG in real-time (second part).

@author: Greta & Sofie


EEG_classification.py and EEG_analysis_RT and offline fused..


"""

# Imports 
import numpy as np
from scipy.stats import zscore
import os 
import mne
from scipy.signal import detrend
from sklearn.linear_model import LogisticRegression

#### 1) EEG PREPROCESSING FUNCTIONS ####

# EEG preprocessing variables 
LP = 40 # Low-pass filter cut-off
HP = 0 # High-pass filter cut-off
phase = 'zero-double' # FIR filter phase (refer to MNE filtering for options)

def scale1DArray(eeg_array, axis=1):
    '''
    Scales a 2D array, with scaling on the specified axis.
    
    # Arguments
        eeg_array: NumPy array
            Array of EEG data in the following format: [channels, time samples].
            
        axis: int
            Normalization axis.
    
    # Returns
        X_z: NumPy array
            Scaled data array (mean = 0 and std = 1 along specified axis)
    '''
    no_chs = eeg_array.shape[0]
    no_samples = eeg_array.shape[1]

    X_res = np.reshape(eeg_array, (1, no_chs*no_samples))
    X_z = zscore(X_res, axis=axis)
    
    return X_z


def scale2DArray(eeg_array, axis=1):
    '''
    Scales a 3D array to a 2D array, with scaling on the specified axis.
    
    # Arguments
        eeg_array: NumPy array
            Array of EEG data in the following format: [trials, channels, time samples].
            
        axis: int
            Normalization axis.
            
    # Returns
        X_z: NumPy array
            Scaled data array (mean = 0 and std = 1 along specified axis)
    '''    
    no_trials = eeg_array.shape[0]
    no_chs = eeg_array.shape[1]
    no_samples = eeg_array.shape[2]

    X_res = np.reshape(eeg_array, (no_trials, no_chs*no_samples))
    X_z = zscore(X_res, axis=axis)

    return X_z


def analyzeVar(p, threshold):
    '''
    Analyzes computed SSP projection vectors and only uses a projection vector if the vector independently explains more variance than the specified threshold. 
    
    # Arguments
        p: list
            List containing variance explained values by each computed SSP projection vector.
            
        Threshold: float
            Value between 0 and 1.
    
    # Returns
        threshold_idx: list
            Indices of SSP projection vectors above the defined threshold.
    '''
    
    threshold_idx = []
    for count, val in enumerate(p):
        if val > threshold:
            threshold_idx.append(count)
            
    return threshold_idx

def computeSSP(EEG, info, threshold):
    '''
    Computes SSP projections of epoched EEG data.
    
    # Arguments
        EEG: MNE EpochsArray data structure
            Epoched EEG data in MNE format.
        
        info: MNE info structure
        
        Threshold: float
            Value between 0 and 1.
    
    # Returns
        threshold_projs: list    
            List of SSP projection vectors above the pre-defined threshold (variance explained).
    '''
    projs = mne.compute_proj_epochs(EEG, n_eeg=10, n_jobs=1, verbose=True)

    p = [projs[i]['explained_var'] for i in range(10)]

    # If variance explained is above the pre-defined threshold, use the SSP projection vector
    threshold_idx = analyzeVar(p, threshold)

    threshold_projs = [] # List with projections above the threshold
    for idx in threshold_idx:
        threshold_projs.append(projs[idx])
        
    return threshold_projs

def applySSP(EEG, info, threshold=0.1,):
    '''
    Applies SSP projections to epoched EEG data. 
    
    # Arguments
        EEG: NumPy array
            Array of EEG data in the following format: [trials, channels, time samples].
        
        info: MNE info structure
        
        Threshold: float
            Value between 0 and 1.
    
    # Returns
        projs: list    
            List of SSP projection vectors above the pre-defined threshold (variance explained).
            
        EEG: NumPy array
            Array of EEG data in the following format: [trials, channels, time samples].
    '''
    EEG = mne.EpochsArray(EEG, info, baseline=None)
    projs = computeSSP(EEG, info, threshold)
    EEG.add_proj(projs)
    EEG.apply_proj()
    EEG = EEG.get_data()
    
    return projs, EEG

def averageStable(stable):
    '''
    Averages neighboring epochs in pairs of two, such that an epoch_c is updated to: (epoch_c+epoch_{c+1})/2
    
    # Arguments
        stable: NumPy array
            Array of EEG data in the following format: [trials, channels, time samples].
    
    # Returns        
        stable_2mean: NumPy array
            Array of averaged EEG data in the following format: [trials, channels, time samples].

    '''
    stable_mid = (stable[:-1] + stable[1:])/2
    stable_2mean = np.zeros(stable.shape)
    stable_2mean[0] = stable[0]
    stable_2mean[1:] = stable_mid
    
    return stable_2mean

def removeEpochs(EEG, info, interpolate=0):
    '''
    Defines ranges for bad epochs and removes these epochs (currently not used).
    Possible to define how large a percentage can maximally be rejected.
    
    # Arguments
        EEG: NumPy array
            Array of EEG data in the following format: [trials, channels, time samples].
        
        info: MNE info structure
        
        interpolate: boolean
            If TRUE, interpolates bad epochs.
 
    # Returns
        EEG: NumPy array
            Array of EEG data in the following format: [trials, channels, time samples].
        
    '''
    # Estimate common peak-to-peak for channels and trials
    minAmp = np.min(EEG, axis=2)
    maxAmp = np.max(EEG, axis=2)    
    peak2peak = maxAmp - minAmp
    reject_thres = np.mean(peak2peak) + 4*np.std(peak2peak)*4
    reject = dict(eeg = reject_thres)
    flat_thres = np.mean(peak2peak)/100
    flat = dict(eeg = flat_thres)
    no_trials = EEG.shape[0]
    
    EEG = mne.EpochsArray(EEG, info, baseline=None)    
    epochs_copy = EEG.copy()
    epochs_copy.drop_bad(reject=reject, flat=flat, verbose=None)
    log = epochs_copy.drop_log
    
    if any(log):
        not_blank = np.concatenate([x for x in (log) if x])
        bad_channels, bad_counts = np.unique(not_blank, return_counts=True)

        p = 0.05
        bad_above_thres = []
        while p==0.05 or len(bad_above_thres)>5:
            thres = p*no_trials
            bad_above_thres = bad_channels[bad_counts>thres]
            p = p+0.05
   
        if len(bad_above_thres):
            EEG.info['bads'] = bad_above_thres
            print('Dropping channels:', bad_above_thres)
            if interpolate == 1:
                EEG.interpolate_bads(reset_bads = True)
            else:
                EEG.drop_channels(bad_above_thres)
                print('Not interpolating bad channels')
                
            # Repeat checking for bad epochs
            epochs_copy = EEG.copy()
            epochs_copy.drop_bad(reject=reject, flat=flat, verbose=None)
            log = epochs_copy.drop_log
        
        
        bad_epochs = [i for i,x in enumerate(log) if x]  
        if len(bad_epochs) > 0.20*no_trials:
            reject_thres = reject_thres*2
            reject = dict(eeg=reject_thres)
            flat_thres = flat_thres/2
            flat = dict(eeg=flat_thres)
            epochs_copy = EEG.copy()
            epochs_copy.drop_bad(reject=reject, flat=flat, verbose=None)
            log=epochs_copy.drop_log
            bad_epochs = [i for i,x in enumerate(log) if x]  
            if len(bad_epochs) > 0.60*no_trials:
                print('More than 60% are bad, keeping all epochs')
                log = []
        EEG.drop(bad_epochs,reason='Reject')
    else:
        bad_epochs = []
        bad_above_thres = []

    EEG = EEG.get_data()

    return EEG

def createInfoMNE(reject_ch=0, sfreq=100):
    '''
    Creates an MNE info data structure.
    
    # Arguments
        reject_ch: boolean
            Whether to reject predefined channels.
            
        sfreq: int
            Sampling frequency.
        
    # Returns
        info: MNE info structure
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
    
def preproc1epoch(eeg, info, projs=[], SSP=True, reject=None, mne_reject=1, reject_ch=None, flat=None, bad_channels=[] ,opt_detrend=1):
    '''
    Preprocesses epoched EEG data.
    
    # Arguments
        eeg: numPy array
            EEG epoch in the following format: [time samples, channels].
        
        info: MNE info structure. 
            Predefined info structure. Can be generated using createInfoMNE function.
            
        projs: list
            MNE SSP projector objects. Used if SSP = True. 
            
        SSP: boolean.
            Whether to apply SSP projectors (artefact correction) to the EEG epoch.
            
        reject: boolean
            Whether to reject channels, either manually defined or based on MNE analysis.
            
        mne_reject: boolean
            Whether to use MNE rejection based on the built-in function: epochs._is_good. 
            
        reject_ch: boolean
            Whether to reject nine predefined channels (can be changed to any channels).
            
        flat: boolean
            Input for the MNE built-in function: epochs._is_good. See function documentation.
            
        bad_channels: list
            Input for the MNE built-in function: epochs._is_good. Manual rejection of channels. See function documentation.
            
        opt_detrend: boolean
            Whether to apply temporal EEG detrending (linear).
    
    # Preprocessing steps - based on inputs 
    
        Linear temporal detrending
        
        Initial rejection of pre-defined channels 
        
        Bandpass filtering (currently 0-40 Hz, defined by variables: LP, HP, phase)
        
        Resampling to 100 Hz
        
        SSP artefact correction 
        
        Analysis and rejection of bad channels
            Interpolation of bad channels
            
        Average re-referencing
        
        Baseline correction
    
    # Returns
        epoch: NumPy array
            Preprocessed EEG epoch in NumPy array.
    
    '''
    
    n_samples = eeg.shape[0]
    n_channels = eeg.shape[1]
    eeg = np.reshape(eeg.T,(1,n_channels,n_samples))
    tmin = -0.1 # Baseline start, i.e. 100 ms before stimulus onset
    
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
    epoch.apply_baseline(baseline=(None,0), verbose=False)
    
    # Apply SSP projectors
    if SSP == True:
        epoch.add_proj(projs)
        epoch.apply_proj()
        
    if reject is not None: # Rejection of channels, either manually defined or based on MNE analysis. Currently not used.
        if mne_reject == 1: # Use MNE method to reject+interpolate bad channels
            from mne.epochs import _is_good
            from mne.io.pick import channel_indices_by_type    
            # reject=dict(eeg=100)
            idx_by_type = channel_indices_by_type(epoch.info)
            A,bad_channels = _is_good(epoch.get_data()[0], epoch.ch_names, channel_type_idx=idx_by_type, reject=reject, flat=flat, full_report=True)
            print(A)
            if A == False:
                epoch.info['bads'] = bad_channels    
                epoch.interpolate_bads(reset_bads=True, verbose=False)
        else: # Predefined bad_channels 
            epoch.drop_channels(bad_channels)
    
    # Re-referencing
    epoch.set_eeg_reference(verbose=False)
    
    # Apply baseline after rereference
    epoch.apply_baseline(baseline=(None,0), verbose=False)
        
    epoch = epoch.get_data()[0]
    
    return epoch

#### CLASSIFICATION FUNCTIONS ####

def extractCat(indicesFile, exp_type='fused'):
    ''' 
    Extracts .csv file with file directories of shown experimental trials (two images for each trial).
    
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


def trainLogReg(X,y):
    '''
    Trains a classifier on EEG epochs from stable blocks: (n_trials,n_channels,n_samples)
    No offset for correction of classifier bias.

    Input:
    - X: Epochs. 
    - y: Binary category array.
    
    Output:
    - clf: SAGA LogReg with L1 penalty, fitted on X and y.
    '''
    
    X = scale2DArray(X,axis=1)
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1')
    
    clf = classifier.fit(X,y)

#    pred_prob =  clf.predict_proba(X) 
#    score_train = clf.score(X,y) 
    
    return clf


def trainLogReg_cross(X,y):
    '''
    Trains a classifier on EEG epochs from stable blocks: (600,n_channels,n_samples). 
    For the first NF run (n_run=0).
    Offset for correction of classifier bias computed in 3-fold cross-validation (mean of 3 offset values)

    Input:
    - X: Epochs (after preprocessing, SSP correction and two-trial averaging). Length 600 trials.
    - y: Binary category array.
    
    Output:
    - clf: SAGA LogReg with L1 penalty, fitted on X and y.
    - offset: Offset value for correction of classification bias.
    '''
    
    X = scale2DArray(X,axis=1)
    
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1',max_iter=100)
    
    # score = []
    # conf = []
    
    # Estimation of offset for correction of classifier bias. 3-fold cross-validation. 
    # Cross-validation fold number 1.
    X_train = X[0:400,:]
    y_train = y[0:400]
    
    clf = classifier.fit(X_train,y_train)
    X_val = X[400:,:]
    # y_val = y[400:]
    pred_prob_val = clf.predict_proba(X_val)
    pred_sort = np.sort(pred_prob_val[:,0])
    offset1 = 0.5 - pred_sort[100]
    
    # Following piece of code can be uncommented for validation accuracy and confusion matrices.
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    # Cross-validation fold number 2.
    X_train = np.concatenate((X[0:200,:],X[400:600,:]),axis=0)
    y_train = np.concatenate((y[0:200],y[400:600]),axis=0)
    clf = classifier.fit(X_train,y_train)
    X_val = X[200:400,:]
    # y_val = y[200:400]
    pred_prob_val = clf.predict_proba(X_val)
    pred_sort = np.sort(pred_prob_val[:,0])
    offset2 = 0.5-pred_sort[100]
    
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    # Cross-validation fold number 3.
    X_train = X[200:,:]
    y_train = y[200:]
    clf = classifier.fit(X_train,y_train)
    X_val = X[:200,:]
    # y_val = y[:200]
    pred_prob_val = clf.predict_proba(X_val)
    
    pred_sort = np.sort(pred_prob_val[:,0])
    offset3 = 0.5-pred_sort[100]
    
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    offset = (offset1+offset2+offset3)/3
    
    clf = classifier.fit(X,y)

    # pred_prob = clf.predict_proba(X) 
    # score_train = clf.score(X,y) 
    
    return clf, offset #,score,conf

def trainLogReg_cross2(X,y):
    '''
    Trains a classifier on EEG epochs from stable blocks and recent NF blocks: (800,n_channels,n_samples)
    The offset is computed based on the most recent 200 trials.
    
    Input:
    - X: Epochs (after preprocessing, SSP correction and two-trial averaging). Length 800 trials.
    - y: Binary category array.
    
    Output:
    - clf: SAGA LogReg with L1 penalty, fitted on X and y.
    - offset: Offset value for correction of classification bias.
    '''
    
    X = scale2DArray(X,axis=1)
    
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1',max_iter=100)
    
    # score = []
    # conf = []
    X_train = X[0:600,:]
    y_train = y[0:600]
    
    clf = classifier.fit(X_train,y_train)
    X_val = X[600:,:]
    # y_val = y[600:]
    pred_prob_val = clf.predict_proba(X_val)
    pred_sort = np.sort(pred_prob_val[:,0])
    offset = 0.5-pred_sort[100]
    
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    clf = classifier.fit(X,y)

#    pred_prob =  clf.predict_proba(X) 
#    score_train = clf.score(X,y) 
    
    return clf, offset #,score,conf

def testEpoch(clf,epoch,y=None):
    '''
    Tests an epoch based on a trained classifier. 

    Input:
    - clf: Trained classifier.
    - X: Epoch (after preprocessing, SSP correction and averaging).
    
    Output:
    - pred_prob: Prediction probability of classifier.
    - clf.predict(epoch): Predicted, binary value of the classifier.
    '''
    
    epoch = scale1DArray(epoch)
    # y_pred = clf.predict(epoch)
    
    pred_prob =  clf.predict_proba(epoch)

    return pred_prob, clf.predict(epoch) 


def sigmoid_base(x, A, B, C, D):
    '''
    Sigmoid 4 parameter base.
    '''
    
    return ((A-D)/(1+10**((C-x)*B))) + D

def sigmoid(x):
    '''
    Transfer function for mapping prediction probability to alpha value between 0.17 and 0.98.
    Transfer function not continuous.
    
    Input:
    - x: Float between -1 and 1, prediction probability value.
    
    Output:
    - alpha: Float between 0.17 and 0.98, mapped using the sigmoid function with specified parameters.
    
    '''
    
    offset1 = 0.0507
    offset2 = 0.0348
    A1 = 0.98 + offset1
    B = 1.3 # Steepness
    C = 0 # Inflection point
    D1 = 1 - A1 # 0.17
    D2 = 0.17 - offset2
    A2 = 1 - D2
    
    if x >= 0:
        alpha = sigmoid_base(x,A1,B,C,D1)

    else:
        alpha = sigmoid_base(x,A2,B,C,D2)
        
    return alpha

