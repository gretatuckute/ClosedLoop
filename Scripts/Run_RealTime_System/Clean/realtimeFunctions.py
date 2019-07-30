# -*- coding: utf-8 -*-

'''
Functions for working with real-time EEG data in Python:
Standardizing, scaling, artefact correction (SSP projections), preprocessing, classification.

The first part contains EEG signal processing and preprocessing, while the second part is EEG decoding in real-time.
'''

# Imports 
import numpy as np
from scipy.stats import zscore
import os 
import mne
from scipy.signal import detrend
from sklearn.linear_model import LogisticRegression

#### 1) EEG PREPROCESSING FUNCTIONS ####

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
    
def preproc1epoch(eeg, info, projs=[], SSP=True, reject=None, mne_reject=1, reject_ch=None, flat=None, bad_channels=[] ,opt_detrend=1, HP=0, LP=40, phase='zero-double'):
    '''    
    Preprocesses EEG data epoch-wise. 
    
    # Arguments
        eeg: numPy array
            EEG epoch in the following format: [time samples, channels].
        
        info: MNE info structure. 
            Predefined info structure. Can be generated using createInfoMNE function.
            
        projs: list
            MNE SSP projector objects. Used if SSP = True. 
            
        SSP: boolean
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
        
        HP: int
            High-pass filter cut-off, default 0 Hz.
        
        LP: int
            Low-pass filter cut-off, default 40 Hz.

        phase: string
            FIR filter phase (refer to MNE filtering function for options), default 'zero-double'.

    
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

def extractCat(indicesFile):
    ''' 
    Extracts information from a .csv file with file directories of shown experimental trials (two images for each trial).
    
    # Arguments
        indicesFile: csv file
            A .csv file containing experimental trials as rows, and the following columns: 
            1) Experimental trial number, 2) Attentive category string name, e.g. 'indoor' or 'female',
            3) Binary category label, i.e. 0 or 1, 4) String of file directory to image 1,
            5) String of file directory to image 2.
        
    # Returns
        binary_categories: list
            List with 0 denoting scenes, and 1 denoting faces.
    
    '''
    
    colnames = ['1', 'att_cat', 'binary_cat', '3', '4']
    data = pd.read_csv(indicesFile, names=colnames)
    categories = data.att_cat.tolist()
    binary_categories = data.binary_cat.tolist()
    del categories[0:1]
    del binary_categories[0:1]

    return binary_categories


def trainLogReg(X, y):
    '''
    Trains a logistic regression classifier based on recorded EEG epochs. 
    No offset for correction of classifier bias.

    # Arguments
        X: NumPy array
            EEG data in the following format: [trials, channels, samples]
            
        y: NumPy Array
            Binary category array.
    
    # Returns
        clf: scikit-learn classifier object
            SAGA LogReg with L1 penalty, fitted on X and y.
    '''
    
    X = scale2DArray(X, axis=1)
    classifier = LogisticRegression(solver='saga', C=1, random_state=1, penalty='l1')
    
    clf = classifier.fit(X,y)

#    pred_prob =  clf.predict_proba(X) 
#    score_train = clf.score(X,y) 
    
    return clf


def trainLogRegCV(X, y):
    '''
    Trains a logistic regression classifier based on recorded EEG epochs. 
    For the first neurofeedback run (n_run = 0), using 600 trials.

    The classifier bias is corrected (i.e. tendency of the classifier to be overly confident in predicting one of the categories):
    The offset for correction of classifier bias is computed in 3-fold cross-validation (mean of 3 offset values).
    
    For validation accuracy and confusion matrices, code in the function can be uncommented.

    # Arguments
        X: NumPy array
            EEG data in the following format: [trials, channels, samples]
            
        y: NumPy Array
            Binary category array.
    
    # Returns
        clf: scikit-learn classifier object
            SAGA LogReg with L1 penalty, fitted on X and y.
            
        offset: float
            Float value for correcting the prediction probability of the classifier.
    '''
    X = scale2DArray(X, axis=1)
    
    classifier = LogisticRegression(solver='saga', C=1, random_state=1, penalty='l1', max_iter=100)
    
    # score = []
    # conf = []
    
    # Estimation of offset for correction of classifier bias. 3-fold cross-validation. 
    # Cross-validation fold number 1.
    X_train = X[0:400,:]
    y_train = y[0:400]
    
    clf = classifier.fit(X_train, y_train)
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
    X_train = np.concatenate((X[0:200,:], X[400:600,:]), axis=0)
    y_train = np.concatenate((y[0:200], y[400:600]), axis=0)
    clf = classifier.fit(X_train, y_train)
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
    clf = classifier.fit(X_train, y_train)
    X_val = X[:200,:]
    # y_val = y[:200]
    pred_prob_val = clf.predict_proba(X_val)
    
    pred_sort = np.sort(pred_prob_val[:,0])
    offset3 = 0.5-pred_sort[100]
    
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    offset = (offset1+offset2+offset3)/3
    
    clf = classifier.fit(X, y)

    # pred_prob = clf.predict_proba(X) 
    # score_train = clf.score(X,y) 
    
    return clf, offset #,score,conf

def trainLogRegCV2(X, y):
    '''
    Trains a logistic regression classifier based on recorded EEG epochs. 
    For neurofeedback runs after the first one (n_run > 0), using 800 trials.

    The classifier bias is corrected (i.e. tendency of the classifier to be overly confident in predicting one of the categories):
    The offset for correction of classifier bias is computed based on the most recent 200 trials.
    
    For validation accuracy and confusion matrices, code in the function can be uncommented.

    # Arguments
        X: NumPy array
            EEG data in the following format: [trials, channels, samples]
            
        y: NumPy Array
            Binary category array.
    
    # Returns
        clf: scikit-learn classifier object
            SAGA LogReg with L1 penalty, fitted on X and y.
            
        offset: float
            Float value for correcting the prediction probability of the classifier.
    '''   
    X = scale2DArray(X, axis=1)
    
    classifier = LogisticRegression(solver='saga', C=1, random_state=1, penalty='l1', max_iter=100)
    
    # score = []
    # conf = []
    X_train = X[0:600,:]
    y_train = y[0:600]
    
    clf = classifier.fit(X_train, y_train)
    X_val = X[600:,:]
    # y_val = y[600:]
    pred_prob_val = clf.predict_proba(X_val)
    pred_sort = np.sort(pred_prob_val[:,0])
    offset = 0.5-pred_sort[100]
    
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    clf = classifier.fit(X, y)

#    pred_prob =  clf.predict_proba(X) 
#    score_train = clf.score(X,y) 
    
    return clf, offset #,score,conf

def testEpoch(clf, epoch):
    '''
    Classifies an epoch based on a trained classifier. 

    # Arguments
        clf: scikit-learn classifier object
        
        epoch: NumPy array
            EEG Epoch in the following format: [channels, samples].
    
    # Returns
        pred_prob: float (range = {0;1})
            Prediction probability of classifier.
        
        pred: int (0 or 1)
            Predicted, binary value of the classifier.
    '''
    
    epoch = scale1DArray(epoch)
    # y_pred = clf.predict(epoch)
    
    pred_prob =  clf.predict_proba(epoch)
    pred = clf.predict(epoch) 

    return pred_prob, pred


def sigmoidBase(x, A, B, C, D):
    '''
    Sigmoid 4 parameter base.
    '''
    
    return ((A-D)/(1+10**((C-x)*B))) + D

def sigmoid(x):
    '''
    Transfer function for mapping prediction probability to alpha value between 0.17 and 0.98.
    Transfer function not continuous.
    
    # Arguments
        x: float (range = {-1,1}
            Classifier output value, 1 denoting a maximum correct prediction probability. 
    
    # Returns
        alpha: float (range = {0.17,0.98}
            Alpha value mapped using the sigmoid function with specified parameters.
    
    '''
    
    offset1 = 0.0507
    offset2 = 0.0348
    A1 = 0.98 + offset1
    B = 1.3 # Steepness
    C = 0 # Inflection point
    D1 = 1 - A1 
    D2 = 0.17 - offset2
    A2 = 1 - D2
    
    if x >= 0:
        alpha = sigmoidBase(x,A1,B,C,D1)

    else:
        alpha = sigmoidBase(x,A2,B,C,D2)
        
    return alpha

