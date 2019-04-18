# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:02:15 2019

@author: Greta
"""

#%% Imports
import pickle
from matplotlib import pyplot as plt
import itertools
import os
import numpy as np
import mne
from scipy.signal import detrend

#%% Functions
def createInfo(reject_ch=False,sfreq=100):
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


def preprocEpoch(eeg,info,reject_ch=None,HP=0,LP=40,opt_detrend=1):

    '''
    Preprocesses epoched EEG data.
    
    # Input
    - eeg: numPy array. EEG epoch in the following format: (time samples, channels).
    - info: MNE info data structure. Predefined info containing channels etc. Can be generated using create_info_mne function.
    - reject_ch: bool. Whether to reject nine predefined channels.
    - HP: int. Offset value for FIR highpass filter.
    - LP: int. Offset value for FIR lowpass filter.
    - opt_detrend: bool. Whether to apply temporal EEG detrending (linear).
    
    # Preprocessing steps - based on inputs 
    - Linear temporal detrending
    - Rejection of initial, predefined channels 
    - Bandpass filter (default 0-40Hz)
    - Resample to 100Hz
    - Rereference to average
    - Baseline correction
    
    # Output
    - epoch: numPy array (channels, time samples). Preprocessed EEG epoch.
    
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
    epoch.filter(HP, LP, fir_design='firwin', phase='zero-double', verbose=False)
    
    # Downsample
    epoch.resample(100, npad='auto',verbose=False)
    
    # Apply baseline correction
    epoch.apply_baseline(baseline=(None,0),verbose=False)
    
    # Rereferencing
    epoch.set_eeg_reference(verbose=False)
    
    # Apply baseline after rereference
    epoch.apply_baseline(baseline=(None,0),verbose=False)
        
    epoch = epoch.get_data()[0]
    
    return epoch


def createEpochsArray(EEG,info,add_events=None):
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
    
    else:
        EEG = mne.EpochsArray(EEG, info, baseline=None, tmin=-0.1)
    
    return EEG

def extractEvokeds(epochsArray,add_events=None):
    
    if add_events is not None:  
        evokeds = [epochsArray[name].average() for name in ('scene','face')]
        
    else:
        evokeds = [epochsArray.average()]
        
    return evokeds



#%% Perform preprocessing and SSP on all the stable blocks.

# EEG is a file with epoched EEG data in the format (trials,time samples, channels)
    # y

# MNE info files
info_fs500 = createInfo(reject_ch=False,sfreq=500)
info_fs100 = createInfo(reject_ch=True,sfreq=100)

EEG_plot = np.zeros((EEG.shape[0]),(EEG.shape[2]),(EEG.shape[1])) 

for trial in range(EEG.shape[0]):
    epoch = stable_blocks[trial,:,:]
    epoch = preprocEpoch(epoch,info_fs500)
    EEG_plot[trial,:,:] = epoch

EEG_array = createEpochsArray(EEG_plot,info_fs100,add_events=None)

evokeds = extractEvokeds(EEG_array,add_events=None)

# Make animation
# Save animation

# If events added, they can be animated e.g. evokeds[0] for the first condition.

fig,anim = evokeds.animate_topomap(times=np.linspace(0.00, 0.79, 50),frame_rate=10,blit=False)
anim.save('Brainmation.gif', writer='imagemagick', fps=10)