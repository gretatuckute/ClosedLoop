'''
Toolbox for creating GIFs in Python based on neural time series data using MNE, Matplotlib, SciPy and NumPy.

@author: Greta Tuckute, April 2019
'''

#%% Imports

from matplotlib import pyplot as plt
import numpy as np
import mne
from scipy.signal import detrend

#%% Functions

def createInfo(channel_names,fs=100):
    '''
    Creates an MNE info data structure.
    
    # Input:
        channel_names: List containing names of (M)EEG channels as strings.
        fs: int. Sampling frequency (default 100Hz).
        
    # Output:
        info: MNE info data structure.
    '''
    
    channel_types = ['eeg']*(len(channel_names))
    montage = 'standard_1020' 
    
    info = mne.create_info(channel_names, fs, channel_types, montage)
    
    return info


def preprocEpoch(eeg,info,tmin=0,HP=0,LP=40,linear_detrend=1):

    '''
    Preprocesses epoched (M)EEG data.
    
    # Input
    - eeg: numPy array. (M)EEG epoch in the following format: (time samples, channels).
    - info: MNE info data structure. Predefined info containing channels etc. Can be generated using createInfo function.
    - tmin: float. Start time before event, e.g. -0.1 (default 0).
    - HP: int. Offset value for FIR highpass filter.
    - LP: int. Offset value for FIR lowpass filter.
    - linear_detrend: bool. Whether to apply temporal EEG detrending (linear).
    
    # Preprocessing steps - based on inputs 
    - Linear temporal detrending
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
    
    # Temporal detrending:
    if linear_detrend == 1:
        eeg = detrend(eeg, axis=2, type='linear')
        
    epoch = mne.EpochsArray(eeg, info, tmin=tmin, baseline=None, verbose=False)
    
    # Bandpass filtering
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


def createEpochsArray(EEG,info,tmin=0,add_events=None,event_id=None):
    '''
    Computes and applies SSP projections of epoched EEG data (bandpass filtered and resampled to 100Hz). 
    
    # Input
    - EEG: numPy array. Epoched and processed (M)EEG in the following format: (trials, channels, time samples).
    - info: mne info struct.
    - tmin: float. Start time before event, e.g. -0.1 (default 0).
    - add_events: numPy array with integers indicative of events/categories.
    - event_id: dictionary containing event/category name as key, and integer value as key. E.g. dict(scene=0, face=1)
    
    # Output
    - EEG in MNE EpochsArray structure. If add_events is not none, events/categories are added to the MNE EpochsArray.
    
    '''
    
    if add_events is not None:
        events_list = add_events
        n_epochs = len(events_list)
        events_list = [int(i) for i in events_list]
        events_add = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int), events_list]
    
        EEG = mne.EpochsArray(EEG, info, baseline=None, tmin=tmin,events=events_add, event_id=event_id)
    
    else:
        EEG = mne.EpochsArray(EEG, info, baseline=None, tmin=tmin)
    
    return EEG

def extractEvoked(epochsArray,add_events=None,event_id=None):
    
    event_tuple = tuple(event_id.keys())
    
    if add_events is not None:  
        evokedArray = [epochsArray[name].average() for name in event_tuple]
        
    else:
        evokedArray = epochsArray.average()
        
    return evokedArray


#%% Load data, preprocess M(EEG) and create GIF
    
# Load sample data. 
# EEG.npy is an array with epoched EEG data in the format (trials, time samples, channels). 
# y.npy is a binary array with events/conditions.

EEG = np.load('EEG.npy')
y = np.load('y.npy')

# Create MNE info files
# channel_names contains channel names for a standard 1020 EEG montage.
channel_names = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']

info_fs500 = createInfo(channel_names=channel_names,fs=500) # Info structure for a sampling rate of 500Hz.
info_fs100 = createInfo(channel_names=channel_names,fs=100) # Info structure for a sampling rate of 100Hz.

resampled_fs = int((EEG.shape[1])/5) # Number of time samples after downsampling from 500Hz to 100Hz in preprocEpoch function.

# Preprocess EEG data epoch-wise.
EEG_plot = np.zeros(((EEG.shape[0]),(EEG.shape[2]),resampled_fs)) 

for trial in range(EEG.shape[0]):
    epoch = EEG[trial,:,:]
    epoch = preprocEpoch(epoch,info_fs500,tmin=-0.1)
    EEG_plot[trial,:,:] = epoch

# Convert EEG_plot numPy array into an MNE EpochsArray structure
EEG_array = createEpochsArray(EEG_plot,info_fs100,tmin=-0.1,add_events=None)

# Convert MNE EpochsArray structure to an Evoked structure (averaged across trials, or averaged across events/conditions)
evokedArray = extractEvoked(EEG_array,add_events=None)


# Make animation
times = np.linspace(0.00, 0.79, 50) # Manually defined time points for animation (in seconds)
fig, anim = evokedArray.animate_topomap(times=times,frame_rate=10,blit=False)

# If events/conditions are added, an animation can be created based on various conditions by indexing the evokedArray, e.g.: evokedArray[0] for the first condition.

# Save animation as GIF 
anim.save('Brainmation.gif', writer='imagemagick', fps=10)









