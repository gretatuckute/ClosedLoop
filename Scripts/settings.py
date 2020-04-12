# -*- coding: utf-8 -*-
'''
Set up experimental, recording, preprocessing and classification parameters.

Initialization of paths for system scripts, subjects directory, and data directory.
'''

import os

###### EXPERIMENTAL VARIABLES (experimentFunctions) ######
subjID = '01' 
expDay = '2'  # Neurofeedback day
monitor_size = [1910, 1070]

# Stimuli presentation times (in Hz)
frameRate = 60
probeTime = frameRate * 6 # Frames to display probe word
fixTime = frameRate * 2 # Frames to display fixation cross
stimTime = frameRate # Frames for presenting each image

# Experiment structure and length
numRuns = 5 # Number of neurofeedback runs
numBlocks = 8 # Number of blocks within each run
blockLen = 50 # Number of trials within each block



###### EEG SAMPLING (runSystem) ###### 
samplingRate = 500 
samplingRateResample = 100
baselineTime = -0.1 # Baseline for each epoch (i.e. before stimuli onset) (seconds)
epochTime = 0.800 # Seconds (i.e. after stimuli onset) (seconds)

maxBufferData = 2 # Maximum amount of data to store in the buffer (seconds)

# All channels in the system
channelNames = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1',\
                'T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']

# If manually preselecting channels
rejectChannels = True
channelNamesSelected =  ['P7','P4','Cz','Pz','P3','P8','O1','O2','C4','F4','C3','F3','Oz','PO3','FC5',\
                         'FC1','CP5','CP1','CP2','CP6','FC2','FC6','PO4']
channelNamesExcluded = list(set(channelNames) - set(channelNamesSelected))



###### EEG preprocessing (realtimeFunctions) ###### 
highpass = 0  # Hz
lowpass = 40 # Hz
detrend = True # Temporal detrending of EEG signal (linear)
filterPhase = 'zero-double'
montage = 'standard_1020' 
SSP = True # Whether to apply SSP artifact correction
thresholdSSP = 0.1 # SSP variance threshold for rejection of SSP projections



###### EEG real-time classification (realtimeFunctions) ###### 
from sklearn.linear_model import LogisticRegression # Import and specify classifier of interest
classifier = LogisticRegression(solver='saga', C=1, random_state=1, penalty='l1', max_iter=100)



###### DIRECTORY STRUCTURE ###### 
def base_dir_init(): # Base directory for ClosedLoop GitHub
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return base_dir

def script_path_init():
    script_path = base_dir_init() + '\Scripts'
    return script_path

def data_path_init(): # Data (images) storage directory
    data_path = base_dir_init() + '\imageStimuli'
    return data_path

def subject_path_init(): # Subjects directory, for storing EEG data 
    subject_path = base_dir_init() + '\subjectsData'
    return subject_path

if __name__ == '__main__':
    base_dir = base_dir_init()
    print('====== Current directory ======')
    print(base_dir)
   