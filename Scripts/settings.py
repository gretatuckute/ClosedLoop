# -*- coding: utf-8 -*-
'''
Initialization of paths for system scripts, subjects directory, and data directory.
'''

import os

## DEFINE VARIABLES FOR THE EXPERIMENT (experimentFunctions) ##
subjID = '01' 
expDay = '2'  # feedback day
monitor_size = [1910, 1070]

# Initializing stimuli presentation times (in Hz)
frameRate = 60
probeTime = frameRate * 6 # frames to display probe word
fixTime = frameRate * 2 # frames to display fixation cross
stimTime = frameRate # frames for presenting each image


numRuns = 5 # Number of neurofeedback runs
numBlocks = 8 # Number of blocks within each run
blockLen = 50 # Number of trials within each block



## EEG preprocessing (realtimeFunctions)
highpass = 0
lowpass = 40 
filterPhase = 'zero-double'

montage = 'standard_1020' 

## runSystem
samplingRate = 500 
samplingRateResample = 100
baselineTime = -0.1 # Baseline for each epoch (i.e. before stimuli onset) (seconds)
epochTime = 0.800 # Seconds (i.e. after stimuli onset) (seconds)

maxBufferData = 2 # Maximum amount of data to store in the buffer (seconds)

# All channels in the system
channelNames = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']
# If manually preselecting channels
channelNamesSelected =  ['P7','P4','Cz','Pz','P3','P8','O1','O2','C4','F4','C3','F3','Oz','PO3','FC5','FC1','CP5','CP1','CP2','CP6','FC2','FC6','PO4']
channelNamesExcluded = list(set(channelNames) - set(channelNamesSelected))





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
   