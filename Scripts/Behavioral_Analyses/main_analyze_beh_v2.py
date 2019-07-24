# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:38:53 2019

@author: Greta

# Variables: subjID, day, (min/max second values), 

# Output: Average response time overall and divided into lures/non-lures. 
Lure: Number of correct inhibitions (CR), and number of not correctly inhibited responses (FR)
Non-lure: Number of false inhibitions.

"""
#%% Only run once for each subject

# Imports
import glob
import os
os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\')
import numpy as np
import csv
from matplotlib import pyplot as plt
import collections
import pickle
from responseTime_func import * 

subjID = '26'

# Initialize dict
d = {}
d['subjID'] = subjID

#%%
for day in ['1','3','4','5']:
   
    expDay = day
    
    dataDir = 'P:\\closed_loop_data\\' + str(subjID) + '\\'
    os.chdir(dataDir)
    
    fileLst = glob.glob(dataDir + '/*.csv') 
    
    catFile, stimuliFile, keypressFile = findFiles(fileLst,expDay)
    
    d['imageTimeFile_day_'+str(expDay)] = stimuliFile
    d['keypressFile_day_'+str(expDay)] = keypressFile
    
    #################### Extract keypress and stimuli time data ###################
    data1 = extractInfo(keypressFile)
    data2 = extractInfo(stimuliFile)
    
    split1 = splitString(data1)
    
    keypressTimes, warningLst1 = passWarning(split1)
    stimuliTimes = extractStimuliTimes(data2)
    
    # Matching the times
    nStimuli = len(stimuliTimes)
    nKeypress = len(keypressTimes)
    
    if nStimuli != 800:
        print('WARNING, number of createIndices not 800')
        raise ValueError
    
    if 500 > nKeypress > 1100:
        print('WARNING, number of keypresses too low/high')
        raise ValueError
    
    responseTimes,pairs = matchTimes(stimuliTimes, keypressTimes, 0.15, 1.15)
    
    #d['responseTimes'] = responseTimes
    
    ################## Extract lure indices and RTs #####################
    lureIdx, non_lureIdx, lure_RT, nonlure_RT, CR_idx, FR_idx = findRTs(catFile,responseTimes)
    
    #d['lure_RT'] = lure_RT
    #d['nonlure_RT'] = nonlure_RT
    
    # Count how many lures are inhibited correctly
    lure_RT_copy = np.copy(lure_RT)
    lure_RT_count = (~np.isnan(lure_RT_copy)) # Assigns False to nan, i.e. correctly inhibited, and True to falsely inhibited (a respone time recorded)
    
    unique_lure, counts_lure = np.unique(lure_RT_count, return_counts=True)
    no_CI_lure = (counts_lure[0]) # No. correct inhibitions
    no_NI_lure = (counts_lure[1]) # No. not inhibited
    
    d['inhibitions_lure_day_'+str(expDay)] = no_CI_lure
    d['no_Inhibitions_lure_day_'+str(expDay)] = no_NI_lure
    
    # Count how many non-lures are inhibited 
    nonlure_RT_copy = np.copy(nonlure_RT)
    nonlure_RT_count = (~np.isnan(nonlure_RT_copy)) # Assigns False to nan, i.e. correctly inhibited, and True to falsely inhibited (a respone time recorded)
    
    unique_nonlure, counts_nonlure = np.unique(nonlure_RT_count, return_counts=True)
    no_CI_nlure = (counts_nonlure[0]) # No. inhibitions, thus a keypress was withheld during a non-lure stimuli
    no_NI_nlure = (counts_nonlure[1]) # No. not inhibited, correct keypress
    
    d['inhibitions_nonlure_day_'+str(expDay)] = no_CI_nlure
    d['no_Inhibitions_nonlure_day_'+str(expDay)] = no_NI_nlure
    
    # Mean of lure RTs, and mean of non-lure RTs (not including inhibited responses)
    lure_RT_mean = np.nanmean(lure_RT)
    nonlure_RT_mean = np.nanmean(nonlure_RT)
    
    d['lure_RT_mean_day_'+str(expDay)] = lure_RT_mean
    d['nonlure_RT_mean_day_'+str(expDay)] = nonlure_RT_mean
    
    
    print(d)
    
    ################# Check RTs based on either correct or false response (surrounding that lure) ###############
    
    # For each experimental day, check RT around lure (-3 to 3), and add these mean vals to dictionary.
    
    surrounding_CR_Lst = []
    surrounding_FR_Lst = []
    
    lure_RT_add,surrounding_CR,surrounding_CR_mean,surrounding_FR,surrounding_FR_mean = RTaroundLure(lureIdx, lure_RT, responseTimes,'b',3,CR_idx,FR_idx)    
    surrounding_CR_Lst.append(surrounding_CR_mean)
    surrounding_FR_Lst.append(surrounding_FR_mean)
    
    lure_RT_add,surrounding_CR,surrounding_CR_mean,surrounding_FR,surrounding_FR_mean = RTaroundLure(lureIdx, lure_RT, responseTimes,'b',2,CR_idx,FR_idx)    
    surrounding_CR_Lst.append(surrounding_CR_mean)
    surrounding_FR_Lst.append(surrounding_FR_mean)
    
    lure_RT_add,surrounding_CR,surrounding_CR_mean,surrounding_FR,surrounding_FR_mean = RTaroundLure(lureIdx, lure_RT, responseTimes,'b',1,CR_idx,FR_idx)    
    surrounding_CR_Lst.append(surrounding_CR_mean)
    surrounding_FR_Lst.append(surrounding_FR_mean)
    
    lure_RT_add,surrounding_CR,surrounding_CR_mean,surrounding_FR,surrounding_FR_mean = RTaroundLure(lureIdx, lure_RT, responseTimes,'b',0,CR_idx,FR_idx)    
    surrounding_CR_Lst.append(surrounding_CR_mean)
    surrounding_FR_Lst.append(surrounding_FR_mean)
    
    lure_RT_add,surrounding_CR,surrounding_CR_mean,surrounding_FR,surrounding_FR_mean = RTaroundLure(lureIdx, lure_RT, responseTimes,'a',1,CR_idx,FR_idx)    
    surrounding_CR_Lst.append(surrounding_CR_mean)
    surrounding_FR_Lst.append(surrounding_FR_mean)
    
    lure_RT_add,surrounding_CR,surrounding_CR_mean,surrounding_FR,surrounding_FR_mean = RTaroundLure(lureIdx, lure_RT, responseTimes,'a',2,CR_idx,FR_idx)    
    surrounding_CR_Lst.append(surrounding_CR_mean)
    surrounding_FR_Lst.append(surrounding_FR_mean)
    
    lure_RT_add,surrounding_CR,surrounding_CR_mean,surrounding_FR,surrounding_FR_mean = RTaroundLure(lureIdx, lure_RT, responseTimes,'a',3,CR_idx,FR_idx)    
    surrounding_CR_Lst.append(surrounding_CR_mean)
    surrounding_FR_Lst.append(surrounding_FR_mean)
    
    ticks = ['-3','-2','-1','lure','1','2','3']
    
    plt.figure()
    plt.plot(surrounding_CR_Lst,color='green')
    plt.plot(surrounding_FR_Lst,color='red')
    plt.xticks(np.arange(0,7,1),ticks)
    
    d['surrounding_CR_Lst_day_'+str(expDay)] = surrounding_CR_Lst
    d['surrounding_FR_Lst_day_'+str(expDay)] = surrounding_FR_Lst
    
    
    del catFile, stimuliFile, keypressFile
    
    
    if expDay == '5':
        pkl_arr = [d]
    
        saveDir = 'P:\\closed_loop_data\\beh_analysis\\'
        os.chdir(saveDir)
    
    # PICKLE TIME
        fname = 'Beh_subjID_'+str(subjID)+'.pkl'
        with open(fname, 'wb') as fout:
            pickle.dump(pkl_arr, fout)


