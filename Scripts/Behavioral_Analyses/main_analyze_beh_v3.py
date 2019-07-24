# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:38:53 2019

@author: Greta

# Variables: subjID, day, (min/max second values), 

# Output: Average response time overall and divided into lures/non-lures. 
Lure: Number of correct inhibitions (CR), and number of not correctly inhibited responses (FR)
Non-lure: Number of false inhibitions.

V3 is the extended version.

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
from scipy import stats


#%% 

for subjID in subjID_all[:2]:

    # Initialize dict
    d = {}
    d['subjID'] = subjID
    
    for day in ['1','2','3','4','5']:
       
        expDay = day
        
        dataDir = 'P:\\closed_loop_data\\' + str(subjID) + '\\'
        os.chdir(dataDir)
        
        fileLst = glob.glob(dataDir + '/*.csv') 
        
        catFile, stimuliFile, keypressFile = findFiles(fileLst,expDay)
        
        d['FILE_imageTime_day'+str(expDay)] = stimuliFile
        d['FILE_keypress_day'+str(expDay)] = keypressFile
        
        #################### Extract keypress and stimuli time data ###################
        data1 = extractInfo(keypressFile)
        data2 = extractInfo(stimuliFile)
        
        split1 = splitString(data1)
        
        keypressTimes, warningLst1 = passWarning(split1)
        stimuliTimes = extractStimuliTimes(data2)
    
        # Matching the times
        nStimuli = len(stimuliTimes)
        nKeypress = len(keypressTimes)
        
        d['TOTAL_keypress_day'+str(expDay)] = nKeypress
        
        responseTimes,pairs = matchTimes(stimuliTimes, keypressTimes, 0, 1)
        
        d['responseTimes_day'+str(expDay)] = responseTimes
        
        lureIdx, non_lureIdx, lure_RT, nonlure_RT, CR_idx, FR_idx = findRTs(catFile,responseTimes)
        
        # Append responseTimes to dict. Run findRTs afterwards, inside a function, that computes the below, and output sensitivity etc without intermediates
        # Add responsetimes trend
        
        # mask = ~np.isnan(responseTimes)
        # maskLen = np.sum(mask)
        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(maskLen),responseTimes[mask])
                
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
        
    #    plt.figure()
    #    plt.plot(surrounding_CR_Lst,color='green')
    #    plt.plot(surrounding_FR_Lst,color='red')
    #    plt.xticks(np.arange(0,7,1),ticks)
        
        d['surrounding_CR_Lst_day_'+str(expDay)] = surrounding_CR_Lst
        d['surrounding_FR_Lst_day_'+str(expDay)] = surrounding_FR_Lst
        
        del catFile, stimuliFile, keypressFile
        
        if expDay == '5':
            pkl_arr = [d]
        
            saveDir = 'P:\\closed_loop_data\\beh_analysis\\V3_0_1000\\'
            os.chdir(saveDir)
        
        # PICKLE TIME
            fname = 'BehV3_subjID_0_1000_'+str(subjID)+'.pkl'
            with open(fname, 'wb') as fout:
                pickle.dump(pkl_arr, fout)

#%% Investigate cutoff

intervalLst = []

for subjID in subjID_all:
    print(subjID)
    for day in ['1','3']:
       
        expDay = day
        
        dataDir = 'P:\\closed_loop_data\\' + str(subjID) + '\\'
        os.chdir(dataDir)
        
        fileLst = glob.glob(dataDir + '/*.csv') 
        
        catFile, stimuliFile, keypressFile = findFiles(fileLst,expDay)
        
        #################### Extract keypress and stimuli time data ###################
        data1 = extractInfo(keypressFile)
        data2 = extractInfo(stimuliFile)
        
        split1 = splitString(data1)
        
        keypressTimes, warningLst1 = passWarning(split1)
        stimuliTimes = extractStimuliTimes(data2)
        
        for stimTime in stimuliTimes:
            for keyTime in keypressTimes:
    
                if stimTime < keyTime and keyTime < stimTime+1:
                    intervalTime = keyTime - stimTime
                    intervalLst.append(intervalTime)
        
plt.figure(70)
plt.title('Response times respective to stimulus onset across participants')
plt.xlabel('Response time (s)')
plt.ylabel('Count')
plt.hist(intervalLst,bins=200,color='black')
