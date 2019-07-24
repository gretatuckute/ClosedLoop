
"""
Created on Mon Jan 14 11:38:12 2019

@author: Greta
"""

import os
os.chdir('P:\\closed_loop_data\\pilots\\02\\')
# os.chdir('C:\\Users\\nicped\\Documents\\GitLab\\project\\SUBJECTS\\02\\')

import numpy as np
import csv
from matplotlib import pyplot as plt
import pandas as pd
from scipy.spatial import distance
import collections


keypressFile = 'keypress_subjID_02_day_1_03-04-19_15-14.csv'
stimuliFile = 'imageTime_subjID_02_day_1_03-04-19_15-14.csv'
catFile = 'createIndices_02_day_1.csv'

#%%

def extractInfo(timeFile):
    data = []
    with open(timeFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for row in csv_reader:           
            data.append(row)
            
    return data
    
def splitString(data):
    splitLst = []
    for entry in data:
        for c in entry:
            split = c.split()
            splitLst.append(split)
    
    return splitLst

def passWarning(splitLst):
    times = []
    warningLst = []
    
    for entry in splitLst:
        print(entry)
        if len(entry) > 0:
            
            if entry[1] == 'WARNING' or entry[0] == 'but' or entry[1] == 'Using':
                warningLst.append(entry)

            else:
                times.append(float(entry[0]))
            
    return times, warningLst

def extractStimuliTimes(data):
    times = []
    
    for entry in data:
        if entry[1] == '0':
            pass
        else:
            times.append(float(entry[1]))
        
    return times

def extractCat(indicesFile):
    colnames = ['1', 'att_cat', 'bin_cat', 'img1', 'img2']
    data = pd.read_csv(indicesFile, names=colnames)
    dominant_cat = data.att_cat.tolist()
    del dominant_cat[0:1]
    
    shown_img = data.img1.tolist() # From img1
    del shown_img[0:1]
    
    shown_cat = []
    for entry in shown_img:
        split = entry.split('\\')
        # print(split)
        shown_cat.append(split[-2])

    return dominant_cat, shown_cat

def findClosestAfter(keyLst, stimTime, minVal, maxVal):
    '''
    Matches a time value (stimTime) to the closest value in a list of time values (keyLst).
    Conditions: 1) The stimTime must have a lower value than the matched time in keyLst, 
    and 2) The distance between matched items must be between a specific min and max value (minVal/maxVal). 
    
    
    # Input
    keyLst: List of time values
    stimTime: Float, time to be matched to the keyLst
    minVal: float or int, minimum value for matching
    maxVal: float or int, maximum value for matching
    
    # Output
    keyTimeClosest: the time value in keyLst that is most closely matched to stimTime based on given conditions
    
    '''
    
    newKeyLst = []
    
    for keyTime in keyLst:
        if keyTime > stimTime: # Ensure that keyTime is after stimTime
            if (keyTime - stimTime) <= maxVal and (keyTime - stimTime) >= minVal:
                newKeyLst.append(keyTime)
    
    if len(newKeyLst) > 0: 
        keyTimeClosest = min(newKeyLst, key=lambda x:abs(x-stimTime)) # Find the keypress that is closest to stimTime
    else:
        keyTimeClosest = 0 # If no keypress times satisfy the conditions, add 0 to the list 
    
    return keyTimeClosest

def matchTimes(stimLst, keyLst, minVal, maxVal):
    '''
    Matches time values (stimLst) to the closest value in a list of time values (keyLst).
    
    Conditions: 1) The stimTime must have a lower value than the matched time in keyLst, 
    and 2) The distance between matched items must be between a specific min and max value (minVal/maxVal). 
    
    # Input
    stimLst: List of time values
    keyLst: List of time values
    minVal: float or int, minimum value for matching
    maxVal: float or int, maximum value for matching
    
    # Output
    responseTimes: list of the subtracted, matched times in stimLst and keyLst
    pairs: Indices of matched pairs, in case a match was found
    
    '''
    keyLst_copy = keyLst[:]
    pairs = []
    responseTimes = []
    
    for idx, stimTime in enumerate(stimLst): # stimTime is an element in stimuli list
        keyTimeClosest = findClosestAfter(keyLst_copy, stimTime, minVal, maxVal) 
        
        if keyTimeClosest != 0:
            RT = keyTimeClosest - stimTime
            pairs.append([idx, keyLst.index(keyTimeClosest)])
        
        if keyTimeClosest == 0:
            RT = 0
        
        responseTimes.append(RT)
        
    responseTimes = np.asarray(responseTimes)
    responseTimes[responseTimes == 0] = np.nan

    return responseTimes,pairs


#%% Extract keypress and stimuli time data
data1 = extractInfo(keypressFile)
data2 = extractInfo(stimuliFile)

split1 = splitString(data1)

keypressTimes, warningLst1 = passWarning(split1)
stimuliTimes = extractStimuliTimes(data2)

# Matching the times
nStimuli = len(stimuliTimes)
nKeypress = len(keypressTimes)

responseTimes1,pairs1 = matchTimes(stimuliTimes, keypressTimes, 0.2, 1)

#%% Categories and lures

def findRTs(catFile, responseTimeLst):
    '''
    Extracts when lures where shown in the experiment, and matches response times to lures and non-lures.
    
    
    # Input
    catFile: category file for extraction of shown categories.
    responseTimeLst: list of response times for the shown, experimental stimuli
    
    # Output
    lureIdx: list of indices of lures shown in the experiment
    non_lureIdx: list of indicies of non-lures shown in the experiment
    lure_RT: list of response times for the lure stimuli
    nonlure_RT: list of response times for the non lure stimuli
    CR_idx: List, indices of correctly rejected lures (inhibited response)
    FR_idx: List, indices of falsely rejected lures (non-inhibited response)
    
    '''
    # Extract categories from category file
    domCats, shownCats = extractCat(catFile)
    
    lureLst = [] 
    lureIdx = [] # Lure indices 
    
    CRlst = []
    CR_idx = [] # Indices of correctly rejected lures

    # Figure out whether a shown stimuli is a lure 
    for count, entry in enumerate(domCats):
        if entry == shownCats[count]:
            lureLst.append('true')
        else:
            lureLst.append('lure')
            lureIdx.append(count)
        
    allIdx = range(domCats) 
    non_lureIdx = [x for x in allIdx if x not in lureIdx]

    # Response times of lures and non lures       
    lure_RT = np.zeros(len(lureIdx))
    nonlure_RT = np.zeros(len(non_lureIdx))

    for count, idx in enumerate(lureIdx):
        lure_RT[count] = responseTimeLst[idx]      
    
    for count, idx in enumerate(non_lureIdx):
        nonlure_RT[count] = responseTimeLst[idx]
        
        
    # Indices of correctly and falsely rejected lures
    # Correct reject, i.e. response to lure inhibited
    for val in lure_RT:
        if np.isnan(val):
            CRlst.append(True)
        else:
            CRlst.append(False)

    for counter,value in enumerate(CRlst):
        if value == True:
            CR_idx.append(counter)
    
    all_lures = range(len(lureIdx))
    
    FR_idx = [x for x in all_lures if x not in CR_idx] # false rejects, i.e. not inhibited
    
    # Add NaN values instead of zero
    lure_RT[lure_RT == 0] = np.nan
    nonlure_RT[nonlure_RT == 0] = np.nan
    
    return lureIdx, non_lureIdx, lure_RT, nonlure_RT, CR_idx, FR_idx

#%% Extract lure indices and RTs

lureIdx, non_lureIdx, lure_RT, nonlure_RT, CR_idx, FR_idx = findRTs(catFile,responseTimes1)

correctInhib = collections.Counter(lure_RT).most_common(1)
falseInhib = collections.Counter(nonlure_RT).most_common(1) #How many times a keypress was withheld during a non lure stimuli


#%% Check RTs based on either correct or false response (surrounding that lure)
# Extract RTs around lure idx. Extract vals before the lure

def RTaroundLure(lureIdx, lureRT, responseTimeLst, surrounding, number, CR_idx, FR_idx):
    '''
    Computes the response times surrounding lures based on whether it is before or after the lure (surrounding), 
    and how many trials from the lure it is (number).
    
    # Input
    lureIdx: List (length 80, with indices ranging 0 to 799), indices of experimental lure stimuli. 
    lureRT: List (length 80), response times for experimental lure stimuli
    responseTimeLst
    
    '''
    
    lure_RT_add = np.zeros(len(lureIdx))
    
    for count, idx in enumerate(lureIdx):
        if surrounding == 'b': #before    
            idxn = idx - number
                    
        if surrounding == 'a': #after
            idxn = idx + number
        
        print('lure idx new: ', idxn)
        
        if -1 < idxn < (len(responseTimeLst)-1): # Ensure that the response time is within the limits of the experiment
            
            print('Lure idx new available in response times lst')
            # Check whether idxn (new idx) is also a lure
            if idxn in lureIdx: 
                # Check whether the response time was also a lure. In that case, append None
                print('wups,coincide!')
                lure_RT_add[count] = None
                
            
            if idxn not in lureIdx:
                print('hugh, all good, no overlap')
                lure_RT_add[count] = responseTimeLst[idxn]
                
        else:
            print('Lure idx new NOT available in response times lst')
            lure_RT_add[count] = None
    
    # 
    
    surrounding_CR = np.zeros(len(CR_idx))
    surrounding_FR = np.zeros(len(FR_idx))
    
    for count, idx in enumerate(CR_idx):
        surrounding_CR[count] = lure_RT_add[idx]  

    surrounding_CR_mean = np.nanmean(surrounding_CR)
        
    for count, idx in enumerate(FR_idx):
        surrounding_FR[count] = lure_RT_add[idx] 
        
    surrounding_FR_mean = np.nanmean(surrounding_FR)
    
    
    return lure_RT_add, surrounding_CR, surrounding_CR_mean, surrounding_FR, surrounding_FR_mean
    
#%%

lure_RT_add1,surrounding_CR1, surrounding_CR_mean1, surrounding_FR1, surrounding_FR_mean1 = RTaroundLure(lureIdx, lure_RT, responseTimes1,'b',1,CR_idx,FR_idx)    
    

