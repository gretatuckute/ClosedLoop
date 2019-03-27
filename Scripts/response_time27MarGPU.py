# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:45:47 2019

@author: nicped
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:38:12 2019

@author: Greta
"""

import os
#os.chdir('P:\\closed_loop_data\\02\\')
os.chdir('C:\\Users\\nicped\\Documents\\GitLab\\project\\SUBJECTS\\02\\')


import numpy as np
import csv
from matplotlib import pyplot as plt
import pandas as pd
from scipy.spatial import distance
import collections


keypressFile = 'keypress_subjID_02_day_1_03-04-19_15-14.csv'
stimuliFile = 'imageTime_subjID_02_day_1_03-04-19_15-14.csv'
catFile = 'createIndices_02_day_1.csv'

#keypressFile = 'keypress_subjID_11_day_1_03-12-19_16-18.csv'
#stimuliFile = 'imageTime_subjID_11_day_1_03-12-19_16-18.csv'
#catFile = 'createIndices_11_day_1.csv'




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
        if keyTime > stimTime:
            if (keyTime - stimTime) <= maxVal and (keyTime - stimTime) >= minVal:
                newKeyLst.append(keyTime)
    
    if len(newKeyLst) > 0:
        keyTimeClosest = min(newKeyLst, key=lambda x:abs(x-stimTime))
    else:
        keyTimeClosest = 0
    
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
    
    '''
    domCats, shownCats = extractCat(catFile)
    lureLst = []
    lureIdx = [] # Find lure indices 
        
    for count, entry in enumerate(domCats):
        if entry == shownCats[count]:
            lureLst.append('true')
        else:
            lureLst.append('lure')
            lureIdx.append(count)
        
        
    allIdx = range(nStimuli) 
    non_lureIdx = [x for x in allIdx if x not in lureIdx]
       
    lure_RT = np.zeros(len(lureIdx))

    for count, idx in enumerate(lureIdx):
        lure_RT[count] = responseTimeLst[idx]      

    nonlure_RT = np.zeros(len(non_lureIdx))
    
    for count, idx in enumerate(non_lureIdx):
        nonlure_RT[count] = responseTimeLst[idx]
        
    return lureIdx, non_lureIdx, lure_RT, nonlure_RT

lureIdx, non_lureIdx, lure_RT, nonlure_RT = findRTs(catFile,responseTimes1)

correctInhib = collections.Counter(lure_RT).most_common(1)
falseInhib = collections.Counter(nonlure_RT).most_common(1) #How many times a keypress was withheld during a non lure stimuli


#%% Check RTs based on either correct or false response (surrounding that lure)
# Extract RTs around lure idx. Extract vals before the lure

def RTaroundLure(lureIdx, lureRT, responseTimeLst,surrounding,number):
    '''
    Computes the response times surrounding lures based on whether it is before or after the lure (surrounding), 
    and how many trials from the lure it is (number).
    
    # Input
    lureIdx: indices of experimental lure stimuli
    lureRT: response times for experimental lure stimuli
    
    '''
    
    lure_RT_add = np.zeros(len(lureIdx))
    
    # catch if lure idx is 0, what if there are two lures in a row
    
    for count, idx in enumerate(lureIdx):
        print(idx)
        if idx == 0:
            pass
        
        else:
            
            if surrounding == 'b' and (idx-lureIdx[count-number]) == 1: # Consecutive lures
                lure_RT_add[count] = 100
                print('the one right before was a lure, passed')
                
            else:
                
            
                if surrounding == 'b': #before    
                    idxn = idx - number
                if surrounding == 'a': #after
                    idxn = idx + number
                    
                lure_RT_add[count] = responseTimeLst[idxn] #This is the RT before/after the lure
                
            # In the b loop it goes down and erases everything below here..
            
            if count != (len(lureIdx))-1:
                if surrounding == 'a' and (lureIdx[count+number]-idx) == 1: # Consecutive lures
                    lure_RT_add[count] = 200
                    print('the one right after was a lure, passed')
                
                else:
                    
                
                    if surrounding == 'b': #before    
                        idxn = idx - number
                    if surrounding == 'a': #after
                        idxn = idx + number
                        
                    lure_RT_add[count] = responseTimeLst[idxn] #This is the RT before/after the lure
        
    
    
    
    
    
    # Find indices of correctly rejected lures
    CRlst = [] # correct reject list. If true, means RT to lure is 0, i.e. inhibited
    
    for val in lure_RT:
        if val == 0:
            CRlst.append(True)
        else:
            CRlst.append(False)
    
    CR_idx = [] # Indices of correctly rejected lures
    for counter,value in enumerate(CRlst):
        if value == True:
            CR_idx.append(counter)
    
    all_lures = range(len(lureIdx))
    
    FR_idx = [x for x in all_lures if x not in CR_idx] # false rejects, i.e. not inhibited
    
    before_CR = np.zeros(len(CR_idx))
    
    for count, idx in enumerate(CR_idx):
        before_CR[count] = lure_RT_add[idx]  
    
    before_CR_mean = np.mean(before_CR)
    
    before_FR = np.zeros(len(FR_idx))
    
    for count, idx in enumerate(FR_idx):
        before_FR[count] = lure_RT_add[idx] 
        
    before_FR_mean = np.mean(before_FR)

### Therefore, subject 02 from experiments week 10 has a RT mean before a correct reject of 0.55 s, and mean before a false reject of 0.38