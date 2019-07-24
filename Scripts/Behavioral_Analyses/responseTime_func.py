
"""
Created on Mon Jan 14 11:38:12 2019

@author: Greta
"""

# IMPORTS 

import numpy as np
import csv
from matplotlib import pyplot as plt
import pandas as pd
from scipy.spatial import distance
import collections

#%% FUNCTIONS

def extractInfo(csvFile):
    '''
    Extracts information from a CSV file.
    
    # Arguments
        csvFile: CSV file
        
    # Returns
        data: list
    
    '''
    data = []
    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for row in csv_reader:           
            data.append(row)
            
    return data
    
def splitString(data):
    '''
    Splits strings in a list by space.
    
    # Arguments
        data: list
        
    # Returns
        splitLst: list
    '''
    splitLst = []
    for entry in data:
        for c in entry:
            split = c.split()
            splitLst.append(split)
    
    return splitLst

def passWarning(splitLst):
    '''
    Extracts information from the first string in a list of lists.
    If the string is not as expected (a number), the function returns a list with the unexpected entries (warningLst).
    
    # Arguments
        splitLst: list of lists (containing strings)
        
    # Returns
        times: list containing the first string as a float
        warningLst: list of unexpected list entries (strings)
    '''
    times = []
    warningLst = []
    
    for entry in splitLst:
        if len(entry) == 4:
            try:
                times.append(float(entry[0]))
            except ValueError:
                print('Unexpected entry: ',entry,'\nEntry appended to warningLst')
                warningLst.append(entry)
        else:
            warningLst.append(entry)
            
    return times, warningLst

def extractStimuliTimes(data):
    '''
    Extracts the second entry in a list of lists and converts the entry to a float.
    
    # Arguments
        data: list of lists (containing strings)
        
    # Returns
        times: list of floats
    '''
    
    times = []
    
    for entry in data:
        if entry[1] == '0':
            pass
        else:
            times.append(float(entry[1]))
        
    return times

def extractCat(indicesFile):
    ''' 
    Extracts experimental categories from a CSV file. 0 denoting scenes, and 1 denoting faces.
    
    # Arguments
        indicesFile: CSV file
            File containing strings of file directories of shown experimental trials (two images for each trial).
    
    # Returns
        dominant_cat: list
        shown_cat: list
    
    '''
    
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
        
    allIdx = range(len(domCats)) 
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

def findRTsBlocks(catFile, responseTimeLst, block = False):
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
    if block == False:
        domCats, shownCats = extractCat(catFile)
    
    # Block-wise
    if block != False:
        domCats, shownCats = extractCat(catFile)
        domCats = domCats[((block-1)*50):(block*50)]
        shownCats = shownCats[((block-1)*50):(block*50)]
        responseTimeLst = responseTimeLst[((block-1)*50):(block*50)]
    
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
        
    allIdx = range(len(domCats)) 
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
    
    # Count how many lures are inhibited correctly
    lure_RT_copy = np.copy(lure_RT)
    lure_RT_count = (~np.isnan(lure_RT_copy)) # Assigns False to nan, i.e. correctly inhibited, and True to falsely inhibited (a respone time recorded)
    
    unique_lure, counts_lure = np.unique(lure_RT_count, return_counts=True)
    
    if len(counts_lure) == 2:
        no_CI_lure = (counts_lure[0]) # correct inhibitions. Inhibitions_lure.
        no_NI_lure = (counts_lure[1]) # not inhibited. no_Inhibitions_lure.
    if len(counts_lure) == 1:
        if unique_lure == False:
            no_CI_lure = int(counts_lure) # correct inhibitions. Inhibitions_lure.
            no_NI_lure = 0 # not inhibited. no_Inhibitions_lure.
        if unique_lure == True:
            no_CI_lure = 0 # correct inhibitions. Inhibitions_lure.
            no_NI_lure = int(counts_lure) # not inhibited. no_Inhibitions_lure.
            
    # Count how many non-lures are inhibited 
    nonlure_RT_copy = np.copy(nonlure_RT)
    nonlure_RT_count = (~np.isnan(nonlure_RT_copy)) # Assigns False to nan, i.e. correctly inhibited, and True to falsely inhibited (a respone time recorded)
    
    unique_nonlure, counts_nonlure = np.unique(nonlure_RT_count, return_counts=True)
    if len(counts_nonlure) == 2:
        no_I_nlure = (counts_nonlure[0]) # Inhibitions, thus a keypress was withheld during a non-lure stimuli. Inhibitions_nonlure.
        no_NI_nlure = (counts_nonlure[1]) # Not inhibited, correct keypress. no_Inhibitions_nonlure.
    if len(counts_nonlure) == 1:
        if unique_nonlure == False:
            print('Subject inhibited all non-lure stimuli')
            no_I_nlure = int(counts_nonlure)
            no_NI_nlure = 0
        if unique_nonlure == True: # Not inhibited, thus if the person responded on all nontrials (correct)
            print('Subject did not inhibit any non-lure stimuli',catFile)
            no_I_nlure = 0 # None inhibited
            no_NI_nlure = int(counts_nonlure) # Should be 45 presses
               
    # Mean of lure RTs, and mean of non-lure RTs (not including inhibited responses)
    lure_RT_mean = np.nanmean(lure_RT)
    nonlure_RT_mean = np.nanmean(nonlure_RT)
    
    # Mean of overall responseTimes
    RT_mean = np.nanmean(responseTimeLst)
    
    # Check for number of NaN, i.e. no response
    nNaN = np.isnan(responseTimeLst).sum()

    return no_CI_lure, no_NI_lure, no_I_nlure, no_NI_nlure, lure_RT_mean, nonlure_RT_mean, RT_mean, nNaN


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
    
    if number != 0:
        
        for count, idx in enumerate(lureIdx):
            if surrounding == 'b': #before    
                idxn = idx - number
                        
            if surrounding == 'a': #after
                idxn = idx + number
            
            # print('lure idx new: ', idxn)
            
            if -1 < idxn < (len(responseTimeLst)-1): # Ensure that the response time is within the limits of the experiment
                
                # print('Lure idx new available in response times lst')
                # Check whether idxn (new idx) is also a lure
                if idxn in lureIdx: 
                    # Check whether the response time was also a lure. In that case, append None
                    # print('wups,coincide!')
                    lure_RT_add[count] = None
                    
                
                if idxn not in lureIdx:
                    # print('hugh, all good, no overlap')
                    lure_RT_add[count] = responseTimeLst[idxn]
                    
            else:
                # print('Lure idx new NOT available in response times lst')
                lure_RT_add[count] = None
    
    if number == 0:
        for count, idx in enumerate(lureIdx):
            lure_RT_add[count] = responseTimeLst[idx]
    
    surrounding_CR = np.zeros(len(CR_idx))
    surrounding_FR = np.zeros(len(FR_idx))
    
    for count, idx in enumerate(CR_idx):
        surrounding_CR[count] = lure_RT_add[idx]  

    surrounding_CR_mean = np.nanmean(surrounding_CR)
        
    for count, idx in enumerate(FR_idx):
        surrounding_FR[count] = lure_RT_add[idx] 
        
    surrounding_FR_mean = np.nanmean(surrounding_FR)
    
    
    return lure_RT_add, surrounding_CR, surrounding_CR_mean, surrounding_FR, surrounding_FR_mean
    
def findFiles(fileLst,expDay):
    '''
    
    
    '''
    
    splitLst = []
    for entry in fileLst:
        split = entry.split('\\')
        splitLst.append(split)
    
    charLst = []
    for entry in splitLst:
        last = entry[-1]
        split = last.split('_')
        charLst.append(split)
    
    imageTimeLst = []
    keypressTimeLst = []
    expTimeLst = []
    keyTimeLst = []
    
    for entry in charLst:
        # Find createIndices file
        if len(entry) == 4:
            if entry[-1] == (expDay + '.csv'):
                catFile = '_'.join(entry)    
        
        # Find imageTime and keypress files
        if len(entry) >= 5:
            
            if entry[4] == expDay:
                if entry[0] == 'imageTime':
                    expTime = entry[-1]
                    expTimeLst.append(expTime)
                    fileNameImage = '_'.join(entry)
                    imageTimeLst.append(fileNameImage)
                    
                if entry[0] == 'keypress':
                    keyTime = entry[-1]
                    keyTimeLst.append(keyTime)
                    fileNameKey = '_'.join(entry)
                    keypressTimeLst.append(fileNameKey)
                
                    
    # Check if several imageTime files exist               
    if len(imageTimeLst) == 1:
        for idx, keyTime in enumerate(keyTimeLst):
            if keyTime == expTimeLst[0]: # Check whether the time stamp for the two files is identical
                keypressFile = keypressTimeLst[idx]
                stimuliFile = imageTimeLst[0]
                
    if len(imageTimeLst) > 1:
        print('WARNING, multiple imageTime files for specified day and subject ID')
                
    
    print('\n ####### Using following files for behavioral analysis ######## \n')
    print('createIndices file: ', catFile)
    print('imageTime file: ', stimuliFile)
    print('keypress file: ', keypressFile)
    
    return catFile, stimuliFile, keypressFile
    
def outputStableLureIdx(subjID):
    catFile = 'P:\\closed_loop_data\\' + str(subjID) + '\\createIndices_'+subjID+'_day_2.csv'

    # Extract categories from category file
    domCats, shownCats = extractCat(catFile)
    
    lureLst = [] 
    lureIdx = [] # Lure indices 
    
    CRlst = []
    CR_idx = [] # Indices of correctly rejected lures

    # Figure out whether a shown stimuli is a lure 
    for count, entry in enumerate(domCats):
        if entry == shownCats[count]:
            lureLst.append(0)
        else:
            lureLst.append(1) # lures = 1
            lureIdx.append(count)
        
    allIdx = range(len(domCats)) 
    non_lureIdx = [x for x in allIdx if x not in lureIdx]
    
    # extract for stable
    
    lureLst_stable_fbrun = np.concatenate([lureLst[400+n*400:600+n*400] for n in range(5)]) # Stable blocks feedback run    
    lureStable = np.concatenate((lureLst[:400], lureLst_stable_fbrun))
    
    return lureStable
    