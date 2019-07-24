# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:01:33 2019

@author: Greta
Script  for checking how NF impact the response on day 2

"""

# Load alpha_test and clf_output_test from pickl file 
import pickle
import matplotlib
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import mne
from scipy.stats import zscore
from pandas import Series
import pandas as pd
from sklearn.cross_decomposition import CCA

scriptsDir = 'C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\'
os.chdir(scriptsDir)
from responseTime_func import extractCat, findRTsBlocks
from variables import *

#%%
d_all2 = {}

for subj in subjID_all:
    with open(EEGDir+'09May_subj_'+subj+'.pkl', "rb") as fin:
         d_all2[subj] = (pickle.load(fin))[0]

#%%
def extractVal2(wkey):
    '''
    Extracts a value from the EEG dict, d_all2.
    '''
    subsAll = []
    subsNF = []
    subsC = []
    
    for key, value in d_all2.items():
        subsNF_result = []
        subsC_result = []
        
        for k, v in value.items():        
            if k == wkey:                
                subsAll.append(v)
                
                if key in subjID_NF:
                    subsNF_result.append(v)
                if key in subjID_C:
                    subsC_result.append(v)
        
        if len(subsNF_result) == 1:
            subsNF.append(subsNF_result[0])
            
        if len(subsC_result) == 1:
            subsC.append(subsC_result[0])
    
    return subsAll, subsNF, subsC

#%% Analyze RT session block-wise

def extractStatsBlockDay2(wkey):
    '''
    
    '''
    wanted_measure_lst = ['sen','spec','fpr','acc','rt','nan']
    w_idx = wanted_measure_lst.index(wkey)
   
    subsAll = []
    subsNF = []
    subsC = []
    
    for key, value in statsBlockDay2_all.items():
        result = value[:,w_idx]
        # print(result.shape)
        subsAll.append(result)
        
        if key in subjID_NF:
            subsNF.append(result)
        if key in subjID_C:
            subsC.append(result) 
    
    return subsAll, subsNF, subsC

def computeStats(subjID):
    '''Computes stats based on days (statsDay) and blocks for each day (both statsBlock: day 1, 3, 4, 5 and statsBlock_day2)
    '''
    # 150-1000 ms analysis
    # with open(saveDir+'//V3_150_1000//BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
    #     sub = (pickle.load(fin))[0]
    
    with open(saveDir+'BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]
    
    statsDay = np.zeros((5,9))
    statsBlock = np.zeros((4,16,6))
    statsBlock_day2 = np.zeros((48,6))
    
    idx_c = 0
    
    for idx, expDay in enumerate(['1','2','3','4','5']):
        
        # Load catFile
        catFile = 'P:\\closed_loop_data\\' + str(subjID) + '\\createIndices_'+subjID+'_day_'+expDay+'.csv'
        
        if subjID == '11' and expDay == '2':
            responseTimes = [np.nan]
            nKeypress = np.nan
        else:
            responseTimes = sub['responseTimes_day'+expDay]
            nKeypress = sub['TOTAL_keypress_day'+expDay]
        
        if subjID == '11' and expDay == '2':
            CI_lure, NI_lure, I_nlure, NI_nlure, lure_RT_mean, nonlure_RT_mean, RT_mean, nNaN = [np.nan]*8
        else:
            CI_lure, NI_lure, I_nlure, NI_nlure, lure_RT_mean, nonlure_RT_mean, RT_mean, nNaN = findRTsBlocks(catFile,responseTimes,block=False)
    
        # Compute stats (for visibility)
        TP = NI_nlure
        FP = NI_lure
        TN = CI_lure
        FN = I_nlure
        
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        FPR = FP/(FP+TN)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
#        accuracy_all = (TP+TN)/nKeypress # Out of all the keypresses that day
        
        statsDay[idx,:] = sensitivity, specificity, FPR, accuracy, lure_RT_mean, nonlure_RT_mean, RT_mean, nKeypress, nNaN

        # Block-wise
        for block in range(1,int(len(responseTimes)/50)+1):
            print(block)
            
            if subjID == '11' and expDay == '2': 
                TN, FP, FN, TP, lure_RT_mean, nonlure_RT_mean, RT_mean, nNaN = [np.nan]*8
            else:
                TN, FP, FN, TP, lure_RT_mean, nonlure_RT_mean, RT_mean, nNaN = findRTsBlocks(catFile,responseTimes,block=block)
            
            sensitivity = TP/(TP+FN)
            specificity = TN/(TN+FP)
            FPR = FP/(FP+TN)
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            
            if expDay == '2':
                statsBlock_day2[block-1,:] = sensitivity, specificity, FPR, accuracy, RT_mean, nNaN
        
            if expDay != '2':
                statsBlock[idx_c,block-1,:] = sensitivity, specificity, FPR, accuracy, RT_mean, nNaN
                
        if expDay != '2':            
            idx_c += 1

    return statsDay, statsBlock, statsBlock_day2

#%% Extract stats for day 2 for all subjs

#%%


#%% Extract stats for all subjects
statsBlockDay2_all = {}

for idx,subjID in enumerate(subjID_all):
    statsDay, statsBlock, statsBlock_day2 = computeStats(subjID)
    
    statsBlockDay2_all[subjID] = statsBlock_day2 


#%%

def blockAlpha():
    '''
    Extracts alpha, accuracy and clf output for NF and C groups individually.
    '''
    subsAll_a, subsNF_a, subsC_a = extractVal2('ALPHA_test')
    subsAll_clf, subsNF_clf, subsC_clf = extractVal2('CLFO_test')
    
    # # Test RT accuracy for reality check
    # for idx, item in enumerate(subsAll_clf):
    #     # above_a = len(np.where((np.array(item)>0.5))[0])/len(item)
    #     above_clfo = len(np.where((np.array(item)>0))[0])/len(item)
    #     print(subjID_all[idx],above_clfo)
    
    # Alpha and clf output and alpha accuracy per block. For all subjects
    # a_per_block = np.zeros((22,n_it*4)) # Subjects as rows, and blocks as columns
    # acc_per_block = np.zeros((22,n_it*4))
    # clfo_per_block = np.zeros((22,n_it*4))
    
    
    # for idx, item in enumerate(subsAll_a):
    #     k = 0
    #     for b in range(n_it*4):
    #         a_per_block[idx,b] = (np.mean(item[k:k+50])) # Mean alpha per block
    #         acc_per_block[idx,b] = len(np.where((np.array(item[k:k+50])>0.5))[0])/50
    #         k += 50
    
    # for idx, item in enumerate(subsAll_clf):
    #     k = 0
    #     for b in range(n_it*4):
    #         clfo_per_block[idx,b] = (np.mean(item[k:k+50])) 
    #         k += 50
    
    a_per_block_NF = np.zeros((11,n_it*4)) # Subjects as rows, and blocks as columns
    a_per_block_C = np.zeros((11,n_it*4))
    
    acc_per_block_NF = np.zeros((11,n_it*4))
    acc_per_block_C = np.zeros((11,n_it*4))
    
    clfo_per_block_NF = np.zeros((11,n_it*4))
    clfo_per_block_C = np.zeros((11,n_it*4))
    
    # NF alpha + acc
    for idx, item in enumerate(subsNF_a):
        k = 0
        for b in range(n_it*4):
            a_per_block_NF[idx,b] = (np.mean(item[k:k+50])) # Mean alpha per block
            acc_per_block_NF[idx,b] = len(np.where((np.array(item[k:k+50])>0.5))[0])/50
            k += 50
            
    # C alpha + acc
    for idx, item in enumerate(subsC_a):
        k = 0
        for b in range(n_it*4):
            a_per_block_C[idx,b] = (np.mean(item[k:k+50])) # Mean alpha per block
            acc_per_block_C[idx,b] = len(np.where((np.array(item[k:k+50])>0.5))[0])/50
            k += 50
    
    # NF clf output
    for idx, item in enumerate(subsNF_clf):
        k = 0
        for b in range(n_it*4):
            clfo_per_block_NF[idx,b] = (np.mean(item[k:k+50])) 
            k += 50
            
     # C clf output
    for idx, item in enumerate(subsC_clf):
        k = 0
        for b in range(n_it*4):
            clfo_per_block_C[idx,b] = (np.mean(item[k:k+50])) 
            k += 50
    
    return a_per_block_NF, a_per_block_C, acc_per_block_NF, acc_per_block_C, clfo_per_block_NF, clfo_per_block_C

def getAvgBeh(wanted_measure):
    '''
    Returns averaged beh stats per block for NF subjects and C subjects
    '''
    # Extract wanted behavioral measure
    subsAll_beh, subsNF_beh, subsC_beh = extractStatsBlockDay2(wanted_measure)
    
    NFBlocks_idx = np.sort(np.concatenate([np.arange(12,8+n_it*8,8),np.arange(13,8+n_it*8,8),np.arange(14,8+n_it*8,8),np.arange(15,8+n_it*8,8)]))
    
    behBlocksNF_all = []
    behBlocksC_all = []
    
    # Chosen beh measure
    for idx, item in enumerate(subsNF_beh):
        behBlocksNF = (np.copy(item))[NFBlocks_idx]
        behBlocksNF_all.append(behBlocksNF)
        
    for idx, item in enumerate(subsC_beh):
        behBlocksC = (np.copy(item))[NFBlocks_idx]
        behBlocksC_all.append(behBlocksC)
    
    # Average over NF behavioral measure
    behBlockNF_avg = np.mean(behBlocksNF_all,axis=0)
    
    # Average over C behavioral measure
    behBlockC_avg = np.mean(behBlocksC_all,axis=0)
    
    return behBlockNF_avg, behBlockC_avg
    

def plotAlphaVSbeh():
    '''
    Plots alpha (or acc or clf output) meaned for NF and C, respectively, vs. chosen behavioral measure.
    
    '''
    # Get averaged behavioral measures for each group
    senBlockNF_avg, senBlockC_avg = getAvgBeh('sen')
    specBlockNF_avg, specBlockC_avg = getAvgBeh('spec')
    accBlockNF_avg, accBlockC_avg = getAvgBeh('acc')
    rtBlockNF_avg, rtBlockC_avg = getAvgBeh('rt')

    # Extract alpha, accuracy and clf output per block
    a_per_block_NF, a_per_block_C, acc_per_block_NF, acc_per_block_C, clfo_per_block_NF, clfo_per_block_C = blockAlpha()
    
    # Average over alpha for NF
    alphaBlockNF_avg = np.mean(a_per_block_NF,axis=0)
    
    # Average over alpha for C
    alphaBlockC_avg = np.mean(a_per_block_C,axis=0)
    
    # Average over clf output for NF
    clfoBlockNF_avg = np.mean(clfo_per_block_NF,axis=0)
    
    # Average over clf output for C
    clfoBlockC_avg = np.mean(clfo_per_block_C,axis=0)
    
    zscore(alphaBlockNF_avg)
    
    
    # Plot NF
    plt.figure(random.randint(0,100))
    plt.xticks(np.arange(1,n_it*4+1),[str(item) for item in np.arange(1,n_it*4+1)])

    # plt.step(np.arange(1,n_it*4+1),alphaBlockNF_avg/np.sum(alphaBlockNF_avg),where='post',label='alpha NF',linewidth=4.0)
    plt.step(np.arange(1,n_it*4+1),zscore(clfoBlockNF_avg),where='post',label='clf output NF',linewidth=4.0)
    # plt.step(np.arange(1,n_it*4+1),senBlockNF_avg/np.sum(senBlockNF_avg),where='post',label='sensitivity NF')
    # plt.step(np.arange(1,n_it*4+1),specBlockNF_avg/np.sum(specBlockNF_avg),where='post',label='specificity NF')
    plt.step(np.arange(1,n_it*4+1),zscore(accBlockNF_avg),where='post',label='accuracy NF')
    # plt.step(np.arange(1,n_it*4+1),zscore(rtBlockNF_avg),where='post',label='response time NF')
    plt.legend()
    
    # Plot C
    plt.figure(random.randint(0,100))
    plt.xticks(np.arange(1,n_it*4+1),[str(item) for item in np.arange(1,n_it*4+1)])

    # plt.step(np.arange(1,n_it*4+1),alphaBlockC_avg/np.sum(alphaBlockC_avg),where='post',label='alpha C',linewidth=4.0)
    plt.step(np.arange(1,n_it*4+1),zscore(clfoBlockC_avg),where='post',label='clf output C',linewidth=4.0)
    # plt.step(np.arange(1,n_it*4+1),senBlockC_avg/np.sum(senBlockC_avg),where='post',label='sensitivity C')
    # plt.step(np.arange(1,n_it*4+1),specBlockC_avg/np.sum(specBlockC_avg),where='post',label='specificity C')
    plt.step(np.arange(1,n_it*4+1),zscore(accBlockC_avg),where='post',label='accuracy C')
    # plt.step(np.arange(1,n_it*4+1),zscore(rtBlockC_avg),where='post',label='response time C')
    plt.legend()

    
def plotMatchedAlphavsBeh(pair,wanted_measure,zscored=False):
    '''
    Does not average across participants, but plots the participant with its matched control participant.
    
    Plots a chosen participant pair, for a wanted behavioral measure
    '''
    d_match_pairs = [[k,v] for k, v in d_match.items()]
    subject_pair = d_match_pairs[pair]
    
    NFBlocks_idx = np.sort(np.concatenate([np.arange(12,8+n_it*8,8),np.arange(13,8+n_it*8,8),np.arange(14,8+n_it*8,8),np.arange(15,8+n_it*8,8)]))

    subsAll_beh, subsNF_beh, subsC_beh = extractStatsBlockDay2(wanted_measure)
    
    behBlocksNF_all = [] # Only NF blocks extracted
    for idx, item in enumerate(subsAll_beh):
        behBlocksNF = (np.copy(item))[NFBlocks_idx]
        behBlocksNF_all.append(behBlocksNF)
    
    # Extract alpha, accuracy and clf output per block
    a_per_block_all, acc_per_block_all, clfo_per_block_all = blockMatchedAlpha() 
    
    # For alpha corr control
    subsAll_a, subsNF_a, subsC_a = extractVal2('ALPHA_test')
    # Check that matchedAlpha is correlated with clf output
    # subsAll_clf, subsNF_clf, subsC_clf = extractVal2('CLFO_test')
    
    matchedAlpha = []
    matchedAlphaBlock = []
    matchedClfBlock = []
    matchedAccBlock = []
    matchedBeh = []
    
    for NF_person, C_person in d_match.items():
        # print(NF_person)
        for subjKey, val in d_all2.items():
            # print(key)
            if C_person == subjKey:
                print(NF_person,subjKey)                
                # print(val_match) # Take this subject, which is the key! extract alpha file from values
                
                # Get behavioral measure for the NF participant and control
                # behBlocksNF has all subjects' chosen behavioral measure. 
                
                print('This is the subjKey, i.e. matched participant ',subjKey)
                control_idx = subjID_all.index(subjKey)
                NF_idx = subjID_all.index(NF_person)
                print('This is the NF idx, i.e. NF participant ',NF_idx)
                print('subj_idx of the matched person ',control_idx)
                
                # Get alpha list or beh measure 
                matched = [behBlocksNF_all[NF_idx],behBlocksNF_all[control_idx]]
                matchedBeh.append(matched)
                
                # Extract alpha 
                matched_a = [subsAll_a[NF_idx],subsAll_a[control_idx]]
                matchedAlpha.append(matched_a)
                
                matched_a_block = [a_per_block_all[NF_idx],a_per_block_all[control_idx]]
                matchedAlphaBlock.append(matched_a_block)
                
                # Add clf output
                matched_clf_block = [clfo_per_block_all[NF_idx],clfo_per_block_all[control_idx]]
                matchedClfBlock.append(matched_clf_block)
                
                # Add decoding accuracy
                matched_acc_block = [acc_per_block_all[NF_idx],acc_per_block_all[control_idx]]
                matchedAccBlock.append(matched_acc_block)
    
    if zscored == True:
        # Manually add the last value as the same as the last
        
        # Plot NF subject vs matched control, with the alpha shown (the NF person's alpha)
        fig,ax=plt.subplots()
        
        for axis in [ax.xaxis]:
            axis.set(ticks=np.arange(1.5,n_it*4+1), ticklabels=[str(item) for item in np.arange(1,n_it*4+1)])
        
        #plt.xticks(np.arange(1,n_it*4+2),[str(item) for item in np.arange(1,n_it*4+1)])
        # Plotting alpha of the NF subject, e.g. subj 7, which was also used for subj 17
        plt.step(np.arange(1,n_it*4+2),np.append(zscore(matchedAlphaBlock[pair][0]),(zscore(matchedAlphaBlock[pair][0])[-1:])),where='post',label='Alpha value (shown for both subjects)',linewidth=2.0, color='black')
        # plt.step(np.arange(1,n_it*4+1),zscore(clfoBlockNF_avg),where='post',label='clf output NF',linewidth=4.0)
        plt.step(np.arange(1,n_it*4+2),np.append(zscore(matchedBeh[pair][0]),zscore(matchedBeh[pair][0])[-1:]),where='post',label='Behavioral accuracy, NF subject',linewidth=1.0,color='tomato')
        plt.step(np.arange(1,n_it*4+2),np.append(zscore(matchedBeh[pair][1]),zscore(matchedBeh[pair][1])[-1:]),where='post',label='Behavioral accuracy, matched control subject',linewidth=1.0,color='dodgerblue')
        plt.xlabel('NF block number, day 2')
        plt.ylabel('Z-scored units')
        plt.legend()
        plt.title('Behavioral accuracy per block for NF subject and matched control: '+str(subject_pair))
    
    if zscored == False:
        fig,ax=plt.subplots()
        
        for axis in [ax.xaxis]:
            axis.set(ticks=np.arange(1.5,n_it*4+1), ticklabels=[str(item) for item in np.arange(1,n_it*4+1)])
        
        plt.step(np.arange(1,n_it*4+2),np.append(matchedAlphaBlock[pair][0],matchedAlphaBlock[pair][0][-1:]),where='post',label='Alpha value (shown for both subjects)',linewidth=2.0, color='black')
        plt.step(np.arange(1,n_it*4+2),np.append(matchedBeh[pair][0],matchedBeh[pair][0][-1:]),where='post',label='Behavioral accuracy, NF subject',linewidth=1.0,color='tomato')
        plt.step(np.arange(1,n_it*4+2),np.append(matchedBeh[pair][1],matchedBeh[pair][1][-1:]),where='post',label='Behavioral accuracy, matched control subject',linewidth=1.0,color='dodgerblue')
        plt.xlabel('NF block number, day 2')
        plt.ylabel('Non z-scored units')
        plt.legend()
        plt.title('Behavioral accuracy per block for NF subject and matched control: '+str(subject_pair))

    # Reality check
    # han = matchedAccBlock[7][0] # Must be the NF person of d_match index 7, i.e. subj 22
    # np.mean(han) # Matches with the RT accuracy for subj 22 in d_all2
    
    return matchedAlpha, matchedAlphaBlock, matchedBeh
    
def blockMatchedAlpha():
    '''
    Returns alpha, accuracy and clf output er block for all subjects.
    
    '''
    subsAll_a, subsNF_a, subsC_a = extractVal2('ALPHA_test')
    subsAll_clf, subsNF_clf, subsC_clf = extractVal2('CLFO_test')
    
    a_per_block_all = np.zeros((22,n_it*4)) # Subjects as rows, and blocks as columns
    acc_per_block_all = np.zeros((22,n_it*4))
    clfo_per_block_all = np.zeros((22,n_it*4))
    
    # alpha and acc all
    for idx, item in enumerate(subsAll_a):
        k = 0
        for b in range(n_it*4):
            a_per_block_all[idx,b] = (np.mean(item[k:k+50])) # Mean alpha per block
            # a_per_block_all = a_per_block_all[]
            
            acc_per_block_all[idx,b] = len(np.where((np.array(item[k:k+50])>0.5))[0])/50
            k += 50    
            
    # clf output all
    for idx, item in enumerate(subsAll_clf):
        k = 0
        for b in range(n_it*4):
            clfo_per_block_all[idx,b] = (np.mean(item[k:k+50])) # Mean alpha per block
            k += 50  
            
    return a_per_block_all, acc_per_block_all, clfo_per_block_all
            
      
#%% Create d_match

d_match = {}

for i in range(len(NF_group)):
    d_match[NF_group[i]] = C_group[i]
    
#%% Analyze block-wise matched pairs
    
# Plot alpha vs behavioral correlation value
# plt.figure(101)
# plt.scatter(matchedAlphaBlock[7][0],matchedBeh[7][1])

# Simple correlations for all pairs
matchedAlpha, matchedAlphaBlock, matchedBeh_sen = plotMatchedAlphavsBeh(pair=7,wanted_measure='sen',zscored=1)
matchedAlpha, matchedAlphaBlock, matchedBeh_spec = plotMatchedAlphavsBeh(pair=10,wanted_measure='spec',zscored=1)
matchedAlpha, matchedAlphaBlock, matchedBeh_acc = plotMatchedAlphavsBeh(pair=10,wanted_measure='acc',zscored=False)
matchedAlpha, matchedAlphaBlock, matchedBeh_rt = plotMatchedAlphavsBeh(pair=7,wanted_measure='rt',zscored=1)

corrs_sen = np.zeros((11,2))
corrs_spec = np.zeros((11,2))
corrs_acc = np.zeros((11,2))
corrs_rt = np.zeros((11,2))

stats.ttest_rel(corrs_sen[:,0],corrs_sen[:,1],nan_policy='omit')
stats.ttest_rel(corrs_acc[:,0],corrs_acc[:,1],nan_policy='omit')
stats.ttest_rel(corrs_spec[:,0],corrs_spec[:,1],nan_policy='omit')
stats.ttest_rel(corrs_rt[:,0],corrs_rt[:,1],nan_policy='omit')



for idx in range(0,11):
    if idx == 2:
        corrs_sen[idx] = np.nan, np.nan
        corrs_spec[idx] = np.nan, np.nan
        corrs_acc[idx] = np.nan, np.nan
        corrs_rt[idx] = np.nan, np.nan
    else:
        corr_NF_sen = np.corrcoef(matchedAlpha_test[idx][0],matchedBeh_sen[idx][0])
        corr_C_sen = np.corrcoef(matchedAlpha_test[idx][0],matchedBeh_sen[idx][1])
        
        corr_NF_spec = np.corrcoef(matchedAlpha_test[idx][0],matchedBeh_spec[idx][0])
        corr_C_spec = np.corrcoef(matchedAlpha_test[idx][0],matchedBeh_spec[idx][1])
        
        corr_NF_acc = np.corrcoef(matchedAlpha_test[idx][0],matchedBeh_acc[idx][0])
        corr_C_acc = np.corrcoef(matchedAlpha_test[idx][0],matchedBeh_acc[idx][1])
        
        corr_NF_rt = np.corrcoef(matchedAlpha_test[idx][0],matchedBeh_rt[idx][0])
        corr_C_rt = np.corrcoef(matchedAlpha_test[idx][0],matchedBeh_rt[idx][1])
        
        corrs_sen[idx] = corr_NF_sen[0][1],corr_C_sen[0][1]
        corrs_spec[idx] = corr_NF_spec[0][1],corr_C_spec[0][1]
        corrs_acc[idx] = corr_NF_acc[0][1],corr_C_acc[0][1]
        corrs_rt[idx] = corr_NF_rt[0][1],corr_C_rt[0][1]

print(np.round(corrs_sen,decimals=3))
print(np.round(corrs_spec,decimals=3))

print(np.round(corrs_acc,decimals=3))
print(np.round(corrs_rt,decimals=3))

#%% ################# CCA ######################

# 1. I want to extract alpha, acc and clf output for each subject, for each NF session
# 2. I need behavioral measures, i.e. TP, FN, FP, TN for each time point

# Create a list of shown alphas per matched pair (i.e. the alphas for NF subject, with 0.5 initialization block)
# matchedAlpha_test contains the NF alpha value as first list, and control alpha as the second list

shownAlphaPair = [sublist[0] for sublist in matchedAlpha]

for sublist in shownAlphaPair:
    for number in np.arange(0,1000,50):
        sublist[number] = 0.5
        sublist[number+1] = 0.5
        sublist[number+2] = 0.5



#%% Function for extraction behavioral measure for each time point


def extractPointResponse(subjID,stable=False):
    '''Extract the response for every single time point
    '''
    
    with open(saveDir+'BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]
        
    # Load catFile
    catFile = 'P:\\closed_loop_data\\' + str(subjID) + '\\createIndices_'+subjID+'_day_2.csv'

    domCats, shownCats = extractCat(catFile)
    
    if subjID == '11':
        responseTimes = [np.nan]
    else:
        responseTimes = sub['responseTimes_day2']

    # Compute stats (for visibility)
    # TP = NI_nlure
    # FP = NI_lure
    # TN = CI_lure
    # FN = I_nlure
    
    pointResponse = [] 
    lureIdx = [] # Lure indices 
    
    # Figure out whether a shown stimuli is a lure 
    for count, entry in enumerate(domCats):
        if entry == shownCats[count]: # For non-lures
            if np.isnan(responseTimes[count]) == True: # If a non lure trial was inhibited
                pointResponse.append('FN') # Inhibited, non lure = FN
            if np.isnan(responseTimes[count]) == False: 
                pointResponse.append('TP')
        if entry != shownCats[count]:
            if np.isnan(responseTimes[count]) == True: # If nan value appears in responseTimeLst, it must have been correctly rejected
                pointResponse.append('TN')
            if np.isnan(responseTimes[count]) == False: # A response during a lure, i.e. FR
                pointResponse.append('FP')
            lureIdx.append(count)
    
    pointResponseNFblocks = np.copy(pointResponse)

    if stable == False:
        e_mock = np.arange((8+n_it*8)*50)
        nf_blocks_idx = np.concatenate([e_mock[600+n*400:800+n*400] for n in range(n_it)]) # Neurofeedback blocks 
        pointResponseNFblocks = pointResponseNFblocks[nf_blocks_idx]
    
    return pointResponseNFblocks

def computeShownAlpha(subjID):
    ''' Computes the shown alphas (using the moving window averaging)
    '''
    
    with open(EEGDir + '09May_subj_' + subjID + '.pkl', "rb") as fin:
        subEEG = (pickle.load(fin))[0]
                
    alphaOutput = subEEG['ALPHA_test']
    
    # Create a mean alpha lst (which subject "viewed", i.e. with a starting proportion of 0.5)
    alphaWINDOWlst = []

    for i in np.arange(0,1000,50):
        edges_none, a = makeRollingWindows(alphaOutput[i:i+50],3,meanalpha=True)
        alphaWINDOWlst.append(a)
    
    alphaMEANlst = []
    
    for sublst in alphaWINDOWlst:
        intermedlst = []
        for entry in sublst:
            m = np.mean(entry)
            intermedlst.append(m)
            
        # Every list needs to have removed the last value in each block, and add a 0.5 in the start instead 
        # Add 3 times 0.5 in the beginning
        del intermedlst[-1:]
        intermedlst2 = [0.5,0.5,0.5] + intermedlst
        alphaMEANlst.append(intermedlst2)
        
    alphameanSHOWN = [item for sublist in alphaMEANlst for item in sublist]
    
    return alphameanSHOWN
    
    

def computeStatsTimepoint(subjID,stable=False):
    '''
    Always input the NF subject. The function finds the control
    Computes TP, FN, FP, TN for each time point, day 2.
    '''
    # Create a mean alpha lst (which subject "viewed", i.e. with a starting proportion of 0.5)
    # In function computeShownAlpha
    
    alphameanSHOWN = computeShownAlpha(subjID)
    
    # Mean this further?
    alphaSHOWN_pd = pd.DataFrame(data=alphameanSHOWN)
    alphaSHOWN_rolling = alphaSHOWN_pd.rolling(window=10, min_periods=1)
    alphaSHOWNRollingMean = alphaSHOWN_rolling.mean()
    
    # Extract the pointResponses (first for NF subject)
    if stable == False:
        pointResponseNF = extractPointResponse(subjID)
    if stable == True:
        pointResponseNF = extractPointResponse(subjID,stable=True)
    
    edgeNF, pointWindowFullNF = makeRollingWindows(pointResponseNF,10)
    
    windowAccNF = []
    for edgeWindow in edgeNF:
        winAcc = computeWindowStats(edgeWindow)
        windowAccNF.append(winAcc)
        
    for windowLst in pointWindowFullNF:
        winAcc = computeWindowStats(windowLst)
        windowAccNF.append(winAcc)
    
    # Find the control subject
    for NFsubj, Csubj in d_match.items():
        if NFsubj == subjID:
            subjID_control = Csubj
      
     # Extract the pointResponses (for control subject)
    if stable == False:
        pointResponseC = extractPointResponse(subjID_control)
    if stable == True:
        pointResponseC = extractPointResponse(subjID_control,stable=True)
    
    edgeC, pointWindowFullC = makeRollingWindows(pointResponseC,10)
    
    windowAccC = []
    for edgeWindow in edgeC:
        winAcc = computeWindowStats(edgeWindow)
        windowAccC.append(winAcc)
        
    for windowLst in pointWindowFullC:
        winAcc = computeWindowStats(windowLst)
        windowAccC.append(winAcc)
    
    if stable == False:
        # Shown windowed 3 mean alpha with additional mean across 10
        alphaSHOWNRollingMean_a = alphaSHOWNRollingMean.as_matrix()
        alphaSHOWNRollingMean_l = [sublist[0] for sublist in alphaSHOWNRollingMean_a]
    if stable == True: 
        alphaSHOWNRollingMean_a = (alphaSHOWNRollingMean.as_matrix()).flatten()
        
        e_mock = np.arange((8+n_it*8)*50)
        nf_blocks_idx = np.concatenate([e_mock[600+n*400:800+n*400] for n in range(n_it)]) # Neurofeedback blocks 
        
        alphaSHOWNRollingMean_l = np.full(len(e_mock), np.nan) #np.zeros((len(e_mock)))
        alphaSHOWNRollingMean_l[nf_blocks_idx] = alphaSHOWNRollingMean_a
    
    # Plot the beh accuracy of NF, and of C and alpha
    plt.figure(random.randint(50,140)) 
    plt.plot(windowAccNF,color='red',label='NF')
    plt.plot(windowAccC,color='blue',label='C')
    plt.plot(alphaSHOWNRollingMean_l,color='black', label='alpha, additional avg (window 10)')
    # plt.plot(alphameanSHOWN,color='green',label='Shown alpha, avg window 3')
    plt.legend()
    
    windowAccNF_a = np.asarray(windowAccNF)
    windowAccC_a = np.asarray(windowAccC)

    # Check for simple correlations
    np.corrcoef(alphaSHOWNRollingMean_l,windowAccNF_a)
    np.corrcoef(alphaSHOWNRollingMean_l,windowAccC_a)
        
    #  After calculating the cross-correlation between the two signals, the maximum (or minimum if the signals are negatively correlated) 
    # of the cross-correlation function indicates the point in time where the signals are best aligned; i.e.,
    # the time delay between the two signals is determined by argmax of the cross-correlation
    # lag between signals is given by the argmax of the cross-correlation
    # lag = np.argmax(correlate(alphaRollingMean_l,windowAccNF_a))
    # c_sig = np.roll(windowAccNF_a, shift=int(np.ceil(lag)))
    
    # Corr with means substracted
    # corr = np.correlate(alphameanSHOWN - np.mean(alphameanSHOWN), 
    #                 windowAccNF_a - np.mean(windowAccNF_a),
    #                 mode='full')
    
    # Corr normalized. First autocorrelation for testing. Gives a corr coef of 1
    # corr2 = np.correlate(alphameanSHOWN - np.mean(alphameanSHOWN), 
    #                 alphameanSHOWN - np.mean(alphameanSHOWN),
    #                 mode='full')/(len(alphameanSHOWN) * np.std(alphameanSHOWN) * np.std(alphameanSHOWN))
    # NF corr
    corrNF = np.correlate(alphaSHOWNRollingMean_l - np.mean(alphaSHOWNRollingMean_l), 
                    windowAccNF_a - np.mean(windowAccNF_a),
                    mode='full')/(len(alphaSHOWNRollingMean_l) * np.std(alphaSHOWNRollingMean_l) * np.std(windowAccNF_a))
    
    corrNF_nonstand = np.correlate(alphameanSHOWN - np.mean(alphameanSHOWN), 
                    windowAccNF_a - np.mean(windowAccNF_a),
                    mode='full')
    
    # C corr
    corrC = np.correlate(alphaSHOWNRollingMean_l - np.mean(alphaSHOWNRollingMean_l), 
                    windowAccC_a - np.mean(windowAccC_a),
                    mode='full')/(len(alphaSHOWNRollingMean_l) * np.std(alphaSHOWNRollingMean_l) * np.std(windowAccC_a))
    
    # plt.figure(random.randint(53,100))
    # plt.plot(corrNF)
    
    # uncorrelated shifts will be 0?
    
    lagNF = corrNF.argmax() - (len(alphaSHOWNRollingMean_l) - 1)
    lagNF_ns = corrNF_nonstand.argmax() - (len(alphaSHOWNRollingMean_l) - 1)

    lagC = corrC.argmax() - (len(alphaSHOWNRollingMean_l) - 1)

    # Find the correlation coefficient:
    corr_coefNF = corrNF[(corrNF.argmax())]
    corr_coefC = corrC[(corrC.argmax())]

    return lagNF, lagC, corr_coefNF, corr_coefC, alphaSHOWNRollingMean_l, windowAccNF_a, windowAccC_a
    
    
                
def makeRollingWindows(a, window, meanalpha=False):
    '''
    If meanalpha = True, only computes the windows (does not do anything with the edges)
    
    '''
    
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    if meanalpha == True:
        stride_array2 = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        edge_windows=[]
    
    if meanalpha == False:
        # If I want to add identical edge windows in the start
        # startAppend = window - 1
        stride_array = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        # stride_0 = stride_array[0]
        # stride_array2 = np.vstack(([stride_0]*startAppend,stride_array))
        
        lacking_windows = len(a) - shape[0]
    
        edge_windows = []
        for n_lack in range(lacking_windows):
            print(n_lack)
            w = a[0:n_lack+1]
            edge_windows.append(w)
            
        stride_array2 = stride_array  

    return edge_windows,stride_array2
     

def computeWindowStats(windowLst):
    # sensitivity = TP/(TP+FN)
    # specificity = TN/(TN+FP)
    # FPR = FP/(FP+TN)
    
    unique, counts = np.unique(windowLst, return_counts=True)
    unique_d = dict(zip(unique, counts))
    
    for key, val in unique_d.items():
        if key == 'TP':
            TP = val
        if key == 'TN':
            TN = val
        if key == 'FN':
            FN = val
        if key == 'FP':
            FP = val
     
    try:
        TP
    except NameError:
        TP = 0
        
    try:
        TN
    except NameError:
        TN = 0
        
    try:
        FN
    except NameError:
        FN = 0
        
    try:
        FP
    except NameError:
        FP = 0
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    return accuracy
    
#%% Compute statsTimepoint for pairs
lagPairs = np.zeros((11,2))
coefPairs = np.zeros((11,2))


for idx,subjID in enumerate(subjID_NF):
    if subjID == '11':
        pass
    else:
        lagNF, lagC, corr_coefNF, corr_coefC, nr, nr2, nr3 = computeStatsTimepoint(subjID)
        lagPairs[idx] = lagNF, lagC
        coefPairs[idx] = corr_coefNF, corr_coefC
        
#%% Extract the shown alpha and beh acc for each pair, and make averaged plot
alphas_all = []
NFacc_all = []
Cacc_all = []

for idx,subjID in enumerate(subjID_NF):
    if subjID == '11':
        pass
    else:
        nr, nr2, nr3, nr4, alphashown, NFacc, Cacc = computeStatsTimepoint(subjID,stable=True)
        alphas_all.append(alphashown)
        NFacc_all.append(NFacc)
        Cacc_all.append(Cacc)

plt.figure(100)
plt.plot(np.mean(alphas_all,axis=0),color='black')
plt.plot(np.mean(NFacc_all,axis=0),color='red')        
plt.plot(np.mean(Cacc_all,axis=0),color='blue')     

# The correlations do not make a lot of sense, bc we just check which time series is averaged to be most "close"...

# np.corrcoef(np.mean(alphas_all,axis=0)[nf_blocks_idx],np.mean(NFacc_all,axis=0)[nf_blocks_idx])   
# np.corrcoef(np.mean(alphas_all,axis=0)[nf_blocks_idx],np.mean(Cacc_all,axis=0)[nf_blocks_idx])

# corrC = np.correlate(np.mean(alphas_all,axis=0)[nf_blocks_idx] - np.mean(np.mean(alphas_all,axis=0)[nf_blocks_idx]), 
#         np.mean(Cacc_all,axis=0)[nf_blocks_idx] - np.mean(np.mean(Cacc_all,axis=0)[nf_blocks_idx]),
#         mode='full')/(1000 * np.std(np.mean(alphas_all,axis=0)[nf_blocks_idx]) * np.std(np.mean(Cacc_all,axis=0)[nf_blocks_idx]))
    

# lagNF = corrNF.argmax() - (len(alphaSHOWNRollingMean_l) - 1)

# lagC = corrC.argmax() - (len(alphaSHOWNRollingMean_l) - 1)

# # Find the correlation coefficient:
# corr_coefNF = corrC[(corrC.argmax())]

    

#%% Check correlation between NF matched alpha subject and that particular subject
        
# Using matchedAlpha from plotMatchedAlphavsBeh function.

alphaCorrs = []
alpha_all = [] # List with all subjects' alpha lists. The decoded ones. Not the ones shown! 
alpha_nf = []
for idx, entry in enumerate(matchedAlpha):
    alphaCorr = np.corrcoef(matchedAlpha[idx][0],matchedAlpha[idx][1]) # NF vs control
    d_match['Match_'+str(NF_group[idx])+'_'+str(C_group[idx])] = alphaCorr[0][1]
    alphaCorrs.append(alphaCorr[0][1])
    alpha_all.append(matchedAlpha[idx][0])
    alpha_all.append(matchedAlpha[idx][1])
    alpha_nf.append(matchedAlpha[idx][0])

np.mean(alphaCorrs)

# Make alpha lists with the shown alphas (e.g. 3 alphas=0.5 for each block)

alpha_all_shown = np.copy(alpha_all)

for number in np.arange(0,1000,50):
    alpha_all_shown[:,number] = 0.5
    alpha_all_shown[:,number+1] = 0.5
    alpha_all_shown[:,number+2] = 0.5
    
# Correlation between alpha and classifier output in general
# alpha_sub13=matchedAlpha[3][0]

# a=d_all2['13']
# clfo_sub13=a['CLFO_test']

# np.corrcoef(alpha_sub13,clfo_sub13)

# plt.hist(alpha_sub13)

# ALPHA PLOT OBS NOT OVER A MOVING 3 AVERAGE. This is also for all subjects, i.e. it is not the ones shown. 
# it is the decoded alphas across all subjects. non averaged.
# Average across all subjects' alpha values:
alpha_all_shownlst = alpha_all_shown.tolist()

plt.figure(3)
plt.hist(alpha_all_shownlst,color=['black']*22,bins=22)
plt.xticks(np.arange(0.1,1.1,0.1),['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])
plt.xlabel('Feedback (mixture proportion of attended category)')
plt.ylabel('Count (number of trials)')
plt.axvline(0.17,color='black',linewidth=0.5)
plt.axvline(0.27,color='black',linewidth=0.5)
plt.axvline(0.5,color='black',linewidth=0.5)
plt.axvline(0.84,color='black',linewidth=0.5)
plt.axvline(0.98,color='black',linewidth=0.5)
plt.title('Feedback (alpha) values for all participants')

# FOR NF
plt.figure(4)
plt.hist(alpha_nf,color=['black']*11,bins=11)
plt.xticks(np.arange(0.1,1.1,0.1),['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])
plt.xlabel('Feedback (mixture proportion of attended category)')
plt.ylabel('Count (number of trials)')
plt.axvline(0.17,color='black',linewidth=0.5)
plt.axvline(0.27,color='black',linewidth=0.5)
plt.axvline(0.5,color='black',linewidth=0.5)
plt.axvline(0.84,color='black',linewidth=0.5)
plt.axvline(0.98,color='black',linewidth=0.5)
plt.title('Feedback (alpha) values for NF (i.e. shown)')

# Moving window 3 averaging added. Only NF alphas shown!!!
alpha_all_shown_window = []

for subjID in subjID_NF:
    s = computeShownAlpha(subjID)
    alpha_all_shown_window.append(s)

plt.figure(5)
plt.hist(alpha_all_shown_window,color=['black']*11,bins=11)
plt.xticks(np.arange(0.1,1.1,0.1),['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])
plt.xlabel('Feedback (mixture proportion of attended category)')
plt.ylabel('Count (number of trials)')
plt.axvline(0.17,color='black',linewidth=0.5)
plt.axvline(0.27,color='black',linewidth=0.5)
plt.axvline(0.5,color='black',linewidth=0.5)
plt.axvline(0.84,color='black',linewidth=0.5)
plt.axvline(0.98,color='black',linewidth=0.5)
plt.title('Feedback (alpha) values for NF (i.e. shown, with moving avg)')

#%% Alpha correlation, shown ones

alphaCorrs_shown = []
alpha_all = [] # List with all subjects' alpha lists. The decoded ones. Not the ones shown! 
for idx, entry in enumerate(matchedAlpha):
    alphaCorr = np.corrcoef(matchedAlpha[idx][0],matchedAlpha[idx][1]) # NF vs control
    d_match['Match_'+str(NF_group[idx])+'_'+str(C_group[idx])] = alphaCorr[0][1]
    alphaCorrs.append(alphaCorr[0][1])
    alpha_all.append(matchedAlpha[idx][0])
    alpha_all.append(matchedAlpha[idx][1])

np.mean(alphaCorrs)

np.corrcoef(alpha_all_shownlst[0],alpha_all_shownlst[1])
np.corrcoef(alpha_all_shownlst[2],alpha_all_shownlst[3])
np.corrcoef(alpha_all_shownlst[4],alpha_all_shownlst[5])



#%% Classifier output pre (and post?) FR and CR


def preFRandCR(subjID):
    '''
    Rewrite this into: 
    Extracts when lures were shown in the experiment, and matches response times to lures and non-lures.
    
    
    # Input
    catFile: category file for extraction of shown categories.
    responseTimeLst: list of response times for the shown, experimental stimuli
    
    # Output
    lureLst 
    
    '''
    
    # Define stuff
    block_len = 50
    
    on_FR = [] # Clf output during FR
    on_CR = []
    
    pre1_FR = [] # Clf output 1 trial before FR
    pre1_CR = []
    
    pre2_FR = [] # Clf output 2 trials before FR
    pre2_CR = []
    
    pre3_FR = [] # Clf output 3 trials before FR
    pre3_CR = []
    
    post1_FR = []
    post1_CR = []
    
    post2_FR = []
    post2_CR = []
    
    post3_FR = []
    post3_CR = []
    
    #
    with open(saveDir + 'BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]
    
    catFile = 'P:\\closed_loop_data\\' + str(subjID) + '\\createIndices_'+subjID+'_day_2.csv'
    
    # Extract categories from category file
    domCats, shownCats = extractCat(catFile)
    
    # Get responseTimes
    responseTimes = sub['responseTimes_day2']
    
    lureLst2 = [] 
    lureIdx = [] # Lure indices 
    
    # Figure out whether a shown stimuli is a lure 
    for count, entry in enumerate(domCats):
        if entry == shownCats[count]:
            lureLst2.append('true')
        else:
            if np.isnan(responseTimes[count]) == True: # If nan value appears in responseTimeLst, it must have been correctly rejected
                lureLst2.append('CR')
            if np.isnan(responseTimes[count]) == False: # A response during a lure, i.e. FR
                lureLst2.append('FR')
            lureIdx.append(count)
            
    with open(EEGDir + '09May_subj_' + subjID + '.pkl', "rb") as fin:
        subEEG = (pickle.load(fin))[0]
        
    # 08May is from the HPC cluster
        
    clf_output = subEEG['CLFO_test']
        
    # with open(EEGDir + '18April_subj_' + subjID + '.pkl', "rb") as fin:
    #     subEEGold = (pickle.load(fin))[0]
    
    lureLst_c = np.copy(lureLst2)
    e_mock = np.arange((8+n_it*8)*block_len)
    nf_blocks_idx = np.concatenate([e_mock[600+n*400:800+n*400] for n in range(n_it)]) # Neurofeedback blocks 
    lureLstNF = lureLst_c[nf_blocks_idx]
        
    for count,trial in enumerate(lureLstNF):
        if trial == 'FR':
            on_FR.append(clf_output[count])
            try:
                pre1_FR.append(clf_output[count-1])
            except:
                pre1_FR.append(np.nan)
            try:
                pre2_FR.append(clf_output[count-2])
            except:
                pre2_FR.append(np.nan)
            try:
                pre3_FR.append(clf_output[count-3])
            except:
                pre3_FR.append(np.nan)
                
            try:
                post1_FR.append(clf_output[count+1])
            except:
                post1_FR.append(np.nan)
            try:
                post2_FR.append(clf_output[count+2])
            except:
                post2_FR.append(np.nan)
            try:
                post3_FR.append(clf_output[count+3])
            except:
                post3_FR.append(np.nan)
            
        if trial == 'CR':
            on_CR.append(clf_output[count])
            try:
                pre1_CR.append(clf_output[count-1])
            except:
                pre1_CR.append(np.nan)
            try:
                pre2_CR.append(clf_output[count-2])
            except:
                pre2_CR.append(np.nan)
            try:
                pre3_CR.append(clf_output[count-3])
            except:
                pre3_CR.append(np.nan)
            
            try:
                post1_CR.append(clf_output[count+1])
            except:
                post1_CR.append(np.nan)
            try:
                post2_CR.append(clf_output[count+2])
            except:
                post2_CR.append(np.nan)
            try:
                post3_CR.append(clf_output[count+3])
            except:
                post3_CR.append(np.nan)
    
    return np.nanmean(pre1_FR), np.nanmean(pre1_CR)

#%%           
preFR_NF = []
preCR_NF = []

preFR_C = []
preCR_C = []

     
for subjID in subjID_NF:
    print(subjID)
    if subjID == '11':
        continue
    else:
        preFR, preCR = preFRandCR(subjID)
        preFR_NF.append(preFR)
        preCR_NF.append(preCR)
    
for subjID in subjID_C:
    print(subjID)
    preFR, preCR = preFRandCR(subjID)
    preFR_C.append(preFR)
    preCR_C.append(preCR)
    
# Running it for: averaging over the three precesing trials! 
preFR_NF_a = np.mean(np.reshape(np.asarray(preFR_NF),[3,10]),axis=0)
preCR_NF_a = np.mean(np.reshape(np.asarray(preCR_NF),[3,10]),axis=0)

preFR_C_a = np.mean(np.reshape(np.asarray(preFR_C),[3,11]),axis=0)
preCR_C_a = np.mean(np.reshape(np.asarray(preCR_C),[3,11]),axis=0)

#%% Rearrange C so it can be matched
idxLst = [2,0,1,4,9,3,6,10,7,5,8]
newSortC = [y for x,y in sorted(zip(idxLst,subjID_C))] # I.e. just sort behavioral measure based on this list. 

preFR_C_sort = [y for x,y in sorted(zip(idxLst,preFR_C_a))]
preCR_C_sort = [y for x,y in sorted(zip(idxLst,preCR_C_a))]

# Delete subj 15
preFR_C_sort = np.delete(preFR_C_sort,2)
preCR_C_sort = np.delete(preCR_C_sort,2)

# Find diffs
NF_diff = preCR_NF_a - preFR_NF_a
C_diff = preCR_C_sort - preFR_C_sort
C_diff_unsorted = preCR_C_a - preFR_C_a


print(stats.ttest_rel(NF_diff,C_diff))
print(stats.ttest_rel(NF_diff,C_diff_unsorted))

#%%
# Errorbars
sem_FR_NF = np.std(preFR_NF)/np.sqrt(10)
sem_CR_NF = np.std(preCR_NF)/np.sqrt(10)

sem_FR_C = np.std(preFR_C)/np.sqrt(11)
sem_CR_C = np.std(preCR_C)/np.sqrt(11)

# 3 trials meaned
sem_FR_NF = np.std(preFR_NF_a)/np.sqrt(10)
sem_CR_NF = np.std(preCR_NF_a)/np.sqrt(10)

sem_FR_C = np.std(preFR_C_a)/np.sqrt(11)
sem_CR_C = np.std(preCR_C_a)/np.sqrt(11)



#%% Plot
plt.figure(random.randint(0,100))
plt.ylabel('Mean classifier output 3 trials before lure')
plt.xticks([1,2,3,4],['NF, FA','NF, CR','Control, FA','Control, CR'])# 'Control day 1, part 2', 'Control day 3, part 2'])
plt.title('Behavioral performance linked to neurofeedback')

plt.scatter(np.full(10,1),preFR_NF_a,color='tomato')
plt.scatter(np.full(10,2),preCR_NF_a,color='tomato')
plt.scatter(np.full(11,3),preFR_C_a,color='dodgerblue')
plt.scatter(np.full(11,4),preCR_C_a,color='dodgerblue')
plt.grid(color='gainsboro',linewidth=0.5)

for i in range(10):
    plt.plot([(np.full(10,1))[i],(np.full(10,2))[i]], [(preFR_NF_a)[i],(preCR_NF_a)[i]],color='tomato')

for i in range(11):
    plt.plot([(np.full(11,3))[i],(np.full(11,4))[i]], [(preFR_C_a)[i],(preCR_C_a)[i]],color='dodgerblue')
    
plt.plot([(np.full(1,1)),(np.full(1,2))], [(np.mean(preFR_NF_a)),np.mean(preCR_NF_a)],color='black')
plt.plot([(np.full(1,3)),(np.full(1,4))], [(np.mean(preFR_C_a)),np.mean(preCR_C_a)],color='black')

(_, caps, _) = plt.errorbar(np.full(1,1),np.mean(preFR_NF_a),yerr=sem_FR_NF, capsize=8, color='black',elinewidth=2,barsabove=True)
for cap in caps:
    cap.set_markeredgewidth(2)
(_, caps, _) = plt.errorbar(np.full(1,2),np.mean(preCR_NF_a),yerr=sem_CR_NF, capsize=8, color='black',elinewidth=2,barsabove=True)
for cap in caps:
    cap.set_markeredgewidth(2)

(_, caps, _) = plt.errorbar(np.full(1,3),np.mean(preFR_C_a),yerr=sem_FR_C, capsize=8, color='black',elinewidth=2,barsabove=True)
for cap in caps:
    cap.set_markeredgewidth(2)
(_, caps, _) = plt.errorbar(np.full(1,4),np.mean(preCR_C_a),yerr=sem_FR_C, capsize=8, color='black',elinewidth=2,barsabove=True)
for cap in caps:
    cap.set_markeredgewidth(2)
    
    
    





#%%



print(stats.ttest_ind(preFR_NF,preCR_NF,nan_policy='omit'))
print(stats.ttest_ind(preFR_C,preCR_C,nan_policy='omit'))

# 3 averaged
print(stats.ttest_rel(preFR_NF_a,preCR_NF_a,nan_policy='omit'))
print(stats.ttest_ind(preFR_C_a,preCR_C_a,nan_policy='omit'))

#%% Plot with subject annotation
subjID_NF_a = np.array(subjID_NF)
subjID_NF_a = np.delete(subjID_NF_a,2)

subjID_C_a = np.array(subjID_C)

fig,ax = plt.subplots()
plt.ylabel('Mean classifier 3 trials pre lure')
plt.xticks([1,2,3,4],['NF, false alarms','NF, correct rejections','Control, false alarms','Control, correct rejections'])# 'Control day 1, part 2', 'Control day 3, part 2'])
plt.title(title)

ax.scatter(np.full(10,1),preFR_NF,color='lightsalmon')
ax.scatter(np.full(10,2),preCR_NF,color='lightsalmon')
ax.scatter(np.full(11,3),preFR_C,color='powderblue')
ax.scatter(np.full(11,4),preCR_C,color='powderblue')

# Annotation
for i, txt in enumerate(subjID_NF_a):
    ax.annotate(txt, ((np.full(10,1))[i], (preFR_NF)[i]))
    
for i, txt in enumerate(subjID_NF_a):
    ax.annotate(txt, ((np.full(10,2))[i], (preCR_NF)[i]))
    
for i, txt in enumerate(subjID_C_a):
    ax.annotate(txt, ((np.full(11,3))[i], (preFR_C)[i]))
    
for i, txt in enumerate(subjID_C_a):
    ax.annotate(txt, ((np.full(11,4))[i], (preCR_C)[i]))


for i in range(10):
    plt.plot([(np.full(10,1))[i],(np.full(10,2))[i]], [(preFR_NF)[i],(preCR_NF)[i]],color='lightsalmon')

for i in range(11):
    plt.plot([(np.full(11,3))[i],(np.full(11,4))[i]], [(preFR_C)[i],(preCR_C)[i]],color='powderblue')
    
plt.plot([(np.full(1,1)),(np.full(1,2))], [(np.mean(preFR_NF)),np.mean(preCR_NF)],color='black')
plt.plot([(np.full(1,3)),(np.full(1,4))], [(np.mean(preFR_C)),np.mean(preCR_C)],color='black')

# (_, caps, _) = plt.errorbar(np.full(1,1),np.mean(preFR_NF),yerr=sem_FR_NF, capsize=8, color='black',elinewidth=2,barsabove=True)
# for cap in caps:
#     cap.set_markeredgewidth(2)
# (_, caps, _) = plt.errorbar(np.full(1,2),np.mean(preCR_NF),yerr=sem_CR_NF, capsize=8, color='black',elinewidth=2,barsabove=True)
# for cap in caps:
#     cap.set_markeredgewidth(2)

# (_, caps, _) = plt.errorbar(np.full(1,3),np.mean(preFR_C),yerr=sem_FR_C, capsize=8, color='black',elinewidth=2,barsabove=True)
# for cap in caps:
#     cap.set_markeredgewidth(2)
# (_, caps, _) = plt.errorbar(np.full(1,4),np.mean(preCR_C),yerr=sem_FR_C, capsize=8, color='black',elinewidth=2,barsabove=True)
# for cap in caps:
#     cap.set_markeredgewidth(2)
    

print(stats.ttest_ind(preFR_NF,preCR_NF,nan_policy='omit'))
print(stats.ttest_ind(preFR_C,preCR_C,nan_policy='omit'))    
    
    
    


#%% For generating plots, subjID 26

with open(saveDir+'BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
    sub = (pickle.load(fin))[0]

with open(EEGDir + '09May_subj_' + subjID + '.pkl', "rb") as fin:
    subEEG = (pickle.load(fin))[0]
# Load catFile
# catFile = 'P:\\closed_loop_data\\' + str(subjID) + '\\createIndices_'+subjID+'_day_2.csv'
# domCats, shownCats = extractCat(catFile)

clfOutput = subEEG['CLFO_test']              
alphaOutput = subEEG['ALPHA_test']        

plt.figure(random.randint(0,100))
plt.plot(clfOutput[500:550])
plt.plot(alphaOutput[500:550])

# Create mean alphas for the entire length of alpha
alphaWINDOWlst = []

for i in np.arange(0,1000,50):
    edges, a = makeRollingWindows(alphaOutput[i:i+50],3,meanalpha=True)
    alphaWINDOWlst.append(a)

alphaMEANlst = []

for sublst in alphaWINDOWlst:
    intermedlst = []
    for entry in sublst:
        m = np.mean(entry)
        intermedlst.append(m)
        
    # Every list needs to have removed the last value in each block, and add a 0.5 in the start instead 
    # Add 3 times 0.5 in the beginning
    del intermedlst[-1:]
    intermedlst2 = [0.5,0.5,0.5] + intermedlst
    alphaMEANlst.append(intermedlst2)
    
# 11th block must be 
chosen_meanalpha = alphaMEANlst[10]

# By using pandas
colnames = ['idx', 'mean_alpha']
data = pd.read_csv('P:\\closed_loop_data\\' + str(subjID) + '\\MEANalpha_subjID_26__day_204-09-19_08-48.csv', names=colnames)
mean_alpha = data.mean_alpha.tolist() 
del mean_alpha[0]
# Len of mean_alpha is 940, because for each block, the first 3 are not appended as mean alpha (0.5 mixture)

# Validate mean in running windows usind pandas
# chosen_block_pd = pd.DataFrame(data=chosen_block)
# chosen_block_rolling = chosen_block_pd.rolling(window=3, min_periods=3)
# chosen_block_Mean = chosen_block_rolling.mean()

# # Every 47th is a new block in mean_alpha
# mean_alpha_chosen = [0.5,0.5,0.5]+mean_alpha[470:470+47] # same as chosen_block_Mean
chosen_clfoutput = clfOutput[500:550]
chosen_clfoutput[0] = 0
chosen_clfoutput[1] = 0
chosen_clfoutput[2] = 0

chosen_alpha = alphaOutput[500:550] # 11th block


# Clf output plot
plt.figure(random.randint(0,100))
plt.plot(chosen_clfoutput,color='black')
plt.ylabel('Attended category decoding')
plt.xlabel('Trial number')
plt.xticks(np.arange(0,60,10),[str(item) for item in np.arange(0,60,10)])
plt.yticks(np.arange(-1,2,1),[str(item) for item in np.arange(-1,2,1)])

# Mean alpha plot (or it could be the real alpha...)
plt.figure(random.randint(0,100))
plt.plot(chosen_meanalpha,color='black')
plt.ylabel('Mixture proportion of attended category') # meaned over a running window of 3 trials)
plt.xlabel('Trial number')
plt.xticks(np.arange(0,60,10),[str(item) for item in np.arange(0,60,10)])
plt.yticks(np.arange(0,1.5,0.5),[str(item) for item in np.arange(0,1.5,0.5)])


# Clf output and alpha
plt.figure(random.randint(0,100))
plt.plot(chosen_clfoutput)
plt.plot(chosen_alpha)
plt.plot(chosen_meanalpha,color='red')


#%% Extract mean alphaVals for NF subjects, in order to plot their actual seen values

subsAll_alphas, subsNF_alphas, subsC_alphas = extractVal2('ALPHA_test')

subsNF_meanAlphas = []
for alphalst in subsNF_alphas:
    alphamean = np.mean(alphalst)
    subsNF_meanAlphas.append(alphamean)

np.save(scriptsDir+'subsNF_meanAlphas.npy',subsNF_meanAlphas)



