# -*- coding: utf-8 -*-
"""
Created on Fri May 24 19:23:28 2019

@author: Greta
"""
import pickle
from matplotlib import pyplot as plt
import os
import statistics
import numpy as np
from scipy import stats
import seaborn as sns
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import zscore
import matplotlib

from responseTime_func import findRTsBlocks
from variables import *

#%%
statsDay_all = {} # NOT block-wise, statsDay. Checked 30April whether this corresponds to the other (V2) beh measures. OK.
statsBlock_all = {}
statsBlockDay2_all = {}

# Extract stats for all subjects
for idx,subjID in enumerate(subjID_all):
    statsDay, statsBlock, statsBlock_day2 = computeStats(subjID)
    
    statsDay_all[subjID] = statsDay
    statsBlock_all[subjID] = statsBlock
    statsBlockDay2_all[subjID] = statsBlock_day2

#%%
def extractSurroundingRTs(subjID):
        
    with open(saveDir + 'BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]
    
    CR1 = sub['surrounding_CR_Lst_day_1']
    CR3 = sub['surrounding_CR_Lst_day_3']
    
    FR1 = sub['surrounding_FR_Lst_day_1']
    FR3 = sub['surrounding_FR_Lst_day_3']
    
    # Computing day 1 and 3 together
    CR13m = [statistics.mean(k) for k in zip(CR1, CR3)]
    FR13m = [statistics.mean(k) for k in zip(FR1, FR3)]
    
    return np.asarray(CR1), np.asarray(CR3), np.asarray(FR1), np.asarray(FR3), np.asarray(CR13m), np.asarray(FR13m)


def computeStats(subjID):
    '''Computes stats based on days (statsDay) and blocks for each day (both statsBlock: day 1, 3, 4, 5 and statsBlock_day2)
    '''
    
    with open(saveDir + 'BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
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

#%% Functions for extracting stats values from statsDay_all, statsBlock_all and statsBlockDay2_all
    
def extractStatsDay(day_idx,wanted_measure):
    #sensitivity, specificity, FPR, accuracy, lure_RT_mean, nonlure_RT_mean, RT_mean, nKeypress, nNaN
    day_idx = day_idx-1
    
    wanted_measure_lst = ['sen','spec','fpr','acc','rt_lure','rt_nlure','rt','nkeypress','nan']
    w_idx = wanted_measure_lst.index(wanted_measure)
    
    subsAll = []
    subsNF = []
    subsC = []
    
    for key, value in statsDay_all.items():
#        print(value.shape)
        result = value[day_idx,w_idx]
#        print(result.shape)
        subsAll.append(result)
        
        if key in subjID_NF:
            subsNF.append(result)
        if key in subjID_C:
            subsC.append(result) 
    
    return subsAll, subsNF, subsC

def extractDividedStats(wanted_measure):
    #sensitivity, specificity, FPR, accuracy, lure_RT_mean, nonlure_RT_mean, RT_mean, nKeypress, nNaN    
    wanted_measure_lst = ['sen','spec','fpr','acc','rt','nan']
    w_idx = wanted_measure_lst.index(wanted_measure)
    
    subsAll = []
    subsAll_NFBlocks = []
    subsAll_stableBlocks = []

    subsNF = []
    subsNF_NFBlocks = []
    subsNF_stableBlocks = []
    
    subsC = []
    subsC_NFBlocks = []
    subsC_stableBlocks = []
    
    allBlocks_idx = np.arange(0,48)
    NFBlocks_idx = np.sort(np.concatenate([np.arange(12,8+n_it*8,8),np.arange(13,8+n_it*8,8),np.arange(14,8+n_it*8,8),np.arange(15,8+n_it*8,8)]))
    stableBlocks_idx = [x for x in allBlocks_idx if x not in NFBlocks_idx]
    
    for key, value in statsBlockDay2_all.items():
        result = value[:,w_idx]
        
        resultNF = result[NFBlocks_idx]
        resultStable = result[stableBlocks_idx]
        
        subsAll.append(np.mean(result))
        subsAll_NFBlocks.append(np.mean(resultNF))
        subsAll_stableBlocks.append(np.mean(resultStable))
        
        if key in subjID_NF:
            subsNF.append(np.mean(result))
            subsNF_NFBlocks.append(np.mean(resultNF))
            subsNF_stableBlocks.append(np.mean(resultStable))
            
        if key in subjID_C:
            subsC.append(np.mean(result)) 
            subsC_NFBlocks.append(np.mean(resultNF))
            subsC_stableBlocks.append(np.mean(resultStable))
    
    return subsAll, subsAll_NFBlocks, subsAll_stableBlocks, subsNF, subsNF_NFBlocks, subsNF_stableBlocks, subsC, subsC_NFBlocks, subsC_stableBlocks 

        
def extractStatsBlock(day,wanted_measure):
    '''
    
    '''
    if day == 1:
        day_idx = 0
    if day == 3:
        day_idx = 1
    if day == 4:
        day_idx = 2
    if day == 5:
        day_idx = 3
    
    wanted_measure_lst = ['sen','spec','fpr','acc','rt','nan']
    w_idx = wanted_measure_lst.index(wanted_measure)
   
    subsAll = []
    subsNF = []
    subsC = []
    
    for key, value in statsBlock_all.items():
        result = value[day_idx,:,w_idx]
        # print(result.shape)
        subsAll.append(result)
        
        if key in subjID_NF:
            subsNF.append(result)
        if key in subjID_C:
            subsC.append(result) 
    
    return subsAll, subsNF, subsC

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

#%%
def make4Bars(wanted_measure,title,ylabel,part2=False,RER=False):
    '''
    Comparing day 1 and 3, NF and controls, using the statsDay_all structure and extract function.
    
    Connected lines bar plot.
    '''
    if part2 == True:
        all_d1, NF_d1, C_d1 = extractStatsDay(4,wanted_measure)
        all_d3, NF_d3, C_d3 = extractStatsDay(5,wanted_measure)
        
        print('NF mean day 1, part 2: ',round(np.mean(NF_d1),3))
        print('NF mean day 3, part 2: ',round(np.mean(NF_d3),3))
        print('C mean day 1, part 2: ',round(np.mean(C_d1),3))
        print('C mean day 3, part2: ',round(np.mean(C_d3),3))
        
    if part2 == False:
        
        if RER == False:        
            all_d1, NF_d1, C_d1 = extractStatsDay(1,wanted_measure)
            all_d3, NF_d3, C_d3 = extractStatsDay(3,wanted_measure)
            
            print('NF mean day 1: ',round(np.mean(NF_d1),3))
            print('NF mean day 3: ',round(np.mean(NF_d3),3))
            print('C mean day 1: ',round(np.mean(C_d1),3))
            print('C mean day 3: ',round(np.mean(C_d3),3))
            
        if RER == True:
            all_d1, NF_d1, C_d1 = extractStatsDay(1,wanted_measure)
            all_d3, NF_d3, C_d3 = extractStatsDay(3,wanted_measure)
            
            NF_d1 = NF_d1/(1-np.asarray(NF_d1))
            NF_d3 = NF_d3/(1-np.asarray(NF_d3))
            
            C_d1 = C_d1/(1-np.asarray(C_d1))
            C_d3 = C_d3/(1-np.asarray(C_d3))
            
            print('NF mean day 1: ',round(np.mean(NF_d1),3))
            print('NF mean day 3: ',round(np.mean(NF_d3),3))
            print('C mean day 1: ',round(np.mean(C_d1),3))
            print('C mean day 3: ',round(np.mean(C_d3),3))
    
    y_min = np.min([np.min(all_d1),np.min(all_d3)])
    y_max = np.max([np.max(all_d1),np.max(all_d3)])
    
    # Connect lines
    plt.figure(random.randint(0,100))
    plt.ylabel(ylabel)
    
    if wanted_measure == 'nkeypress' or wanted_measure == 'nan':
        plt.ylim([y_min-5,y_max+5])
        
    else:
        plt.ylim([y_min-0.01,y_max+0.01])
    
    if part2 == True:
        plt.xticks([1,2,3,4],['NF day 1, part 2','NF day 3, part 2', 'Control day 1, part 2', 'Control day 3, part 2'])
        
    if part2 == False:
        plt.xticks([1,2,3,4],['NF day 1','NF day 3', 'Control day 1', 'Control day 3'])
        
    plt.title(title)
    
    plt.bar(1,np.mean(NF_d1),color=(0,0,0,0),edgecolor='tomato')
    plt.bar(2,np.mean(NF_d3),color=(0,0,0,0),edgecolor='brown')
    plt.bar(3,np.mean(C_d1),color=(0,0,0,0),edgecolor='dodgerblue')
    plt.bar(4,np.mean(C_d3),color=(0,0,0,0),edgecolor='navy')
    
    plt.scatter(np.full(11,1),NF_d1,color='tomato')
    plt.scatter(np.full(11,2),NF_d3,color='brown')
    plt.scatter(np.full(11,3),C_d1,color='dodgerblue')
    plt.scatter(np.full(11,4),C_d3,color='navy')
    plt.grid(color='gainsboro',linewidth=0.5)
    
    for i in range(11):
        plt.plot([(np.full(11,1))[i],(np.full(11,2))[i]], [(NF_d1)[i],(NF_d3)[i]],color='gray')
        plt.plot([(np.full(11,3))[i],(np.full(11,4))[i]], [(C_d1)[i],(C_d3)[i]],color='gray')
        
    t_fb = stats.ttest_rel(NF_d1,NF_d3)
    t_c = stats.ttest_rel(C_d1,C_d3)
    t_baseline = stats.ttest_ind(NF_d1,C_d1)
    
    if part2 == True:
        print('P-value between day 1 and 3, part 2, NF: ', round(t_fb[1],3))
        print('P-value between day 1 and 3, part 2, control: ', round(t_c[1],3))
        print('P-value between NF and control group day 1 (baseline), part 2: ', round(t_baseline[1],3))
    
    if part2 == False:
        print('P-value between day 1 and 3, NF: ', round(t_fb[1],3))
        print('P-value between day 1 and 3, control: ', round(t_c[1],3))
        print('P-value between NF and control group day 1 (baseline): ', round(t_baseline[1],3))
    
    return plt, t_fb, t_c, all_d1


def make2Bars(wanted_measure,title,ylabel,RER=False):
    '''
    Comparing day 2, NF and controls, using the statsDay_all structure and extract function.    
    '''
    all_d2, NF_d2, C_d2 = extractStatsDay(2,wanted_measure)
    
    if RER == True:
        all_d2 = all_d2/(1-np.asarray(all_d2))
        NF_d2 = NF_d2/(1-np.asarray(NF_d2))
        C_d2 = C_d2/(1-np.asarray(C_d2))
    
    print('NF mean day 2: ',round(np.nanmean(NF_d2),3))
    print('C mean day 2: ',round(np.mean(C_d2),3))
    
    y_min = np.nanmin(all_d2)
    y_max = np.nanmax(all_d2)
    
    # Connect lines
    plt.figure(random.randint(0,100))
    plt.ylabel(ylabel)
    if wanted_measure != 'nkeypress':
        plt.ylim([y_min-0.01,y_max+0.01])
    else:
        plt.ylim([y_min-10,y_max+10])
        
    plt.xticks([1,2],['NF day 2','Control day 2'])
    plt.title(title)
    
    plt.bar(1,np.nanmean(NF_d2),color=(0,0,0,0),edgecolor='tomato')
    plt.bar(2,np.mean(C_d2),color=(0,0,0,0),edgecolor='dodgerblue')
    
    plt.scatter(np.full(11,1),NF_d2,color='tomato')
    plt.scatter(np.full(11,2),C_d2,color='dodgerblue')
    
    # Omit one subject from C?
    t = stats.ttest_ind(NF_d2,C_d2,nan_policy='omit')
        
    return plt, t, all_d2


#%% ########## BLOCK-WISE ANALYSIS #############

def behBlock(day,wanted_measure,title,ylabel):
    all_d, NF_d, C_d = extractStatsBlock(day,wanted_measure)
    
    # Find participants below or above mean
    mean1 = np.mean(all_d,axis=0)
    meanstd = np.std(all_d,axis=0)
    for idx,subj in enumerate(all_d):
        for blockno,val in enumerate(subj):
            if val < mean1[blockno] - (3*meanstd[blockno]):
                print('subjID: ', subjID_all[idx], 'Block no. ',blockno+1)
        
    
    subsNF_RT_acc = np.load(scriptsDir+'subsNF_RT_acc.npy').flatten()
    subsC_RT_acc = np.load(scriptsDir+'subsC_RT_acc.npy').flatten()
    
    normNF = matplotlib.colors.Normalize(vmin=np.min(subsNF_RT_acc), vmax=np.max(subsNF_RT_acc))
    normC = matplotlib.colors.Normalize(vmin=np.min(subsC_RT_acc), vmax=np.max(subsC_RT_acc))
    cmReds=cm.get_cmap("Reds")
    cmBlues=cm.get_cmap("Blues")
    
    # Plot NF subjects
    plt.figure(random.randint(0,100))
    for j in range(len(NF_d)):
        plt.plot(NF_d[j],c=cmReds(normNF(subsNF_RT_acc[j])),linewidth=1)
        
    plt.plot(np.mean(NF_d,axis=0),label='Mean NF group',color='tomato',linewidth=2.5)
    
    # Plot C subjects
    for i in range(len(C_d)):
        plt.plot(C_d[i],c=cmBlues(normC(subsC_RT_acc[i])),linewidth=1)
        
    plt.plot(np.mean(C_d,axis=0),label='Mean control group',color='dodgerblue',linewidth=2.5)
    
    # plt.plot(np.mean(all_d,axis=0),label='Mean all participants',color='black',linewidth=2.0)
    plt.title(title)
    plt.xticks(np.arange(0,16),[str(item) for item in np.arange(1,17)])
    plt.xlabel('Block number')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(color='gainsboro',linewidth=0.5)
    
    t = stats.ttest_ind(NF_d,C_d)
    
    for pval in t[1]:
        if pval < 0.05:
            print('significant')
    print(t[1])


def outputResponseTimes(subjID,expDay):
    '''Outputs responseTimes for a chosen day
    '''
    with open(saveDir+'BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]
    
    responseTimes = sub['responseTimes_day'+expDay]
    
    return responseTimes

def plotResponseTimes(expDay):
    responseTimes_all = []

    # Extract responseTimes for all subjects
    for idx,subjID in enumerate(subjID_all):
        if subjID == '11' and expDay == '2':
            pass
        else:
            responseTimes = outputResponseTimes(subjID,expDay)
            responseTimes_all.append(responseTimes)
    
    plt.figure(random.randint(0,100))
    colorAll = sns.color_palette("Set2",22)
    plt.title('Response times, EEG session')
    
    if expDay != '2':
        plt.xticks(np.arange(0,850,50),[str(item) for item in np.arange(0,850,50)])
    else:
        plt.xticks(np.arange(0,2450,100),[str(item) for item in np.arange(0,2450,100)],rotation=90)
        
    plt.xlabel('Trial number')
    plt.ylabel('Response time (ms)')
    
    for j in range(len(responseTimes_all)):
        plt.plot(responseTimes_all[j],color=colorAll[j],linewidth=0.3)
        
    plt.plot(np.nanmean(responseTimes_all,axis=0),label='Mean response time (all participants)',color='black',linewidth=1.5)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('rt_alld2.eps', bbox_inches = "tight")
    
def matchedSubjects(wanted_measure,title,RER=False):

    # Extract values for day 1 and 3
    all_d1, NF_d1, C_d1 = extractStatsDay(1,wanted_measure)
    all_d3, NF_d3, C_d3 = extractStatsDay(3,wanted_measure)
    
    if RER == False:
        diff = np.asarray(all_d3) - np.asarray(all_d1)
    if RER == True:
        all_d1 = all_d1/(1-np.asarray(all_d1))
        NF_d1 = NF_d1/(1-np.asarray(NF_d1))
        C_d1 = C_d1/(1-np.asarray(C_d1))
        
        all_d3 = all_d3/(1-np.asarray(all_d3))
        NF_d3 = NF_d3/(1-np.asarray(NF_d3))
        C_d3 = C_d3/(1-np.asarray(C_d3))
        
        diff = (np.asarray(all_d3) - np.asarray(all_d1))

    NF_group_match = []
    C_group_match = []

    for idx,subjID in enumerate(subjID_all):
        if subjID in NF_group:
            NF_group_match.append([subjID,diff[idx]])
        else:
            C_group_match.append([subjID,diff[idx]])
    
    C_group_match_re = []
    for subjID in C_group:
        for subORDER in C_group_match:
            if subjID == subORDER[0]:
                C_group_match_re.append([subjID,subORDER[1]])
    
    NF = [item[1] for item in NF_group_match]
    C = [item[1] for item in C_group_match_re]
    
    # Remove pair 26-34
    # del NF[8] 
    # del C[8]
    
    print(stats.ttest_rel(NF,C))

    plt.figure(random.randint(0,100))
    plt.bar(1,np.mean(NF),color=(0,0,0,0),edgecolor='tomato',label='Mean change from day 1 to 3, NF group')
    plt.bar(2,np.mean(C),color=(0,0,0,0),edgecolor='dodgerblue',label='Mean change from day 1 to 3, control group')
    
    plt.scatter(np.full(11,1),NF,color='tomato')
    plt.scatter(np.full(11,2),C,color='dodgerblue')
    
    for i in range(11):
        plt.plot([(np.full(11,1))[i],(np.full(11,2))[i]], [(NF)[i],(C)[i]],color='gray')
        
    plt.xticks([1,2],['NF group','Control group'])
    plt.ylabel(r'$\Delta$ response time (day 1 to 3)')
    plt.title(title)
    plt.grid(color='gainsboro',linewidth=0.5)
    
    
def NFimprovement(wanted_measure):

    all_d1, NF_d1, C_d1 = extractStatsDay(1,wanted_measure)
    all_d2, NF_d2, C_d2 = extractStatsDay(2,wanted_measure)
    
    del NF_d1[2]
    del NF_d2[2]
    
    print('NF mean day 1 and 2: ',round(np.mean(NF_d1),3),round(np.mean(NF_d2),3))
    print('Control mean day 1 and 2: ',round(np.mean(C_d1),3),round(np.mean(C_d2),3))
    
    t_NF = stats.ttest_rel(NF_d1,NF_d2)
    t_C = stats.ttest_rel(C_d1,C_d2)
    
    if t_NF[1] < 0.001:
        print('T-test between NF group day 1 and 2\n T-statistic = '+str(round(t_NF[0],4))
              +'\n P-value = '+'{:0.3e}'.format(t_NF[1]))
        print('T-test between control group day 1 and 2\n T-statistic = '+str(round(t_C[0],3))
              +'\n P-value = '+'{:0.3e}'.format(t_C[1]))
    else:
        print('T-test between NF group day 1 and 2\n T-statistic = '+str(round(t_NF[0],4))
              +'\n P-value = '+str(round(t_NF[1],3)))
        print('T-test between control group day 1 and 2\n T-statistic = '+str(round(t_C[0],3))
              +'\n P-value = '+str(round(t_C[1],3)))
        
def matchedSubjects2(wanted_measure,title,relative=False,NFblocks=False):
    '''
    Matched pairs, looks at the difference between day 1 and day 2 (overall behavioral measure for that day)
    If NFblocks == True, the day2 stats is only based on NF (or stable?) blocks
    '''
    
    # Extract values for day 1 and 2
    all_d1, NF_d1, C_d1 = extractStatsDay(1,wanted_measure)
    all_d2, NF_d2, C_d2 = extractStatsDay(2,wanted_measure)

    # Output diff without subj 11 only, for non matched participant analysis
    all_d1_21 = np.copy(all_d1)
    all_d2_21 = np.copy(all_d2)
    
    all_d1_21 = np.delete(all_d1_21,2)
    all_d2_21 = np.delete(all_d2_21,2)
    
    diff_21 = all_d2_21 - all_d1_21
    
    # Delete subj 11
    del all_d1[2]
    # Delete subj 15
    del all_d1[4]
    
    if NFblocks == False:        
        # Delete subj 11
        del all_d2[2]
        # Delete subj 15
        del all_d2[4]
    
    if NFblocks == True:
        all_d2_block, NF_d2_block, C_d2_block = extractStatsBlockDay2(wanted_measure)
        # Only use NF blocks
        NFBlocks_idx = np.sort(np.concatenate([np.arange(12,8+n_it*8,8),np.arange(13,8+n_it*8,8),np.arange(14,8+n_it*8,8),np.arange(15,8+n_it*8,8)]))
    
        behBlocksNF_all = [] # Only NF blocks extracted
        for idx, item in enumerate(all_d2_block):
            behBlocksNF = (np.copy(item))[NFBlocks_idx]
            behBlocksNF_all.append(behBlocksNF)
        
        all_d2 = []
        for subj in behBlocksNF_all:
            subj_avg = np.mean(subj)
            all_d2.append(subj_avg)
            
        # Delete subj 11
        del all_d2[2]
        # Delete subj 15
        del all_d2[4]

    if relative == True:
        diff = (np.asarray(all_d2) - np.asarray(all_d1))/(1-np.asarray(all_d1))
    if relative == False:
        diff = np.asarray(all_d1) - np.asarray(all_d2)

    NF_group_match = []
    C_group_match = []
    
    # Subj 11 and 15 removed
    subjID_20 = ['07','08','13','14','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

    for idx,subjID in enumerate(subjID_20):
        if subjID in NF_group:
            NF_group_match.append([subjID,diff[idx]])
        else:
            C_group_match.append([subjID,diff[idx]])
    
    C_group_match_re = []
    for subjID in C_group:
        for subORDER in C_group_match:
            if subjID == subORDER[0]:
                C_group_match_re.append([subjID,subORDER[1]])
    
    NF = [item[1] for item in NF_group_match]
    C = [item[1] for item in C_group_match_re]
    
    print(stats.ttest_rel(NF,C))

    plt.figure(random.randint(150,250))
    plt.bar(1,np.mean(NF),color=(0,0,0,0),edgecolor='tomato',label='Mean change from day 1 to 2, NF group')
    plt.bar(2,np.mean(C),color=(0,0,0,0),edgecolor='dodgerblue',label='Mean change from day 1 to 2, control group')
    
    plt.scatter(np.full(10,1),NF,color='tomato')
    plt.scatter(np.full(10,2),C,color='dodgerblue')
    
    for i in range(10):
        plt.plot([(np.full(10,1))[i],(np.full(10,2))[i]], [(NF)[i],(C)[i]],color='gray')
        
    plt.xticks([1,2],['NF group','Control group'])
    plt.ylabel(r'$\Delta$ response time (day 1 to 2)')
    plt.grid(color='gainsboro',linewidth=0.5)
    plt.title(title)
    
    return diff, diff_21

def computeDividedCorr(wanted_measure):
    # Load RT acc
    subsNF_RT_acc = np.load(scriptsDir+'subsNF_RT_acc.npy').flatten()
    subsNF_RT_acc = np.delete(subsNF_RT_acc,2)

    subsC_RT_acc = np.load(scriptsDir+'subsC_RT_acc.npy').flatten()
    
    subsAll_RT_acc = np.load(scriptsDir+'subsAll_RT_acc.npy')
    subsAll_RT_acc = np.delete(subsAll_RT_acc, 2)
    
    subsAll, subsAll_NFBlocks, subsAll_stableBlocks, subsNF, subsNF_NFBlocks, subsNF_stableBlocks, subsC, subsC_NFBlocks, subsC_stableBlocks = extractDividedStats(wanted_measure)
    
    subsAll = np.delete(subsAll,2)
    subsAll_NFBlocks = np.delete(subsAll_NFBlocks,2)
    subsAll_stableBlocks = np.delete(subsAll_stableBlocks,2)
    
    subsNF = np.delete(subsNF,2)
    subsNF_NFBlocks = np.delete(subsNF_NFBlocks,2)
    subsNF_stableBlocks = np.delete(subsNF_stableBlocks,2)
    
    # All subs
    print('All subs, all blocks corr: ', round((np.corrcoef(subsAll,subsAll_RT_acc))[0][1],3))
    print('All subs, NF blocks corr: ', round((np.corrcoef(subsAll_NFBlocks,subsAll_RT_acc))[0][1],3))
    print('All subs, stable blocks corr: ', round((np.corrcoef(subsAll_stableBlocks,subsAll_RT_acc))[0][1],3))
    
    # NF subs
    print('NF subs, all blocks corr: ', round((np.corrcoef(subsNF,subsNF_RT_acc))[0][1],3))
    print('NF subs, NF blocks corr: ', round((np.corrcoef(subsNF_NFBlocks,subsNF_RT_acc))[0][1],3))
    print('NF subs, stable blocks corr: ', round((np.corrcoef(subsNF_stableBlocks,subsNF_RT_acc))[0][1],3))
    
     # C subs
    print('C subs, all blocks corr: ', round((np.corrcoef(subsC,subsC_RT_acc))[0][1],3))
    print('C subs, NF blocks corr: ', round((np.corrcoef(subsC_NFBlocks,subsC_RT_acc))[0][1],3))
    print('C subs, stable blocks corr: ', round((np.corrcoef(subsC_stableBlocks,subsC_RT_acc))[0][1],3))

def behVSdecode(wanted_measure,ylabel,LOBO=False,masking=False):
    '''Uses LORO and LOBO'''
    
    all_d1, NF_d1, C_d1 = extractStatsDay(1,wanted_measure)
    
    subsNF_LORO = np.load(scriptsDir+'subsNF_LORO_09May.npy') # omit 09 May if old one
    subsC_LORO = np.load(scriptsDir+'subsC_LORO_09May.npy')
    
    subsNF_LOBO = np.load(scriptsDir+'subsNF_LOBO.npy') 
    subsC_LOBO = np.load(scriptsDir+'subsC_LOBO.npy')
    
    # For all subjects
    r_val_all = np.corrcoef(subsAll_LORO, all_d1)
    print('Correlation coefficient, all participants: ',r_val_all[0][1])
    
    # Create arrays
    if LOBO == False:
        subsC_LORO_a = np.array(subsC_LORO).flatten()
        subsNF_LORO_a = np.array(subsNF_LORO).flatten()
    if LOBO == True:
        subsC_LORO_a = np.array(subsC_LOBO).flatten()
        subsNF_LORO_a = np.array(subsNF_LOBO).flatten()
    
    C_d1_a = np.array(C_d1)
    NF_d1_a = np.array(NF_d1)
    
    subjID_C_a = np.array(subjID_C)
    subjID_NF_a = np.array(subjID_NF)

    fig,ax = plt.subplots()
    
    if masking != False:
        
        # Match the unwanted subject
        try:
            m_idx = subjID_C.index(masking)
        except:
            m_idx = subjID_NF.index(masking)
        
        mask = np.ones([11],dtype=bool)
        mask[m_idx] = 0 # Omit subj 18 in C
                
        # NF
        lm.fit(np.reshape((subsNF_LORO_a),[-1,1]),np.reshape((NF_d1_a),[-1,1]))
        ax.scatter((subsNF_LORO_a.tolist()),(NF_d1_a.tolist()),color='tomato',label='NF')
        ax.plot(np.reshape(subsNF_LORO_a,[-1,1]), lm.predict(np.reshape(subsNF_LORO_a,[-1,1])),linewidth=1,color='tomato')

        r_val_NF = np.corrcoef(subsNF_LORO_a.tolist(),(NF_d1_a.tolist()))
        print('Correlation coefficient, NF group: ',round(r_val_NF[0][1],3))
        
        for i, txt in enumerate(subjID_NF_a):
            ax.annotate(txt, (np.reshape(subsNF_LORO_a,[-1,1])[i], (np.reshape(NF_d1_a,[-1,1])[i])))
        
        # C
        lm.fit(np.reshape((subsC_LORO_a[mask]),[-1,1]),np.reshape((C_d1_a[mask]),[-1,1]))
        ax.scatter((subsC_LORO_a[mask].tolist()),(C_d1_a[mask].tolist()),color='dodgerblue',label='Control')
        ax.plot(np.reshape(subsC_LORO_a[mask],[-1,1]), lm.predict(np.reshape(subsC_LORO_a[mask],[-1,1])),linewidth=1,color='dodgerblue')

        r_val_C = np.corrcoef(subsC_LORO_a[mask].tolist(),(C_d1_a[mask].tolist()))
        print('Correlation coefficient, control group: ',round(r_val_C[0][1],3))
        
        plt.ylabel(ylabel + ' day 1')
        if LOBO == False:
            plt.xlabel('Mean offline decoding accuracy during stable blocks')
        if LOBO == True:
            plt.xlabel('Leave one block out offline decoding accuracy')
        plt.title('Sensitivity pre-training session (day 1)')
        
        # for i, txt in enumerate(subjID_C_a[mask]):
        #     ax.annotate(txt, (np.reshape(subsC_LORO_a[mask],[-1,1])[i], (np.reshape(C_d1_a[mask],[-1,1])[i])))
        
        plt.legend()

    if masking == False:        
        # NF
        lm.fit(np.reshape((subsNF_LORO_a),[-1,1]),np.reshape((NF_d1_a),[-1,1]))
        ax.scatter((subsNF_LORO_a.tolist()),(NF_d1_a.tolist()),color='tomato',label='NF')
        ax.plot(np.reshape(subsNF_LORO_a,[-1,1]), lm.predict(np.reshape(subsNF_LORO_a,[-1,1])),linewidth=1,color='tomato')

        r_val_NF = np.corrcoef(subsNF_LORO_a.tolist(),(NF_d1_a.tolist()))
        print('Correlation coefficient, NF group: ',round(r_val_NF[0][1],3))
        
        # for i, txt in enumerate(subjID_NF_a):
        #     ax.annotate(txt, (np.reshape(subsNF_LORO_a,[-1,1])[i], (np.reshape(NF_d1_a,[-1,1])[i])))
        
        # C
        lm.fit(np.reshape((subsC_LORO_a),[-1,1]),np.reshape((C_d1_a),[-1,1]))
        ax.scatter((subsC_LORO_a.tolist()),(C_d1_a.tolist()),color='dodgerblue',label='Control')
        ax.plot(np.reshape(subsC_LORO_a,[-1,1]), lm.predict(np.reshape(subsC_LORO_a,[-1,1])),linewidth=1,color='dodgerblue')

        r_val_C = np.corrcoef(subsC_LORO_a.tolist(),(C_d1_a.tolist()))
        print('Correlation coefficient, control group: ',round(r_val_C[0][1],3))
        
        plt.ylabel('Sensitivity pre-training session (day 1)')
        if LOBO == False:
            plt.xlabel('Mean offline decoding accuracy during stable blocks')
        if LOBO == True:
            plt.xlabel('Leave one block out offline decoding accuracy')
        
        plt.title('Stable blocks decoding accuracy vs. pre-training sensitivity')
        plt.grid(color='gainsboro',linewidth=0.5)
        
        # for i, txt in enumerate(subjID_C_a):
        #     ax.annotate(txt, (np.reshape(subsC_LORO_a,[-1,1])[i], (np.reshape(C_d1_a,[-1,1])[i])))
        
        plt.legend()
        
        
def behDay2VSdecode(wanted_measure,ylabel,LOBO=False):
    '''Uses LORO and LOBO, and compares with day 2 behavioral performance.'''
    
    all_d2, NF_d2, C_d2 = extractStatsDay(2,wanted_measure)
    
    all_d2 = np.delete(all_d2, 2)
    
    NF_d2 = np.delete(NF_d2, 2)

    subsNF_LORO = np.load(scriptsDir+'subsNF_LORO_09May.npy') # omit 09 May if old one
    subsC_LORO = np.load(scriptsDir+'subsC_LORO_09May.npy')
    
    subsNF_LORO = np.delete(subsNF_LORO, 2)
    
    subsAll_LORO = np.load(scriptsDir+'subsAll_LORO_09May.npy')
    subsAll_LORO = np.delete(subsAll_LORO, 2)
    
    subsNF_LOBO = np.load(scriptsDir+'subsNF_LOBO.npy') 
    subsC_LOBO = np.load(scriptsDir+'subsC_LOBO.npy')
    
    subsNF_LOBO = np.delete(subsNF_LOBO, 2)
    
    # For all subjects
    r_val_all = np.corrcoef(subsAll_LORO, all_d2)
    print('Correlation coefficient, all participants: ',r_val_all[0][1])

    
    # Create arrays
    if LOBO == False:
        subsC_LORO_a = np.array(subsC_LORO).flatten()
        subsNF_LORO_a = np.array(subsNF_LORO).flatten()
    if LOBO == True:
        subsC_LORO_a = np.array(subsC_LOBO).flatten()
        subsNF_LORO_a = np.array(subsNF_LOBO).flatten()
    
    C_d2_a = np.array(C_d2)
    NF_d2_a = np.array(NF_d2)
    
    subjID_C_a = np.array(subjID_C)
    subjID_NF_a = np.array(subjID_NF)
    
    subjID_NF_a = np.delete(subjID_NF_a, 2)

    fig,ax = plt.subplots()
      
    # NF
    lm.fit(np.reshape((subsNF_LORO_a),[-1,1]),np.reshape((NF_d2_a),[-1,1]))
    ax.scatter((subsNF_LORO_a.tolist()),(NF_d2_a.tolist()),color='tomato',label='NF')
    ax.plot(np.reshape(subsNF_LORO_a,[-1,1]), lm.predict(np.reshape(subsNF_LORO_a,[-1,1])),linewidth=1,color='tomato')

    r_val_NF = np.corrcoef(subsNF_LORO_a.tolist(),(NF_d2_a.tolist()))
    print('Correlation coefficient, NF group: ',round(r_val_NF[0][1],3))
    
    # for i, txt in enumerate(subjID_NF_a):
    #     ax.annotate(txt, (np.reshape(subsNF_LORO_a,[-1,1])[i], (np.reshape(NF_d2_a,[-1,1])[i])))
    
    # C
    lm.fit(np.reshape((subsC_LORO_a),[-1,1]),np.reshape((C_d2_a),[-1,1]))
    ax.scatter((subsC_LORO_a.tolist()),(C_d2_a.tolist()),color='dodgerblue',label='Control')
    ax.plot(np.reshape(subsC_LORO_a,[-1,1]), lm.predict(np.reshape(subsC_LORO_a,[-1,1])),linewidth=1,color='dodgerblue')

    r_val_C = np.corrcoef(subsC_LORO_a.tolist(),(C_d2_a.tolist()))
    print('Correlation coefficient, control group: ',round(r_val_C[0][1],3))
    
    plt.ylabel(ylabel + ' EEG session (day 2)')
    if LOBO == False:
        plt.xlabel('Mean offline decoding accuracy during stable blocks (LORO)')
    if LOBO == True:
        plt.xlabel('Leave one block out offline decoding accuracy')
    
    plt.title('Stable blocks decoding accuracy vs. EEG session response time')
    plt.grid(color='gainsboro',linewidth=0.5)

    
    # for i, txt in enumerate(subjID_C_a):
    #     ax.annotate(txt, (np.reshape(subsC_LORO_a,[-1,1])[i], (np.reshape(C_d2_a,[-1,1])[i])))
    
    plt.legend()
    
    
def improvStimuli(wanted_measure,actual_stim=False,rt_acc=False,LORO=False):
    ''' Computes how task difficulty correlates with learning '''
    
    subsNF_RT_acc = np.load(scriptsDir+'subsNF_RT_acc.npy').flatten()
    subsC_RT_acc = np.load(scriptsDir+'subsC_RT_acc.npy').flatten()
    
    all_d1, NF_d1, C_d1 = extractStatsDay(1,wanted_measure)
    all_d3, NF_d3, C_d3 = extractStatsDay(3,wanted_measure)
    
    if wanted_measure == 'rt':
        diffNF = (np.asarray(NF_d1) - np.asarray(NF_d3))
        diffC = (np.asarray(C_d1) - np.asarray(C_d3))
        diff = (np.asarray(all_d1) - np.asarray(all_d3))
    else:
        diffNF = (np.asarray(NF_d3) - np.asarray(NF_d1))
        diffC = (np.asarray(C_d3) - np.asarray(C_d1))
        diff = (np.asarray(all_d3) - np.asarray(all_d1))
    
    if actual_stim == True and rt_acc == True: # RT accuracy
        # For all subjects
        # diff = (np.asarray(all_d3) - np.asarray(all_d1))
        # lm.fit(np.reshape(subsAll_RT_acc,[-1,1]),np.reshape(diff,[-1,1]))

        # plt.scatter(subsAll_RT_acc,diff)
        # plt.plot(np.reshape(subsAll_RT_acc,[-1,1]), lm.predict(np.reshape(subsAll_RT_acc,[-1,1])),linewidth=1)
        # plt.ylabel('Change in sensitivity from day 1 to 3\n Positive values indicate improvement')
        # np.corrcoef(subsAll_RT_acc,diff)
        plt.figure(random.randint(0,50))
        plt.xlabel('RT decoding acc')
        # NF
        lm.fit(np.reshape((subsNF_RT_acc),[-1,1]),np.reshape((diffNF),[-1,1]))
        plt.scatter((subsNF_RT_acc.tolist()),(diffNF.tolist()),color='tomato',label='NF')
        plt.plot(np.reshape(subsNF_RT_acc,[-1,1]), lm.predict(np.reshape(subsNF_RT_acc,[-1,1])),linewidth=1,color='tomato')

        r_val_NF = np.corrcoef(subsNF_RT_acc.tolist(),(diffNF.tolist()))
        print('Correlation coefficient, NF group: ',round(r_val_NF[0][1],3))
        
        # for i, txt in enumerate(subjID_NF_a):
        #     ax.annotate(txt, (np.reshape(subsNF_LORO_a,[-1,1])[i], (np.reshape(NF_d1_a,[-1,1])[i])))
        
        # C
        lm.fit(np.reshape((subsC_RT_acc),[-1,1]),np.reshape((diffC),[-1,1]))
        plt.scatter((subsC_RT_acc.tolist()),(diffC.tolist()),color='dodgerblue',label='Control')
        plt.plot(np.reshape(subsC_RT_acc,[-1,1]), lm.predict(np.reshape(subsC_RT_acc,[-1,1])),linewidth=1,color='dodgerblue')
        plt.title('RT decoding accuracy vs. improvement')
        plt.grid(color='gainsboro',linewidth=0.5)
        
    if actual_stim == True and LORO == True: # correlates LORO with improvement, actual values.
        
        subsNF_LORO = np.load(scriptsDir+'subsNF_LORO_09May.npy').flatten()
        subsC_LORO = np.load(scriptsDir+'subsC_LORO_09May.npy').flatten()
        
        subsAll_LORO = np.load(scriptsDir+'subsAll_LORO_09May.npy').flatten()
        
        r_all = np.corrcoef(subsAll_LORO,diff)
        print('Correlation coefficient, all: ',round(r_all[0][1],3))

        fig,ax = plt.subplots()
        plt.xlabel('Mean offline decoding accuracy during stable blocks')
        # NF
        lm.fit(np.reshape((subsNF_LORO),[-1,1]),np.reshape((diffNF),[-1,1]))
        plt.scatter((subsNF_LORO.tolist()),(diffNF.tolist()),color='tomato',label='NF')
        plt.plot(np.reshape(subsNF_LORO,[-1,1]), lm.predict(np.reshape(subsNF_LORO,[-1,1])),linewidth=1,color='tomato')

        r_val_NF = np.corrcoef(subsNF_LORO,diffNF)
        print('Correlation coefficient, NF group: ',round(r_val_NF[0][1],3))
        
        # for i, txt in enumerate(subjID_NF_a):
        #     ax.annotate(txt, (np.reshape(subsNF_LORO,[-1,1])[i], (np.reshape(diffNF,[-1,1])[i])))
        
        # C
        lm.fit(np.reshape((subsC_LORO),[-1,1]),np.reshape((diffC),[-1,1]))
        plt.scatter((subsC_LORO.tolist()),(diffC.tolist()),color='dodgerblue',label='Control')
        plt.plot(np.reshape(subsC_LORO,[-1,1]), lm.predict(np.reshape(subsC_LORO,[-1,1])),linewidth=1,color='dodgerblue')

        plt.ylabel(r'$\Delta$ response time (s) (day 1 to 3)')
        plt.title(r'$\Delta$ Response time vs. stable blocks decoding accuracy')
        plt.grid(color='gainsboro',linewidth=0.5)
        plt.legend()
        
        r_val_C = np.corrcoef(subsC_LORO,diffC)
        print('Correlation coefficient, control group: ',round(r_val_C[0][1],3))
        
        
    if actual_stim == False:
        ''' For controls, add the value that they were actually exposed to, i.e. matched NF participant'''
        # For matching
        NF_group = ['07','08','11','13','14','16','19','22','26','27','30']
        C_group = ['17','18','15','24','21','33','25','32','34','23','31']
        
        subjID_C = ['15','17','18','21','23','24','25','31','32','33','34']

        # I want the C diff structured in terms of which stimuli they saw
        idxLst = [2,0,1,4,9,3,6,10,7,5,8]
        
        newSortC = [y for x,y in sorted(zip(idxLst,subjID_C))] # I.e. just sort behavioral measure based on this list. 
        
        diffCnewsort = [y for x,y in sorted(zip(idxLst,diffC))]
        
        # If using alphas instead 
        subsNF_meanAlphas = np.load(scriptsDir+'subsNF_meanAlphas.npy')

        plt.figure(random.randint(0,30))
        lm.fit(np.reshape((subsNF_meanAlphas),[-1,1]),np.reshape((diffNF),[-1,1]))
        # plt.scatter((subsNF_RT_acc.tolist()),(diffNF.tolist()),color='tomato',label='NF')
        plt.plot(np.reshape(subsNF_meanAlphas,[-1,1]), lm.predict(np.reshape(subsNF_meanAlphas,[-1,1])),linewidth=1,color='tomato')
        
        lm.fit(np.reshape((subsNF_meanAlphas),[-1,1]),np.reshape((diffCnewsort),[-1,1]))
        plt.plot(np.reshape(subsNF_meanAlphas,[-1,1]), lm.predict(np.reshape(subsNF_meanAlphas,[-1,1])),linewidth=1,color='dodgerblue')

        plt.scatter((subsNF_meanAlphas),(diffNF),color='tomato',label='NF')
        plt.scatter((subsNF_meanAlphas),(diffCnewsort),color='dodgerblue',label='Control')
        plt.ylabel(r'$\Delta$ response time (s) (day 1 to 3)')
        plt.xlabel('Mean task-relevant image proportion (alpha)')
        plt.title(r'$\Delta$ Response time vs. mean task-relevant image proportion')
        plt.grid(color='gainsboro',linewidth=0.5)
        plt.legend(framealpha=0.3)
        
        for i in range(11):
            plt.plot([subsNF_meanAlphas[i],subsNF_meanAlphas[i]], [diffNF[i],diffCnewsort[i]],color='gray')
        
        r_C = np.corrcoef(subsNF_meanAlphas,diffCnewsort)
        r_NF = np.corrcoef(subsNF_meanAlphas,diffNF)

        allNewSort = subsNF_meanAlphas.tolist() + subsNF_meanAlphas.tolist()
        allDiffNewSort = diffNF.tolist() + diffCnewsort
        
        r_all = np.corrcoef(allNewSort,allDiffNewSort)
        
        print('Corr coef of all participants: ',round(r_all[0][1],3))
        print('Corr coef of NF participants: ',round(r_NF[0][1],3))
        print('Corr coef of control participants: ',round(r_C[0][1],3))
        
        # Using decoding RT on x-axis, DOUBLE X AXIS LEGEND?
        
        # plt.figure(random.randint(0,30))
        # plt.scatter((subsNF_RT_acc),(diffCnewsort),color='tomato',label='Control with NF stimuli')
        # # plt.scatter((subsNF_RT_acc),(diffC),color='green',label='green')
        # plt.scatter((subsNF_RT_acc),(diffNF),color='green',label='NF')
        # plt.ylabel(r'$\Delta$ sensitivity (day 1 to 3)')
        # plt.xlabel('Real-time decoding a')
        
        # np.corrcoef(subsNF_RT_acc,diffCnewsort)
        
        # allNewSort = subsNF_RT_acc.tolist() + subsNF_RT_acc.tolist()
        # allDiffNewSort = diffNF.tolist() + diffCnewsort
        
        # np.corrcoef(allNewSort,allDiffNewSort)