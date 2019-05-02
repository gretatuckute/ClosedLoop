# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:12:38 2019

V3: Responses extracted 150-1150ms, block-wise information extracted in functions based on responseTimes.

V2 and V3 comparison: Checked 29April, legit. V2 and V3 analysis corresponds.

@author: Greta
"""

import pickle
from matplotlib import pyplot as plt
import os
import statistics
import numpy as np
from scipy import stats
import seaborn as sns
os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\')
from responseTime_func import findRTsBlocks
import random

saveDir = 'P:\\closed_loop_data\\beh_analysis\\'
os.chdir(saveDir)

plt.style.use('seaborn-pastel')

font = {'family' : 'sans-serif',
       'weight' : 1,
       'size'   : 12}

#%% Variables
subjID_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']
subjID_NF = ['07','08','11','13','14','16','19','22','26','27','30']
subjID_C = ['15','17','18','21','23','24','25','31','32','33','34']

#%% RTs surrounding lures
def extractSurroundingRTs(subjID):
        
    with open('BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]
    
    CR1 = sub['surrounding_CR_Lst_day_1']
    CR3 = sub['surrounding_CR_Lst_day_3']
    
    FR1 = sub['surrounding_FR_Lst_day_1']
    FR3 = sub['surrounding_FR_Lst_day_3']
    
    # Computing day 1 and 3 together
    CR13m = [statistics.mean(k) for k in zip(CR1, CR3)]
    FR13m = [statistics.mean(k) for k in zip(FR1, FR3)]
    
    return np.asarray(CR1), np.asarray(CR3), np.asarray(FR1), np.asarray(FR3), np.asarray(CR13m), np.asarray(FR13m)

#%%
surroundingRT_all = np.zeros((22,6),dtype=np.ndarray)

for idx, subjID in enumerate(subjID_all):
    surroundingRT_all[idx,:] = extractSurroundingRTs(subjID)

# Mean over CR13m and FR13m for all subjects, and separately for day 1 and 3
CR1_all = surroundingRT_all[:,0]
CR3_all = surroundingRT_all[:,1]
FR1_all = surroundingRT_all[:,2]
FR3_all = surroundingRT_all[:,3]

CR13_all = surroundingRT_all[:,4]
FR13_all = surroundingRT_all[:,5]

CR1m = np.mean(CR1_all)
CR3m = np.mean(CR3_all)
FR1m = np.mean(FR1_all)
FR3m = np.mean(FR3_all)
CR13m = np.mean(CR13_all)
FR13m = np.mean(FR13_all)

# Standard error of the mean for errorbars
CR1_yerr=(np.std(CR1_all))/np.sqrt(22)
CR3_yerr=(np.std(CR3_all))/np.sqrt(22)
FR1_yerr=(np.std(FR1_all))/np.sqrt(22)
FR3_yerr=(np.std(FR3_all))/np.sqrt(22)
CR13_yerr=(np.std(CR13_all))/np.sqrt(22)
FR13_yerr=(np.std(FR13_all))/np.sqrt(22)


#%% Plot RTs surrounding lures
ticks = ['-3','-2','-1','lure','1','2','3']

# Both days
#plt.plot(all_c13m,color='green',)
#plt.plot(all_f13m,color='red')

plt.figure(1)
plt.errorbar(np.arange(0,7),CR13m,yerr=CR13_yerr,color='green')
plt.errorbar(np.arange(0,7),FR13m,yerr=FR13_yerr,color='red')
plt.title('Response times surrounding lures \nDay 1 and 3 - all participants')
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('RT (ms)')

# Day 1
plt.figure(2)
#plt.plot(all_c1m,color='green')
#plt.plot(all_f1m,color='red')
plt.errorbar(np.arange(0,7),CR1m,yerr=CR1_yerr,color='green')
plt.errorbar(np.arange(0,7),FR1m,yerr=FR1_yerr,color='red')
plt.title('Response times surrounding lures \nDay 1, all participants')
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('RT (ms)')

# Day 3
plt.figure(3)
#plt.plot(all_c3m,color='green')
#plt.plot(all_f3m,color='red')
plt.errorbar(np.arange(0,7),CR3m,yerr=CR3_yerr,color='green')
plt.errorbar(np.arange(0,7),FR3m,yerr=FR3_yerr,color='red')
plt.title('Response times surrounding lures \nDay 3, all participants')
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('RT (ms)')


#%% 
def computeStats(subjID):
    '''Computes stats based on days (statsDay) and blocks for each day (both statsBlock: day 1, 3, 4, 5 and statsBlock_day2)
    '''
    
    with open('BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]
    
    statsDay = np.zeros((5,8))
    statsBlock = np.zeros((4,16,5))
    statsBlock_day2 = np.zeros((48,5))
    
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
        
        accuracy = (TP+TN)/(TP+TN+FP+FN)
#        accuracy_all = (TP+TN)/nKeypress # Out of all the keypresses that day
        
        statsDay[idx,:] = sensitivity, specificity, accuracy, lure_RT_mean, nonlure_RT_mean, RT_mean, nKeypress, nNaN

        # Block-wise
        for block in range(1,int(len(responseTimes)/50)+1):
            print(block)
            
            if subjID == '11' and expDay == '2': 
                TN, FP, FN, TP, lure_RT_mean, nonlure_RT_mean, RT_mean, nNaN = [np.nan]*8
            else:
                TN, FP, FN, TP, lure_RT_mean, nonlure_RT_mean, RT_mean, nNaN = findRTsBlocks(catFile,responseTimes,block=block)
            
            sensitivity = TP/(TP+FN)
            specificity = TN/(TN+FP)
            
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            
            if expDay == '2':
                statsBlock_day2[block-1,:] = sensitivity, specificity, accuracy, RT_mean, nNaN
        
            if expDay != '2':
                statsBlock[idx_c,block-1,:] = sensitivity, specificity, accuracy, RT_mean, nNaN
                
        if expDay != '2':            
            idx_c += 1

    return statsDay, statsBlock, statsBlock_day2

#%% Extract stats for all subjects
statsDay_all = {} # NOT block-wise, statsDay. Checked 30April whether this corresponds to the other (V2) beh measures. OK.
statsBlock_all = {}
statsBlockDay2_all = {}

# Extract stats for all subjects
for idx,subjID in enumerate(subjID_all):
    statsDay, statsBlock, statsBlock_day2 = computeStats(subjID)
    
    statsDay_all[subjID] = statsDay
    statsBlock_all[subjID] = statsBlock
    statsBlockDay2_all[subjID] = statsBlock_day2

#%% Functions for extracting stats values from statsDay_all, statsBlock_all and statsBlockDay2_all
    
def extractStatsDay(day_idx,wanted_measure):
    #sensitivity, specificity, accuracy, lure_RT_mean, nonlure_RT_mean, RT_mean, nKeypress, nNaN
    day_idx = day_idx-1
    
    if wanted_measure == 'sen':
        w_idx = 0
    if wanted_measure == 'spec':
        w_idx = 1
    if wanted_measure == 'acc':
        w_idx = 2
    if wanted_measure == 'rt_lure':
        w_idx = 3
    if wanted_measure == 'rt_nlure':
        w_idx = 4
    if wanted_measure == 'rt':
        w_idx = 5
    if wanted_measure == 'nkeypress':
        w_idx = 6
    if wanted_measure == 'nan':
        w_idx = 7
    
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
        
    if wanted_measure == 'sen':
        w_idx = 0
    if wanted_measure == 'spec':
        w_idx = 1
    if wanted_measure == 'acc':
        w_idx = 2
    if wanted_measure == 'rt':
        w_idx = 3
    if wanted_measure == 'nan':
        w_idx = 4
    
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

#%% 
def make4Bars(wanted_measure,title,ylabel):
    '''
    Comparing day 1 and 3, NF and controls, using the statsDay_all structure and extract function.
    
    Connected lines bar plot.
    '''
    
    all_d1, NF_d1, C_d1 = extractStatsDay(1,wanted_measure)
    all_d3, NF_d3, C_d3 = extractStatsDay(3,wanted_measure)
    
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
    
    for i in range(11):
        plt.plot([(np.full(11,1))[i],(np.full(11,2))[i]], [(NF_d1)[i],(NF_d3)[i]],color='gray')
        plt.plot([(np.full(11,3))[i],(np.full(11,4))[i]], [(C_d1)[i],(C_d3)[i]],color='gray')
        
    t_fb = stats.ttest_rel(NF_d1,NF_d3)
    t_c = stats.ttest_rel(C_d1,C_d3)
    
    print('P-value between day 1 and 3, NF: ', round(t_fb[1],3))
    print('P-value between day 1 and 3, C: ', round(t_c[1],3))

    return plt, t_fb, t_c

def make2Bars(wanted_measure,title,ylabel):
    '''
    Comparing day 2, NF and controls, using the statsDay_all structure and extract function.    
    '''
    
    all_d2, NF_d2, C_d2 = extractStatsDay(2,wanted_measure)
    
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

#%% # Plots comparing day 1 and 3
pl, t_fb, t_c = make4Bars('sen','Sensitivity','Response sensitivity')
pl, t_fb, t_c = make4Bars('spec','Specificity','Response specificity')
pl, t_fb, t_c = make4Bars('acc','Accuracy','Response accuracy')

pl, t_fb, t_c = make4Bars('rt','Overall response time','Response time (s)')
pl, t_fb, t_c = make4Bars('rt_lure','Response time for lures','Response time (s)')
pl, t_fb, t_c = make4Bars('rt_nlure','Response time for non lures','Response time (s)')
pl, t_fb, t_c = make4Bars('nkeypress','Number of total keypresses','Number keypresses')
pl, t_fb, t_c = make4Bars('nan','Number of total NaNs','Number ') # Not saved

#%%  Investigate day 2, using make2Bars and StatsDayAll (NOT block-wise)
pl, t, sen_all_d2 = make2Bars('sen','Sensitivity day 2','Sensitivity') 
pl, t, acc_all_d2 = make2Bars('acc','Accuracy day 2','Accuracy') 
pl, t, rt_all_d2 = make2Bars('rt','RT day 2','RT') 
pl, t, nkeypress_all_d2 = make2Bars('nkeypress','Number of total keypresses','Number keypresses') 

#%% Is good decoding accuracy correlated with a good day 2 behavioral response?
# Omit 11 

# Sensitivity
sen_all_d2_c = np.copy(sen_all_d2)
sen_all_d2_c = sen_all_d2_c[~np.isnan(sen_all_d2_c)]

# Accuracy
acc_all_d2_c = np.copy(acc_all_d2)
acc_all_d2_c = acc_all_d2_c[~np.isnan(acc_all_d2_c)]

# RT
rt_all_d2_c = np.copy(rt_all_d2)
rt_all_d2_c = rt_all_d2_c[~np.isnan(rt_all_d2_c)]

# Load RT accuracy np array
subsAll_RT_acc = np.load('subsAll_RT_acc.npy')
subsAll_c = np.copy(subsAll_RT_acc) # From EEG inv 18April
subsAll_c = np.delete(subsAll_c, 2)

# Sensitivity
plt.scatter(subsAll_c,sen_all_d2_c)
plt.ylabel('Sensitivity day 2')
plt.xlabel('Real-time decoding accuracy (NF blocks)')
plt.title('Sensitivity day 2 vs. real-time decoding accuracy, N=21')

np.corrcoef(sen_all_d2_c,subsAll_c)
stats.linregress(sen_all_d2_c,subsAll_c)

# Accuracy
plt.scatter(subsAll_c,acc_all_d2_c)
plt.ylabel('Accuracy day 2')
plt.xlabel('Real-time decoding accuracy (NF blocks)')
plt.title('Accuracy day 2 vs. real-time decoding accuracy, N=21')

stats.linregress(acc_all_d2_c,subsAll_c)

# Plot RT vs decoding acc
plt.scatter(subsAll_c,rt_all_d2_c)
plt.ylabel('Response time day 2')
plt.xlabel('Real-time decoding accuracy, bias corrected')
plt.title('Response time day 2 vs. real-time decoding accuracy, N=21')

stats.linregress(rt_all_d2_c,subsAll_c)
np.corrcoef(rt_all_d2_c,subsAll_c)

#%% Is good decoding accuracy STABLE blocks correlated with a good day 1 behavioral response?

sen_all_d1, sen_NF_d1, sen_C_d1 = extractStatsDay(1,'sen')
acc_all_d1, acc_NF_d1, acc_C_d1 = extractStatsDay(1,'acc')

# Sensitivity day 1 vs. day2 accuracy
plt.scatter(subsAll_RT_acc,sen_all_d1)
plt.ylabel('Sensitivity day 2')
plt.xlabel('Real-time decoding accuracy (NF blocks)')
plt.title('Sensitivity day 1 vs. real-time decoding accuracy, N=22')

np.corrcoef(sen_all_d1,subsAll_RT_acc)

# LOBO
subsAll_LOBO = np.load('subsAll_LOBO.npy')

# Sensitivity day 1 vs LOBO accuracy day 2
plt.scatter(subsAll_LOBO,sen_all_d1)
np.corrcoef(subsAll_LOBO,sen_all_d1)

# Accuracy
np.corrcoef(subsAll_LOBO,acc_all_d1)

# Only NF
np.corrcoef(np.asarray(subsNF_LOBO).flatten(),np.asarray(sen_NF_d1))
stats.linregress(np.asarray(subsNF_LOBO).flatten(),sen_NF_d1)

# LORO
np.corrcoef(subsAll_LORO,acc_all_d1)

# Somewhere here: check whether good decoding acc corresponds with good day 1 to 3 improvement

#%% ########## BLOCK-WISE ANALYSIS #############

def behBlock(day,wanted_measure,title,ylabel):
    all_d, NF_d, C_d = extractStatsBlock(day,wanted_measure)

    colorC = sns.color_palette("Blues",11)
    colorNF = sns.color_palette("Reds",11)
    
    # Plot NF subjects
    plt.figure(random.randint(0,100))
    for j in range(len(NF_d)):
        plt.plot(NF_d[j],color=colorNF[j],linewidth=0.8)
        
    plt.plot(np.mean(NF_d,axis=0),label='Mean NF group',color='red',linewidth=2.0)
    
    # Plot C subjects
    for i in range(len(C_d)):
        plt.plot(C_d[i],color=colorC[i],linewidth=0.8)
        
    plt.plot(np.mean(C_d,axis=0),label='Mean control group',color='blue',linewidth=2.0)
    
    plt.plot(np.mean(all_d,axis=0),label='Mean all participants',color='black',linewidth=2.0)
    plt.title(title)
    plt.xticks(np.arange(0,16),[str(item) for item in np.arange(1,17)])
    plt.xlabel('Block number, day '+str(day))
    plt.ylabel(ylabel)
    plt.legend()

#%% 
behBlock(1,'sen','Sensitivity across blocks, all participants, day 1','Sensitivity')
behBlock(3,'sen','Sensitivity across blocks, all participants, day 3','Sensitivity')

behBlock(1,'spec','Specificity across blocks, all participants, day 1','Specificity')
behBlock(3,'spec','Specificity across blocks, all participants, day 3','Specificity')

behBlock(1,'acc','Accuracy across blocks, all participants, day 1','Accuracy')
behBlock(3,'acc','Accuracy across blocks, all participants, day 3','Accuracy')

behBlock(1,'rt','Response time across blocks, all participants, day 1','Response time (ms)')
behBlock(3,'rt','Response time across blocks, all participants, day 3','Response time (ms)')

#%% Output responseTimes
def outputResponseTimes(subjID,expDay):
    '''Outputs responseTimes for a chosen day
    '''
    with open('BehV3_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]
    
    responseTimes = sub['responseTimes_day'+expDay]
    
    return responseTimes


#%%
responseTimes_all = []

# Extract responseTimes for all subjects
for idx,subjID in enumerate(subjID_all):
    responseTimes = outputResponseTimes(subjID,'1')
    responseTimes_all.append(responseTimes)
        
responseTimes_m = np.nanmean(responseTimes_all,axis=0)

plt.plot(responseTimes_m)

#%% ResponseTimes plot

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
    plt.title('Response times for all participants N=22, day '+str(expDay))
    
    if expDay != '2':
        plt.xticks(np.arange(0,850,50),[str(item) for item in np.arange(0,850,50)])
    else:
        plt.xticks(np.arange(0,2450,50),[str(item) for item in np.arange(0,2450,50)],rotation=40)
        
    plt.xlabel('Trial number')
    plt.ylabel('Response time (ms)')
    
    for j in range(len(responseTimes_all)):
        plt.plot(responseTimes_all[j],color=colorAll[j],linewidth=0.3)
        
    plt.plot(np.nanmean(responseTimes_all,axis=0),label='Mean response time, all participants, N=22',color='black',linewidth=2.0)
    plt.legend()


#%% 
plotResponseTimes('1')
plotResponseTimes('3')

plotResponseTimes('2')

#%%