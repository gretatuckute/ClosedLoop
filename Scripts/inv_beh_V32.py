# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:12:38 2019

V3: Responses extracted 150-1150ms, block-wise information extracted in functions based on responseTimes.
V2 and V3 comparison: Checked 29April, legit. V2 and V3 analysis corresponds.
V3.2: Implements relative reduction in error rate (RRER) and removes a lot of stuff.

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
import numpy.ma as ma

scriptsDir = 'C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\'
os.chdir(scriptsDir)
from variables import *
from responseTime_func import findRTsBlocks
from beh_FUNC import *

#%% RTs surrounding lures
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
plt.figure(1)
plt.errorbar(np.arange(0,7),CR13m,yerr=CR13_yerr,color='green', label='CR')
plt.errorbar(np.arange(0,7),FR13m,yerr=FR13_yerr,color='red', label='FA')
plt.title('Response times surrounding lure trials') # 
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('Response time (s)')
plt.grid(color='gainsboro',linewidth=0.5)

stats.ttest_rel(CR13m,FR13m,nan_policy='omit')

# Day 1
plt.figure(2)
plt.errorbar(np.arange(0,7),CR1m,yerr=CR1_yerr,color='green')
plt.errorbar(np.arange(0,7),FR1m,yerr=FR1_yerr,color='red')
plt.title('Response times surrounding lures \nDay 1, all participants')
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('RT (ms)')

# Day 3
plt.figure(3)
plt.errorbar(np.arange(0,7),CR3m,yerr=CR3_yerr,color='green')
plt.errorbar(np.arange(0,7),FR3m,yerr=FR3_yerr,color='red')
plt.title('Response times surrounding lures \nDay 3, all participants')
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('RT (ms)')

#%% # Plots comparing day 1 and 3
pl, t_fb, t_c, sen_all_d1 = make4Bars('sen','Sensitivity: Pre- to post-training','Sensitivity',RER=True)
pl, t_fb, t_c, all_d1 = make4Bars('spec','Specificity','Response specificity')
pl, t_fb, t_c, all_d1 = make4Bars('fpr','FPR','False positive rate')
pl, t_fb, t_c, acc_all_d1 = make4Bars('acc','Accuracy: Pre- to post-training','Accuracy',RER=True)

pl, t_fb, t_c, all_d1 = make4Bars('rt','Response time: Pre- to post-training','Response time (s)')
pl, t_fb, t_c, all_d1 = make4Bars('rt_lure','Response time for lures','Response time (s)')
pl, t_fb, t_c, all_d1 = make4Bars('rt_nlure','Response time for non lures','Response time (s)')
pl, t_fb, t_c, all_d1 = make4Bars('nkeypress','Number of total keypresses','Number keypresses')
pl, t_fb, t_c = make4Bars('nan','Number of total NaNs','Number ') # Not saved

# Comparing day 1 and day 3, removing variance
pl, t_fb, t_c, sen_all_d1 = make4BarsRemoveVar('sen','Sensitivity','Response sensitivity')
pl, t_fb, t_c, acc_all_d1 = make4BarsRemoveVar('acc','Accuracy, relative','Response accuracy')
pl, t_fb, t_c, rt_all_d1 = make4BarsRemoveVar('rt','Response time, relative','Response time')
pl, t_fb, t_c, rt_all_d1 = make4BarsRemoveVar('spec','Spec, relative','Response specificity')


#%% Compare day 4 and 5
pl, t_fb, t_c, all_d1 = make4Bars('sen','Sensitivity','Response sensitivity',part2=True)
pl, t_fb, t_c, all_d1 = make4Bars('acc','Accuracy','Response accuracy',part2=True)
pl, t_fb, t_c, all_d1 = make4Bars('rt','Response time','Response time ',part2=True)


#%%  Investigate day 2, using make2Bars and StatsDayAll (NOT block-wise)
pl, t, sen_all_d2 = make2Bars('sen','Sensitivity day 2','Sensitivity',RER=True) 
pl, t, spec_all_d2 = make2Bars('spec','Specificity day 2','Specificity') 
pl, t, acc_all_d2 = make2Bars('acc','Accuracy day 2','Accuracy') 
pl, t, fpr_all_d2 = make2Bars('fpr','FPR day 2','FPR') 
pl, t, rt_all_d2 = make2Bars('rt','RT day 2','RT') 
pl, t, nkeypress_all_d2 = make2Bars('nkeypress','Number of total keypresses','Number keypresses') 

#%% Make 3 day plot
def threeDay(wanted_measure):
    all_d1, NF_d1, C_d1 = extractStatsDay(1,wanted_measure)
    all_d2, NF_d2, C_d2 = extractStatsDay(2,wanted_measure)
    all_d3, NF_d3, C_d3 = extractStatsDay(3,wanted_measure)
    
    # Cmap for NF
    normNF = matplotlib.colors.Normalize(vmin=np.min(subsNF_RT_acc), vmax=np.max(subsNF_RT_acc))
    normC = matplotlib.colors.Normalize(vmin=np.min(subsC_RT_acc), vmax=np.max(subsC_RT_acc))
    
    # Manual colormap
    # colorsNF = [(1, 0.86, 0.8), (0.65, 0, 0)]     # (0.7, 0.17, 0.05)
    # cmNF = LinearSegmentedColormap.from_list('test', colorsNF)
    # cmNFget=cm.get_cmap(cmNF)
    cmReds=cm.get_cmap("Reds")
    cmBlues=cm.get_cmap("Blues")

    plt.figure(random.randint(0,80))
    plt.xticks([1,2,3],['NF day 1','NF day 2', 'NF day 3'])
    plt.ylabel('Behavioral accuracy')        
    plt.title('Behavioral accuracy day 1, 2 and 3 for NF group')
    plt.scatter(np.full(11,1),NF_d1,c=subsNF_RT_acc,cmap='Reds',s=60)
    plt.scatter(np.full(11,2),NF_d2,c=subsNF_RT_acc,cmap='Reds',s=60)
    plt.scatter(np.full(11,3),NF_d3,c=subsNF_RT_acc,cmap='Reds',s=60)
    cbar = plt.colorbar()
    cbar.set_label('Real-time decoding accuracy, NF group')

    for i in range(11):
        plt.plot([(np.full(11,1))[i],(np.full(11,2))[i],(np.full(11,3))[i]],\
                  [(NF_d1)[i],(NF_d2)[i],(NF_d3)[i]],c=cmReds(normNF(subsNF_RT_acc[i])),linewidth=3)
    

    plt.plot([(np.full(11,1))[2],(np.full(11,3))[2]],\
                  [(NF_d1)[2],(NF_d3)[2]],c=cmNFget(normNF(subsNF_RT_acc[2])),linewidth=3)
    
    
    # Make colorbar for controls
    # plt.figure(random.randint(0,80))
    # plt.xticks([1,2,3],['NF day 1','NF day 2', 'NF day 3'])
    # plt.ylabel('Behavioral accuracy')        
    # plt.title('Behavioral accuracy day 1, 2 and 3 for NF group')
    # plt.scatter(np.full(11,1),NF_d1,c=subsC_RT_acc,cmap='Blues',s=60)
    # plt.scatter(np.full(11,2),NF_d2,c=subsC_RT_acc,cmap='Blues',s=60)
    # plt.scatter(np.full(11,3),NF_d3,c=subsC_RT_acc,cmap='Blues',s=60)
    # cbar = plt.colorbar()
    # cbar.set_label('Real-time decoding accuracy, control group')
    
    NF_group_match1 = []
    C_group_match1 = []

    for idx,subjID in enumerate(subjID_all):
        if subjID in NF_group:
            NF_group_match1.append([subjID,all_d1[idx]])
        else:
            C_group_match1.append([subjID,all_d1[idx]])
    
    C_re1 = []
    for subjID in C_group:
        for subORDER in C_group_match1:
            if subjID == subORDER[0]:
                C_re1.append([subjID,subORDER[1]]) 
                
    C_re1 = [item[1] for item in C_re1]
     
    NF_group_match2 = []
    C_group_match2 = []

    for idx,subjID in enumerate(subjID_all):
        if subjID in NF_group:
            NF_group_match2.append([subjID,all_d2[idx]])
        else:
            C_group_match2.append([subjID,all_d2[idx]])
    
    C_re2 = []
    for subjID in C_group:
        for subORDER in C_group_match2:
            if subjID == subORDER[0]:
                C_re2.append([subjID,subORDER[1]]) 
                
    C_re2 = [item[1] for item in C_re2]
    
    NF_group_match3 = []
    C_group_match3 = []

    for idx,subjID in enumerate(subjID_all):
        if subjID in NF_group:
            NF_group_match3.append([subjID,all_d3[idx]])
        else:
            C_group_match3.append([subjID,all_d3[idx]])
    
    C_re3 = []
    for subjID in C_group:
        for subORDER in C_group_match3:
            if subjID == subORDER[0]:
                C_re3.append([subjID,subORDER[1]]) 
                
    C_re3 = [item[1] for item in C_re3]
    
    plt.figure(random.randint(0,80))
    plt.xticks([1,2,3],['Control day 1','Control day 2', 'Control day 3'])
            
    plt.title('Behavioral accuracy day 1, 2 and 3 for control group')
    
    plt.scatter(np.full(11,1),C_re1,c=subsNF_RT_acc,cmap=cmNF,s=60)
    plt.scatter(np.full(11,2),C_re2,c=subsNF_RT_acc,cmap=cmNF,s=60)
    plt.scatter(np.full(11,3),C_re3,c=subsNF_RT_acc,cmap=cmNF,s=60)
    cbar = plt.colorbar()
    cbar.set_label('Real-time decoding accuracy of matched participant')
    plt.ylabel('Behavioral accuracy')
    plt.grid(color='gainsboro',linewidth=0.5)
    
    for i in range(11):
        plt.plot([(np.full(11,1))[i],(np.full(11,2))[i],(np.full(11,3))[i]],\
                  [(C_re1)[i],(C_re2)[i],(C_re3)[i]],c=cmNFget(normNF(subsNF_RT_acc[i])),linewidth=3)#c=cm.hot(i/11))


#%% ########## BLOCK-WISE ANALYSIS #############
behBlock(1,'sen','Sensitivity across blocks, pre-training session','Sensitivity')
behBlock(3,'sen','Sensitivity across blocks, post-training session','Sensitivity')

behBlock(1,'spec','Specificity across blocks, all participants, day 1','Specificity')
behBlock(3,'spec','Specificity across blocks, all participants, day 3','Specificity')

behBlock(1,'acc','Behavioral accuracy across blocks, pre-training session','Behavioral accuracy')
behBlock(3,'acc','Behavioral accuracy across blocks, post-training session','Behavioral accuracy')

behBlock(1,'rt','Response time across blocks, pre-training session','Response time (ms)')
behBlock(3,'rt','Response time across blocks, post-training session','Response time (ms)')

#%% Output responseTimes
responseTimes_all = []

# Extract responseTimes for all subjects
for idx,subjID in enumerate(subjID_all):
    responseTimes = outputResponseTimes(subjID,'1')
    responseTimes_all.append(responseTimes)
        
responseTimes_m = np.nanmean(responseTimes_all,axis=0)

#%% ResponseTimes plot
plotResponseTimes('1')
plotResponseTimes('3')

plotResponseTimes('2')

#%% Make matched pairs
matchedSubjects('sen',r'$\Delta$ Sensitivity - matched participants',RER=True)
matchedSubjects('spec','Specificity improvement - matched participants')
matchedSubjects('acc',r'$\Delta$ Behavioral accuracy - matched participants',RER=True)
matchedSubjects('rt',r'$\Delta$ Response time - matched participants')
# If negative, means that the response time has improved (i.e. shortened) from day 1 to 3. 
# A larger absolute value (the more negative), means better response time improvement

#%% Does NF effect the brain on day 2, during NF? 
# It seems that sensitivity goes down for C, while opposite is true for NF (different t-statistic)

NFimprovement('sen')
NFimprovement('spec')
NFimprovement('acc')
NFimprovement('rt')

#%% Also do this with matched pairs

# Matched day 2-day 1 change
diff_d12_sen, diff_d12_sen_21 = matchedSubjects2('sen',r'$\Delta$ Sensitivity - matched participants',relative=False,NFblocks=False)
diff_d12_spec, diff_d12_spec_21 = matchedSubjects2('spec','Specificity change, day 1 to 2',relative=True) # NF become less specific
diff_d12_acc, diff_d12_acc_21 = matchedSubjects2('acc',r'$\Delta$ Accuracy - matched participants',relative=False,NFblocks=False)

diff_d12_rt, diff_d12_rt_21 = matchedSubjects2('rt',r'$\Delta$ Response time - matched participants')

#%% Create RT decoding acc and LORO acc without subj 11
subs20_RT_acc = np.copy(subsAll_RT_acc)
subs20_RT_acc = np.delete(subs20_RT_acc,2)
subs20_RT_acc = np.delete(subs20_RT_acc,4)

# Only without 11
subs21_RT_acc = np.copy(subsAll_RT_acc)
subs21_RT_acc = np.delete(subs21_RT_acc,2)

subs21_LORO = np.copy(subsAll_LORO)
subs21_LORO = np.delete(subs21_LORO,2)

#%% Is good decoding accuracy correlated with a good day 2 behavioral response?
# Omit 11 

# Sensitivity
sen_all_d2_c = np.copy(sen_all_d2)
sen_all_d2_c = sen_all_d2_c[~np.isnan(sen_all_d2_c)]

# Specificity
spec_all_d2_c = np.copy(spec_all_d2)
spec_all_d2_c = spec_all_d2_c[~np.isnan(spec_all_d2_c)]

# Accuracy
acc_all_d2_c = np.copy(acc_all_d2)
acc_all_d2_c = acc_all_d2_c[~np.isnan(acc_all_d2_c)]

# RT
rt_all_d2_c = np.copy(rt_all_d2)
rt_all_d2_c = rt_all_d2_c[~np.isnan(rt_all_d2_c)]

#%% Day 2 vs. RT decoding acc
sen_all_d2, sen_NF_d2, sen_C_d2 = extractStatsDay(2,'sen')
acc_all_d2, acc_NF_d2, acc_C_d2 = extractStatsDay(2,'acc')
rt_all_d2, rt_NF_d2, rt_C_d2 = extractStatsDay(2,'rt')


sen_NF_d2 = np.delete(sen_NF_d2,2)
acc_NF_d2 = np.delete(acc_NF_d2,2)
rt_NF_d2 = np.delete(rt_NF_d2,2)

# Sensitivity
plt.figure(100)
plt.scatter(subsAll_c,sen_all_d2_c)
plt.ylabel('Sensitivity day 2')
plt.xlabel('Real-time decoding accuracy (NF blocks)')
plt.title('Sensitivity day 2 vs. real-time decoding accuracy, N=21')

np.corrcoef(sen_NF_d2,subsNF_RT_acc)
np.corrcoef(sen_C_d2,subsC_RT_acc)
np.corrcoef(sen_all_d2_c,subsAll_c)

# Accuracy
lm.fit(np.reshape(subsAll_c,[-1,1]),np.reshape(acc_all_d2_c,[-1,1]))

plt.figure(101)
plt.scatter(subsAll_c,acc_all_d2_c)
plt.ylabel('Accuracy day 2')
plt.xlabel('Real-time decoding accuracy (NF blocks)')
plt.title('Accuracy day 2 vs. real-time decoding accuracy, N=21')
plt.plot(np.reshape(subsAll_c,[-1,1]), lm.predict(np.reshape(subsAll_c,[-1,1])),linewidth=0.8,color='black')

stats.linregress(acc_all_d2_c,subsAll_c)
np.corrcoef(acc_NF_d2,subsNF_RT_acc)
np.corrcoef(acc_C_d2,subsC_RT_acc)
np.corrcoef(acc_all_d2_c,subsAll_c)

# Plot RT vs decoding acc
lm.fit(np.reshape(subsAll_c,[-1,1]),np.reshape(rt_all_d2_c,[-1,1]))

plt.figure(102)
plt.scatter(subsAll_c,rt_all_d2_c)
plt.ylabel('Response time day 2')
plt.xlabel('Real-time decoding accuracy, bias corrected')
plt.title('Response time day 2 vs. real-time decoding accuracy, N=21')
plt.plot(np.reshape(subsAll_c,[-1,1]), lm.predict(np.reshape(subsAll_c,[-1,1])),linewidth=0.8,color='black')

stats.linregress(rt_all_d2_c,subsAll_c)
np.corrcoef(rt_NF_d2,subsNF_RT_acc)
np.corrcoef(rt_C_d2,subsC_RT_acc)
np.corrcoef(rt_all_d2_c,subsAll_c)

# Investigate beh 2 divided into STABLE and NF blocks beh measure

#%%
computeDividedCorr('rt')    
    
#%% ################### BEH vs BEH #################
sen_all_d2, sen_NF_d2, sen_C_d2 = extractStatsDay(2,'sen')
acc_all_d2, acc_NF_d2, acc_C_d2 = extractStatsDay(2,'acc')
rt_all_d2, rt_NF_d2, rt_C_d2 = extractStatsDay(2,'rt')

rt_NF_d2 = np.delete(rt_NF_d2,2)
sen_NF_d2 = np.delete(sen_NF_d2,2)
acc_NF_d2 = np.delete(acc_NF_d2,2)


# RT vs sensitivity
lm.fit(np.reshape(rt_all_d2_c,[-1,1]),np.reshape(sen_all_d2_c,[-1,1]))

plt.figure(103)
plt.scatter(rt_all_d2_c,sen_all_d2_c)
plt.ylabel('Sensitivity day 2')
plt.xlabel('Response time (ms)')
plt.title('Response time day 2 vs. sensitivity day 2, N=21')
plt.plot(np.reshape(rt_all_d2_c,[-1,1]), lm.predict(np.reshape(rt_all_d2_c,[-1,1])),linewidth=0.8,color='black')
plt.scatter(rt_all_d2_c,sen_all_d2_c)

np.corrcoef(rt_all_d2_c,sen_all_d2_c)
ma.corrcoef(ma.masked_invalid(rt_NF_d2),ma.masked_invalid(sen_NF_d2))

np.corrcoef(rt_NF_d2,sen_NF_d2)
np.corrcoef(rt_C_d2,sen_C_d2)


# RT vs accuracy
plt.figure(104)
lm.fit(np.reshape(rt_all_d2_c,[-1,1]),np.reshape(acc_all_d2_c,[-1,1]))

plt.scatter(rt_all_d2_c,acc_all_d2_c)
plt.ylabel('Accuracy day 2')
plt.xlabel('Response time (ms)')
plt.title('Response time day 2 vs. accuracy day 2, N=21')
plt.plot(np.reshape(rt_all_d2_c,[-1,1]), lm.predict(np.reshape(rt_all_d2_c,[-1,1])),linewidth=0.8,color='black')
plt.scatter(rt_all_d2_c,acc_all_d2_c)

stats.linregress(rt_all_d2_c,acc_all_d2_c)
np.corrcoef(rt_NF_d2,acc_NF_d2)
np.corrcoef(rt_C_d2,acc_C_d2)

# RT vs specificity
lm.fit(np.reshape(rt_all_d2_c,[-1,1]),np.reshape(spec_all_d2_c,[-1,1]))

plt.scatter(rt_all_d2_c,spec_all_d2_c)
plt.ylabel('Specificity day 2')
plt.xlabel('Response time (ms)')
plt.title('Response time day 2 vs. specificity day 2, N=21')
plt.plot(np.reshape(rt_all_d2_c,[-1,1]), lm.predict(np.reshape(rt_all_d2_c,[-1,1])),linewidth=0.8,color='black')
plt.scatter(rt_all_d2_c,spec_all_d2_c)

stats.linregress(rt_all_d2_c,spec_all_d2_c)

#%% BEH vs BEH day 1 
np.corrcoef(rt_all_d1,sen_all_d1)

np.corrcoef(rt_NF_d1,sen_NF_d1)
np.corrcoef(rt_C_d1,sen_C_d1)

plt.figure(106)
lm.fit(np.reshape(rt_all_d1,[-1,1]),np.reshape(sen_all_d1,[-1,1]))

plt.scatter(rt_all_d1,sen_all_d1)
plt.ylabel('sen day 1')
plt.xlabel('Response time (ms)')
plt.title('Response time day 1 vs. accuracy day 1, N=22')
plt.plot(np.reshape(rt_all_d1,[-1,1]), lm.predict(np.reshape(rt_all_d1,[-1,1])),linewidth=0.8,color='black')

# Acc
np.corrcoef(rt_all_d1,acc_all_d1)
np.corrcoef(rt_NF_d1,acc_NF_d1)
np.corrcoef(rt_C_d1,acc_C_d1)

#%% Day 3
sen_all_d3, sen_NF_d3, sen_C_d3 = extractStatsDay(3,'sen')
acc_all_d3, acc_NF_d3, acc_C_d3 = extractStatsDay(3,'acc')
rt_all_d3, rt_NF_d3, rt_C_d3 = extractStatsDay(3,'rt')

# sen
np.corrcoef(rt_all_d3,sen_all_d3)
np.corrcoef(rt_NF_d3,sen_NF_d3)
np.corrcoef(rt_C_d3,sen_C_d3)

plt.scatter(rt_C_d3,sen_C_d3)

# acc
np.corrcoef(rt_all_d3,acc_all_d3)
np.corrcoef(rt_NF_d3,acc_NF_d3)
np.corrcoef(rt_C_d3,acc_C_d3)
#%% Is good decoding accuracy STABLE blocks correlated with a good day 1 behavioral response?
sen_all_d1, sen_NF_d1, sen_C_d1 = extractStatsDay(1,'sen')
spec_all_d1, spec_NF_d1, spec_C_d1 = extractStatsDay(1,'spec')
acc_all_d1, acc_NF_d1, acc_C_d1 = extractStatsDay(1,'acc')
rt_all_d1, rt_NF_d1, rt_C_d1 = extractStatsDay(1,'rt')

# Sensitivity/accuracy vs LOBO
plt.figure(105)

lm.fit(np.reshape(subsAll_LOBO,[-1,1]),np.reshape(np.array(acc_all_d1),[-1,1]))
plt.scatter(subsAll_LOBO,acc_all_d1,color='brown',label='Accuracy')
plt.ylabel('Accuracy or sensitivity day 1')
plt.xlabel('Leave one block out offline decoding accuracy')
plt.title('Accuracy/sensitivity day 1 vs. offline decoding accuracy, N=22')
plt.plot(np.reshape(subsAll_LOBO,[-1,1]), lm.predict(np.reshape(subsAll_LOBO,[-1,1])),linewidth=0.8,color='brown')

lm.fit(np.reshape(subsAll_LOBO,[-1,1]),np.reshape(np.array(sen_all_d1),[-1,1]))
plt.scatter(subsAll_LOBO,sen_all_d1,color='tomato',label='Sensitivity')
plt.plot(np.reshape(subsAll_LOBO,[-1,1]), lm.predict(np.reshape(subsAll_LOBO,[-1,1])),linewidth=0.8,color='tomato')
plt.legend()

np.corrcoef(subsAll_LOBO,sen_all_d1)
np.corrcoef(subsAll_LOBO,acc_all_d1)

# Sensitivity/accuracy vs LORO
plt.figure(106)

lm.fit(np.reshape(subsAll_LORO,[-1,1]),np.reshape(np.array(acc_all_d1),[-1,1]))
plt.scatter(subsAll_LORO,acc_all_d1,color='brown',label='Accuracy')
plt.ylabel('Accuracy or sensitivity day 1')
plt.xlabel('Leave one run out offline decoding accuracy')
plt.title('Accuracy/sensitivity day 1 vs. offline decoding accuracy, N=22')
plt.plot(np.reshape(subsAll_LORO,[-1,1]), lm.predict(np.reshape(subsAll_LORO,[-1,1])),linewidth=0.8,color='brown')

lm.fit(np.reshape(subsAll_LORO,[-1,1]),np.reshape(np.array(sen_all_d1),[-1,1]))
plt.scatter(subsAll_LORO,sen_all_d1,color='tomato',label='Sensitivity')
plt.plot(np.reshape(subsAll_LORO,[-1,1]), lm.predict(np.reshape(subsAll_LORO,[-1,1])),linewidth=0.8,color='tomato')
plt.legend()

np.corrcoef(subsAll_LORO,sen_all_d1)
np.corrcoef(subsAll_LORO,acc_all_d1)

np.corrcoef(subsAll_LORO,rt_all_d1)


#%% Divided into groups
# Create behavioral day 1 vs offline decoding accuracy plots
behVSdecode('sen','Sensitivity',LOBO=False,masking=False)
behVSdecode('acc','Accuracy',LOBO=False,masking=False)
behVSdecode('spec','Specificity',LOBO=False,masking=False)
behVSdecode('spec','Specificity',LOBO=False,masking='15')
behVSdecode('rt','Response time',LOBO=False,masking=False)

behVSdecode('sen','Sensitivity',LOBO=True,masking=False)
behVSdecode('spec','Specificity',LOBO=True,masking=False)
behVSdecode('acc','Accuracy',LOBO=True,masking=False)
behVSdecode('rt','Response time',LOBO=True,masking=False)

#%% Create behavioral day 2 vs offline decoding accuracy plots
behDay2VSdecode('sen','Sensitivity',LOBO=False)
behDay2VSdecode('acc','Behavioral accuracy',LOBO=False)
behDay2VSdecode('rt','Response time (ms)',LOBO=False)

behDay2VSdecode('sen','Sensitivity',LOBO=True)
behDay2VSdecode('acc','Accuracy',LOBO=True)
behDay2VSdecode('rt','Response time',LOBO=True)

#%% Somewhere here: check whether good decoding acc corresponds with good day 1 to 3 improvement

# I.e. if you had good decoding accuracy, how much did you gain?
# Extract values for day 1 and 3
all_d1, NF_d1, C_d1 = extractStatsDay(1,'sen')
all_d3, NF_d3, C_d3 = extractStatsDay(3,'sen')

diff = (np.asarray(all_d3) - np.asarray(all_d1))
lm.fit(np.reshape(subsAll_RT_acc,[-1,1]),np.reshape(diff,[-1,1]))

plt.figure(109)
plt.scatter(subsAll_RT_acc,diff)
plt.plot(np.reshape(subsAll_RT_acc,[-1,1]), lm.predict(np.reshape(subsAll_RT_acc,[-1,1])),linewidth=0.8)
plt.ylabel('Change in sensitivity from day 1 to 3\n Positive values indicate improvement')

np.corrcoef(subsAll_RT_acc,diff)

# Response time
all_d1, NF_d1, C_d1 = extractStatsDay(1,'rt')
all_d3, NF_d3, C_d3 = extractStatsDay(3,'rt')
diff = (np.asarray(all_d1) - np.asarray(all_d3))

lm.fit(np.reshape(subsAll_RT_acc,[-1,1]),np.reshape(diff,[-1,1]))

plt.figure(110)
plt.scatter(subsAll_RT_acc,diff)
plt.plot(np.reshape(subsAll_RT_acc,[-1,1]), lm.predict(np.reshape(subsAll_RT_acc,[-1,1])),linewidth=0.8)
plt.ylabel('Change in response time from day 1 to 3\n Positive values indicate improvement (i.e. quicker response)')

np.corrcoef(subsAll_RT_acc,diff)

#%% Make improvement plots     
improvStimuli('sen',actual_stim=False)  
# improvStimuli('sen',actual_stim=True)  
improvStimuli('acc',actual_stim=False)  
improvStimuli('rt',actual_stim=False)  

#%% LORO vs delta beh
improvStimuli('sen',actual_stim=True,LORO=True)
improvStimuli('acc',actual_stim=True,LORO=True)
improvStimuli('rt',actual_stim=True,LORO=True)
