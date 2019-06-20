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
from beh_FUNCv3 import *

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
(_, caps, _) = plt.errorbar(np.arange(0,7),CR13m,yerr=CR13_yerr,color='green', capsize=8)
for cap in caps:
    cap.set_markeredgewidth(1)
(_, caps, _) = plt.errorbar(np.arange(0,7),FR13m,yerr=FR13_yerr,color='red', capsize=8)
for cap in caps:
    cap.set_markeredgewidth(1)
    
plt.plot(np.arange(0,7),CR13m,color='green', label='CR')
plt.plot(np.arange(0,7),FR13m,color='red', label='FA')
# plt.title('Response times surrounding lure trials') # 
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('Response time (s)')
plt.grid(color='gainsboro',linewidth=0.5)
plt.legend(loc='lower left')

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
plt.ylabel('RT (s)')

#%% # Plots comparing day 1 and 3
pl, t_fb, t_c, acc_all_d1 = make4Bars('er','Error rate: Pre- to post-training','Error rate')
pl, t_fb, t_c, acc_all_d1 = make4Bars('rer','Relative error rate reduction: Pre- to post-training','Relative error rate reduction')
pl, t_fb, t_c, acc_all_d1 = make4Bars('a','A\': Pre- to post-training','A\'')
pl, t_fb, t_c, all_d1 = make4Bars('rt','Response time: Pre- to post-training','Response time (s)')

#%% Make matched pairs
matchedSubjects('er',r'$\Delta$ Error rate - matched participants')
matchedSubjects('rer',r'$\Delta$ Relative error rate reduction - matched participants')
matchedSubjects('a',r"$\Delta$ A' - matched participants")
matchedSubjects('rt',r'$\Delta$ Response time - matched participants')

#%% Extract H and FA
H_all1, H_NF1, H_C1 = extractHandFA(1,'H')
H_all3, H_NF3, H_C3 = extractHandFA(3,'H')

FA_all1, FA_NF1, FA_C1 = extractHandFA(1,'FA')
FA_all3, FA_NF3, FA_C3 = extractHandFA(3,'FA')

makeHandFA(1)

#%% Compare day 4 and 5
pl, t_fb, t_c, all_d1 = make4Bars('sen','Sensitivity','Response sensitivity',part2=True)
pl, t_fb, t_c, all_d1 = make4Bars('a','Accuracy','Response accuracy',part2=True)
pl, t_fb, t_c, all_d1 = make4Bars('rt','Response time','Response time ',part2=True)
pl, t_fb, t_c, all_d1 = make4Bars('arer','Response time','Response time ',part2=True)

#%%  Investigate day 2, using make2Bars and StatsDayAll (NOT block-wise)
pl, t, sen_all_d2 = make2Bars('er','Sensitivity day 2','Sensitivity')#,RER=True) 
pl, t, acc_all_d2 = make2Bars('rer','Accuracy day 2','Accuracy') 
pl, t, fpr_all_d2 = make2Bars('a','FPR day 2','FPR') 
pl, t, rt_all_d2 = make2Bars('rt','RT day 2','RT') 
pl, t, nkeypress_all_d2 = make2Bars('nkeypress','Number of total keypresses','Number keypresses') 

#%% ########## BLOCK-WISE ANALYSIS #############
behBlock(1,'a',"A' across blocks, pre-training session (day 1)","A'")
behBlock(3,'a',"A' across blocks, post-training session (day 3)","A'")

behBlock(1,'er','Behavioral error rate across blocks, pre-training session','Behavioral error rate')
behBlock(3,'er','Behavioral error rate across blocks, post-training session','Behavioral error rate')

behBlock(1,'rt','Response time across blocks, pre-training session','Response time (s)')
behBlock(3,'rt','Response time across blocks, post-training session','Response time (s)')

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

#%% Does NF effect the brain on day 2, during NF? 
NFimprovement('sen')
NFimprovement('spec')
NFimprovement('acc')
NFimprovement('acc',RER=True)
NFimprovement('rt')

#%% Matched day 2 - day 1 change
diff_d12_er, diff_d12_er_21 = matchedSubjects2('er',r'$\Delta$ Error rate - matched participants day 1 2')
diff_d12_rer, diff_d12_rer_21 = matchedSubjects2('rer',r'$\Delta$ Relative error rate reduction - matched participants day 1 2')
diff_d12_a, diff_d12_a_21 = matchedSubjects2('a',r"$\Delta$ A' - matched participants day 1 2")
diff_d12_a, diff_d12_arer_21 = matchedSubjects2('arer',r"$\Delta$ Relative A' - matched participants day 1 2 ")


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

#%% Day 2 vs. RT decoding acc
acc_all_d2, acc_NF_d2, acc_C_d2 = extractStatsDay(2,'acc')
er_all_d2, er_NF_d2, er_C_d2 = extractStatsDay(2,'er')
rer_all_d2, rer_NF_d2, rer_C_d2 = extractStatsDay(2,'rer')
a_all_d2, a_NF_d2, a_C_d2 = extractStatsDay(2,'a')
rt_all_d2, rt_NF_d2, rt_C_d2 = extractStatsDay(2,'rt')

acc_NF_d2 = np.delete(acc_NF_d2,2)
er_NF_d2 = np.delete(er_NF_d2,2)
rer_NF_d2 = np.delete(rer_NF_d2,2)
a_NF_d2 = np.delete(a_NF_d2,2)
rt_NF_d2 = np.delete(rt_NF_d2,2)

subsNF_RT_acc = np.delete(subsNF_RT_acc,2)

# Sensitivity
plt.figure(100)
plt.scatter(subsAll_c,sen_all_d2_c)
plt.ylabel('Sensitivity day 2')
plt.xlabel('Real-time decoding accuracy (NF blocks)')
plt.title('Sensitivity day 2 vs. real-time decoding accuracy, N=21')

np.corrcoef(rer_NF_d2,subsNF_RT_acc)
np.corrcoef(a_C_d2,subsC_RT_acc)
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

#%% RT accuracy vs all day 2, NF day 2 or stable day 2
computeDividedCorr('a')    
    
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
er_all_d1, er_NF_d1, er_C_d1 = extractStatsDay(1,'er')
rer_all_d1, rer_NF_d1, rer_C_d1 = extractStatsDay(1,'rer')
a_all_d1, a_NF_d1, a_C_d1 = extractStatsDay(1,'a')
rt_all_d1, rt_NF_d1, rt_C_d1 = extractStatsDay(1,'rt')

# RERR Error rate
np.corrcoef(rt_all_d1,rer_all_d1)

np.corrcoef(rt_NF_d1,rer_NF_d1)
np.corrcoef(rt_C_d1,rer_C_d1)

plt.figure(106)
lm.fit(np.reshape(rt_all_d1,[-1,1]),np.reshape(sen_all_d1,[-1,1]))

plt.scatter(rt_all_d1,sen_all_d1)
plt.ylabel('sen day 1')
plt.xlabel('Response time (ms)')
plt.title('Response time day 1 vs. accuracy day 1, N=22')
plt.plot(np.reshape(rt_all_d1,[-1,1]), lm.predict(np.reshape(rt_all_d1,[-1,1])),linewidth=0.8,color='black')

# A
np.corrcoef(rt_all_d1,a_all_d1)
np.corrcoef(rt_NF_d1,a_NF_d1)
np.corrcoef(rt_C_d1,a_C_d1)

#%% Day 3
rer_all_d3, rer_NF_d3, rer_C_d3 = extractStatsDay(3,'rer')
a_all_d3, a_NF_d3, a_C_d3 = extractStatsDay(3,'a')
rt_all_d3, rt_NF_d3, rt_C_d3 = extractStatsDay(3,'rt')

# rer
np.corrcoef(rt_all_d3,rer_all_d3)
np.corrcoef(rt_NF_d3,rer_NF_d3)
np.corrcoef(rt_C_d3,rer_C_d3)

plt.scatter(rt_C_d3,sen_C_d3)

# acc
np.corrcoef(rt_all_d3,a_all_d3)
np.corrcoef(rt_NF_d3,a_NF_d3)
np.corrcoef(rt_C_d3,a_C_d3)
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
behVSdecode('er','Error rate',LOBO=False)
behVSdecode('rer','Error rate',LOBO=False)
behVSdecode('a',"A'",LOBO=False)

behVSdecode('rt','Response time',LOBO=False)

#%% Create behavioral day 2 vs offline decoding accuracy plots
behDay2VSdecode('a',"A'",LOBO=False)
behDay2VSdecode('rer','RERR',LOBO=False)
behDay2VSdecode('rt','Response time (s)',LOBO=False)


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
improvStimuli('a',actual_stim=False)  
improvStimuli('rer',actual_stim=False)  
improvStimuli('rt',actual_stim=False)  

improvStimuli('rer',actual_stim=False)  
improvStimuli('acc',actual_stim=False)  


#%% LORO vs delta beh
improvStimuli('a',actual_stim=True,LORO=True)
improvStimuli('er',actual_stim=True,LORO=True)
improvStimuli('rer',actual_stim=True,LORO=True)

improvStimuli('rt',actual_stim=True,LORO=True)


#%% Three day plots
threeDay('a')












