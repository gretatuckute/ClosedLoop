# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:12:38 2019

First part is the V2 analysis (not block-wise)

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

#%%
with open('Beh_subjID_07.pkl', "rb") as fin:
    sub07 = (pickle.load(fin))[0]   
with open('Beh_subjID_08.pkl', "rb") as fin:
    sub08 = (pickle.load(fin))[0]   
with open('Beh_subjID_11.pkl', "rb") as fin:
    sub11 = (pickle.load(fin))[0]
with open('Beh_subjID_13.pkl', "rb") as fin:
    sub13 = (pickle.load(fin))[0]
with open('Beh_subjID_14.pkl', "rb") as fin:
    sub14 = (pickle.load(fin))[0]
with open('Beh_subjID_15.pkl', "rb") as fin:
    sub15 = (pickle.load(fin))[0]
with open('Beh_subjID_16.pkl', "rb") as fin:
    sub16 = (pickle.load(fin))[0]    
with open('Beh_subjID_17.pkl', "rb") as fin:
    sub17 = (pickle.load(fin))[0]
with open('Beh_subjID_18.pkl', "rb") as fin:
    sub18 = (pickle.load(fin))[0]
with open('Beh_subjID_19.pkl', "rb") as fin:
    sub19 = (pickle.load(fin))[0]
with open('Beh_subjID_21.pkl', "rb") as fin:
    sub21 = (pickle.load(fin))[0]
with open('Beh_subjID_22.pkl', "rb") as fin:
    sub22 = (pickle.load(fin))[0]
with open('Beh_subjID_23.pkl', "rb") as fin:
    sub23 = (pickle.load(fin))[0]
with open('Beh_subjID_25.pkl', "rb") as fin:
    sub25 = (pickle.load(fin))[0]
with open('Beh_subjID_27.pkl', "rb") as fin:
    sub27 = (pickle.load(fin))[0]
with open('Beh_subjID_30.pkl', "rb") as fin:
    sub30 = (pickle.load(fin))[0]
with open('Beh_subjID_31.pkl', "rb") as fin:
    sub31 = (pickle.load(fin))[0]
with open('Beh_subjID_32.pkl', "rb") as fin:
    sub32 = (pickle.load(fin))[0]
with open('Beh_subjID_33.pkl', "rb") as fin:
    sub33 = (pickle.load(fin))[0]
with open('Beh_subjID_34.pkl', "rb") as fin:
    sub34 = (pickle.load(fin))[0]
    
#%% Measures

def computeStats(subjID,expDay):
        
    with open('Beh_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]

    TP = sub['no_Inhibitions_nonlure_day_'+expDay]
    FN = sub['inhibitions_nonlure_day_'+expDay]
    
    TN = sub['inhibitions_lure_day_'+expDay]
    FP = sub['no_Inhibitions_lure_day_'+expDay]
        
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    
    false_positive_rate = FP/(FP+TN)
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
        
    return sensitivity,specificity,false_positive_rate,accuracy 
         
#%% Analyze 4 days
dayLst = ['1','3','4','5']

f07=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f07[count,:]=computeStats('07',day)
f08=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f08[count,:]=computeStats('08',day)
f11=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f11[count,:]=computeStats('11',day)
f13=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f13[count,:]=computeStats('13',day)
f14=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f14[count,:]=computeStats('14',day)
f15=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f15[count,:]=computeStats('15',day)
f16=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f16[count,:]=computeStats('16',day)
f17=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f17[count,:]=computeStats('17',day)
f18=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f18[count,:]=computeStats('18',day)
f19=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f19[count,:]=computeStats('19',day)
f21=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f21[count,:]=computeStats('21',day)
f22=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f22[count,:]=computeStats('22',day)
f23=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f23[count,:]=computeStats('23',day)
f24=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f24[count,:]=computeStats('24',day)
f25=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f25[count,:]=computeStats('25',day)
f26=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f26[count,:]=computeStats('26',day)
f27=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f27[count,:]=computeStats('27',day)
f30=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f30[count,:]=computeStats('30',day)
f31=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f31[count,:]=computeStats('31',day)
f32=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f32[count,:]=computeStats('32',day)
f33=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f33[count,:]=computeStats('33',day)
f34=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f34[count,:]=computeStats('34',day)



#%%
sub_fb=[f07,f08,f11,f13,f14,f16,f19,f22,f26,f27,f30]
sub_c=[f15,f17,f18,f21,f23,f24,f25,f31,f32,f33,f34]

#%%
sen_feedback=np.asarray([f[:,0] for f in sub_fb])
sen_control=np.asarray([f[:,0] for f in sub_c])

spec_feedback=np.asarray([f[:,1] for f in sub_fb])
spec_control=np.asarray([f[:,1] for f in sub_c])

fpr_feedback=np.asarray([f[:,2] for f in sub_fb])
fpr_control=np.asarray([f[:,2] for f in sub_c])

acc_feedback=np.asarray([f[:,3] for f in sub_fb])
acc_control=np.asarray([f[:,3] for f in sub_c])

#%% Sensitivity
plt.figure(1)
plt.bar(1,np.mean(sen_feedback[:,0]),color=(0,0,0,0),edgecolor='tomato',yerr=np.std(sen_feedback[:,0]))
plt.bar(2,np.mean(sen_feedback[:,1]),color=(0,0,0,0),edgecolor='brown',yerr=np.std(sen_feedback[:,1]))
plt.bar(3,np.mean(sen_control[:,0]),color=(0,0,0,0),edgecolor='dodgerblue',yerr=np.std(sen_control[:,0]))
plt.bar(4,np.mean(sen_control[:,1]),color=(0,0,0,0),edgecolor='navy',yerr=np.std(sen_control[:,1]))

plt.scatter(np.full(11,1),sen_feedback[:,0],color='tomato')
plt.scatter(np.full(11,2),sen_feedback[:,1],color='brown')
plt.scatter(np.full(11,3),sen_control[:,0],color='dodgerblue')
plt.scatter(np.full(11,4),sen_control[:,1],color='navy')

plt.ylabel('Response sensitivity')
plt.ylim([0.9,1])
plt.xticks([1,2,3,4],['NF day1','NF day3', 'Control day1', 'Control day3'])
plt.title('Sensitivity')

# Connect lines
plt.figure(10)
plt.ylabel('Response sensitivity')
plt.ylim([0.9,1])
plt.xticks([1,2,3,4],['NF day1','NF day3', 'Control day1', 'Control day3'])
plt.title('Sensitivity')

plt.bar(1,np.mean(sen_feedback[:,0]),color=(0,0,0,0),edgecolor='tomato')
plt.bar(2,np.mean(sen_feedback[:,1]),color=(0,0,0,0),edgecolor='brown')
plt.bar(3,np.mean(sen_control[:,0]),color=(0,0,0,0),edgecolor='dodgerblue')
plt.bar(4,np.mean(sen_control[:,1]),color=(0,0,0,0),edgecolor='navy')

plt.scatter(np.full(11,1),sen_feedback[:,0],color='tomato')
plt.scatter(np.full(11,2),sen_feedback[:,1],color='brown')
plt.scatter(np.full(11,3),sen_control[:,0],color='dodgerblue')
plt.scatter(np.full(11,4),sen_control[:,1],color='navy')

for i in range(11):
    plt.plot([(np.full(11,1))[i],(np.full(11,2))[i]], [(sen_feedback[:,0])[i],(sen_feedback[:,1])[i]],color='gray')
    plt.plot([(np.full(11,3))[i],(np.full(11,4))[i]], [(sen_control[:,0])[i],(sen_control[:,1])[i]],color='gray')

# Day 4 and 5
#plt.figure(3)
#plt.bar(1,np.mean(sen_feedback[:,2]))
#plt.bar(2,np.mean(sen_feedback[:,3]))
#plt.bar(3,np.mean(sen_control[:,2]))
#plt.bar(4,np.mean(sen_control[:,3]))

#%% Specificity
plt.figure(2)
plt.bar(1,np.mean(spec_feedback[:,0]),color=(0,0,0,0),edgecolor='tomato',yerr=np.std(spec_feedback[:,0]))
plt.bar(2,np.mean(spec_feedback[:,1]),color=(0,0,0,0),edgecolor='brown',yerr=np.std(spec_feedback[:,1]))
plt.bar(3,np.mean(spec_control[:,0]),color=(0,0,0,0),edgecolor='dodgerblue',yerr=np.std(spec_control[:,0]))
plt.bar(4,np.mean(spec_control[:,1]),color=(0,0,0,0),edgecolor='navy',yerr=np.std(spec_control[:,1]))

plt.scatter(np.full(11,1),spec_feedback[:,0],color='tomato')
plt.scatter(np.full(11,2),spec_feedback[:,1],color='brown')
plt.scatter(np.full(11,3),spec_control[:,0],color='dodgerblue')
plt.scatter(np.full(11,4),spec_control[:,1],color='navy')

plt.ylabel('Response specificity')
plt.ylim([0.4,1])
plt.xticks([1,2,3,4],['NF day1','NF day3', 'Control day1', 'Control day3'])
plt.title('Specificity')

# Connect lines
plt.figure(12)
plt.bar(1,np.mean(spec_feedback[:,0]),color=(0,0,0,0),edgecolor='tomato')
plt.bar(2,np.mean(spec_feedback[:,1]),color=(0,0,0,0),edgecolor='brown')
plt.bar(3,np.mean(spec_control[:,0]),color=(0,0,0,0),edgecolor='dodgerblue')
plt.bar(4,np.mean(spec_control[:,1]),color=(0,0,0,0),edgecolor='navy')

plt.scatter(np.full(11,1),spec_feedback[:,0],color='tomato')
plt.scatter(np.full(11,2),spec_feedback[:,1],color='brown')
plt.scatter(np.full(11,3),spec_control[:,0],color='dodgerblue')
plt.scatter(np.full(11,4),spec_control[:,1],color='navy')

plt.ylabel('Response specificity')
plt.ylim([0.4,1])
plt.xticks([1,2,3,4],['NF day1','NF day3', 'Control day1', 'Control day3'])
plt.title('Specificity')

for i in range(11):
    plt.plot([(np.full(11,1))[i],(np.full(11,2))[i]], [(spec_feedback[:,0])[i],(spec_feedback[:,1])[i]],color='gray')
    plt.plot([(np.full(11,3))[i],(np.full(11,4))[i]], [(spec_control[:,0])[i],(spec_control[:,1])[i]],color='gray')


#%% FPR
plt.figure(3)
plt.bar(1,np.mean(fpr_feedback[:,0]),color=(0,0,0,0),edgecolor='tomato',yerr=np.std(fpr_feedback[:,0]))
plt.bar(2,np.mean(fpr_feedback[:,1]),color=(0,0,0,0),edgecolor='brown',yerr=np.std(fpr_feedback[:,1]))
plt.bar(3,np.mean(fpr_control[:,0]),color=(0,0,0,0),edgecolor='dodgerblue',yerr=np.std(fpr_control[:,0]))
plt.bar(4,np.mean(fpr_control[:,1]),color=(0,0,0,0),edgecolor='navy',yerr=np.std(fpr_control[:,1]))

plt.scatter(np.full(11,1),fpr_feedback[:,0],color='tomato')
plt.scatter(np.full(11,2),fpr_feedback[:,1],color='brown')
plt.scatter(np.full(11,3),fpr_control[:,0],color='dodgerblue')
plt.scatter(np.full(11,4),fpr_control[:,1],color='navy')

plt.ylabel('False positive rate')
plt.ylim([0,0.6])
plt.xticks([1,2,3,4],['NF day1','NF day3', 'Control day1', 'Control day3'])
plt.title('FPR')

# Connect
plt.figure(13)
plt.bar(1,np.mean(fpr_feedback[:,0]),color=(0,0,0,0),edgecolor='tomato')
plt.bar(2,np.mean(fpr_feedback[:,1]),color=(0,0,0,0),edgecolor='brown')
plt.bar(3,np.mean(fpr_control[:,0]),color=(0,0,0,0),edgecolor='dodgerblue')
plt.bar(4,np.mean(fpr_control[:,1]),color=(0,0,0,0),edgecolor='navy')

plt.scatter(np.full(11,1),fpr_feedback[:,0],color='tomato')
plt.scatter(np.full(11,2),fpr_feedback[:,1],color='brown')
plt.scatter(np.full(11,3),fpr_control[:,0],color='dodgerblue')
plt.scatter(np.full(11,4),fpr_control[:,1],color='navy')

plt.ylabel('False positive rate')
plt.ylim([0,0.6])
plt.xticks([1,2,3,4],['NF day1','NF day3', 'Control day1', 'Control day3'])
plt.title('FPR')

for i in range(11):
    plt.plot([(np.full(11,1))[i],(np.full(11,2))[i]], [(fpr_feedback[:,0])[i],(fpr_feedback[:,1])[i]],color='gray')
    plt.plot([(np.full(11,3))[i],(np.full(11,4))[i]], [(fpr_control[:,0])[i],(fpr_control[:,1])[i]],color='gray')

#%% Accuracy
plt.figure(4)
plt.bar(1,np.mean(acc_feedback[:,0]),color=(0,0,0,0),edgecolor='tomato',yerr=np.std(acc_feedback[:,0]))
plt.bar(2,np.mean(acc_feedback[:,1]),color=(0,0,0,0),edgecolor='brown',yerr=np.std(acc_feedback[:,1]))
plt.bar(3,np.mean(acc_control[:,0]),color=(0,0,0,0),edgecolor='dodgerblue',yerr=np.std(acc_control[:,0]))
plt.bar(4,np.mean(acc_control[:,1]),color=(0,0,0,0),edgecolor='navy',yerr=np.std(acc_control[:,1]))

plt.scatter(np.full(11,1),acc_feedback[:,0],color='tomato')
plt.scatter(np.full(11,2),acc_feedback[:,1],color='brown')
plt.scatter(np.full(11,3),acc_control[:,0],color='dodgerblue')
plt.scatter(np.full(11,4),acc_control[:,1],color='navy')

plt.ylabel('Accuracy')
plt.ylim([0.9,1])
plt.xticks([1,2,3,4],['NF day1','NF day3', 'Control day1', 'Control day3'])
plt.title('Accuracy')

# Connect
plt.figure(14)
plt.bar(1,np.mean(acc_feedback[:,0]),color=(0,0,0,0),edgecolor='tomato')
plt.bar(2,np.mean(acc_feedback[:,1]),color=(0,0,0,0),edgecolor='brown')
plt.bar(3,np.mean(acc_control[:,0]),color=(0,0,0,0),edgecolor='dodgerblue')
plt.bar(4,np.mean(acc_control[:,1]),color=(0,0,0,0),edgecolor='navy')

plt.scatter(np.full(11,1),acc_feedback[:,0],color='tomato')
plt.scatter(np.full(11,2),acc_feedback[:,1],color='brown')
plt.scatter(np.full(11,3),acc_control[:,0],color='dodgerblue')
plt.scatter(np.full(11,4),acc_control[:,1],color='navy')

plt.ylabel('Accuracy')
plt.ylim([0.9,1])
plt.xticks([1,2,3,4],['NF day1','NF day3', 'Control day1', 'Control day3'])
plt.title('Accuracy')

for i in range(11):
    plt.plot([(np.full(11,1))[i],(np.full(11,2))[i]], [(acc_feedback[:,0])[i],(acc_feedback[:,1])[i]],color='gray')
    plt.plot([(np.full(11,3))[i],(np.full(11,4))[i]], [(acc_control[:,0])[i],(acc_control[:,1])[i]],color='gray')


#%% T tests

# Sensitivity
diff_fb=np.mean(sen_feedback[:,1]-sen_feedback[:,0])
diff_c=np.mean(sen_control[:,1]-sen_control[:,0])

diff_fb_p=stats.ttest_rel(sen_feedback[:,0],sen_feedback[:,1])
diff_c_p=stats.ttest_rel(sen_control[:,0],sen_control[:,1])

# Acc
#diff_fb_acc=np.mean(acc_feedback[:,1]-acc_feedback[:,0])
#diff_c_acc=np.mean(acc_control[:,1]-acc_control[:,0])

p_acc_fb=stats.ttest_rel(acc_feedback[:,0],acc_feedback[:,1])
p_acc_c=stats.ttest_rel(acc_control[:,0],acc_control[:,1])


#%% Plots for RT surrounding lures
def extractRTs(subjID):
        
    with open('Beh_subjID_' + subjID + '.pkl', "rb") as fin:
        sub = (pickle.load(fin))[0]
    
    c1 = sub['surrounding_CR_Lst_day_1']
    c3 = sub['surrounding_CR_Lst_day_3']
    
    f1 = sub['surrounding_FR_Lst_day_1']
    f3 = sub['surrounding_FR_Lst_day_3']
    
    # Computing day 1 and 3 together
    c13m = [statistics.mean(k) for k in zip(c1, c3)]
    f13m = [statistics.mean(k) for k in zip(f1, f3)]
    
    # Computing day 1 and 3 separately
    
    return np.asarray(c1), np.asarray(c3), np.asarray(f1), np.asarray(f3), np.asarray(c13m), np.asarray(f13m)


all_RTs = np.zeros((22,6),dtype=np.ndarray)

sub_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

for idx, subj in enumerate(sub_all):
    all_RTs[idx,:] = extractRTs(subj)

# Mean over c13m and f13m for all subjects
all_c1 = all_RTs[:,0]
all_c3 = all_RTs[:,1]
all_f1 = all_RTs[:,2]
all_f3 = all_RTs[:,3]

all_c13 = all_RTs[:,4]
all_f13 = all_RTs[:,5]
# Make into np arrays?
all_c1m = np.mean(all_c1)
all_c3m = np.mean(all_c3)
all_f1m = np.mean(all_f1)
all_f3m = np.mean(all_f3)
all_c13m = np.mean(all_c13)
all_f13m = np.mean(all_f13)

# Standard error of the mean for errorbars
c1_yerr=(np.std(all_c1))/np.sqrt(22)
c3_yerr=(np.std(all_c3))/np.sqrt(22)
f1_yerr=(np.std(all_f1))/np.sqrt(22)
f3_yerr=(np.std(all_f3))/np.sqrt(22)
c13_yerr=(np.std(all_c13))/np.sqrt(22)
f13_yerr=(np.std(all_f13))/np.sqrt(22)


#%% Plots
ticks = ['-3','-2','-1','lure','1','2','3']

# Both days
#plt.plot(all_c13m,color='green',)
#plt.plot(all_f13m,color='red')

plt.figure(5)
plt.errorbar(np.arange(0,7),all_c13m,yerr=c13_yerr,color='green')
plt.errorbar(np.arange(0,7),all_f13m,yerr=f13_yerr,color='red')
plt.title('Day 1 and 3 - all participants')
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('RT (ms)')

# Day 1
plt.figure(6)
#plt.plot(all_c1m,color='green')
#plt.plot(all_f1m,color='red')
plt.errorbar(np.arange(0,7),all_c1m,yerr=c1_yerr,color='green')
plt.errorbar(np.arange(0,7),all_f1m,yerr=f1_yerr,color='red')
plt.title('Day 1, all participants')
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('RT (ms)')

# Day 3
plt.figure(7)
#plt.plot(all_c3m,color='green')
#plt.plot(all_f3m,color='red')
plt.errorbar(np.arange(0,7),all_c3m,yerr=c3_yerr,color='green')
plt.errorbar(np.arange(0,7),all_f3m,yerr=f3_yerr,color='red')
plt.title('Day 3, all participants')
plt.xticks(np.arange(0,7,1),ticks)
plt.xlabel('Trials from lure')
plt.ylabel('RT (ms)')


#%% ################## V3 analysis ####################
def computeStatsBlocks(subjID):
    
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

#%%
# Test for subj 11
#statsDay11, statsBlock11, statsBlock_day211 = computeStatsBlocks('11')
#statsDay, statsBlock, statsBlock_day2 = computeStatsBlocks('07')

sub_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

statsAll = {}

for idx,subjID in enumerate(sub_all):
    statsDay, statsBlock, statsBlock_day2 = computeStatsBlocks(subjID)
    
    statsAll[subjID] = statsBlock

# Mean response time per block, day 2
plt.plot(statsBlock_day2[:,3])

# Mean response time per block, day 1 and 3
plt.plot(statsBlock[0,:,3])
plt.plot(statsBlock[1,:,3])

# Sensitivity across blocks
plt.plot(statsBlock[0,:,0])
plt.plot(statsBlock[1,:,0])

# Accuracy across blocks
plt.plot(statsBlock[0,:,2])
plt.plot(statsBlock[1,:,2])


#%% 
subjIDs_NF = ['07','08','11','13','14','16','19','22','26','27','30']
subjIDs_C = ['15','17','18','21','23','24','25','31','32','33','34']
            
def extractBehVal(day,wanted_measure):
    '''
    Day is 
    1=0
    3=1
    
    for statsBlock
    
    and wanted_measure is 
    0 = sens
    1 = spec
    2 = acc
    3 = RT mean
    4 = nNaN
    
    '''
    
    if day == 1:
        day_idx = 0
    if day == 3:
        day_idx = 1
    if day == 4:
        day_idx = 2
        
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
    
    for key, value in statsAll.items():
        result = value[day_idx,:,w_idx]
        print(result.shape)
        subsAll.append(result)
        
        if key in subjIDs_NF:
            subsNF.append(result)
        if key in subjIDs_C:
            subsC.append(result) 
    
    return subsAll, subsNF, subsC

#%% NOT block-wise, statsDay. Check whether this corresponds to the other (V2) beh measures.
os.chdir(saveDir)

statsDayAll = {}

for idx,subjID in enumerate(sub_all):
    statsDay, statsBlock, statsBlock_day2 = computeStatsBlocks(subjID)
    
    statsDayAll[subjID] = statsDay
#%%
def extractStatsDayAll(day_idx,wanted_measure):
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
    
    for key, value in statsDayAll.items():
#        print(value.shape)
        result = value[day_idx,w_idx]
#        print(result.shape)
        subsAll.append(result)
        
        if key in subjIDs_NF:
            subsNF.append(result)
        if key in subjIDs_C:
            subsC.append(result) 
    
    return subsAll, subsNF, subsC
#%%
# Reality check
senAll_d1, senNF_d1, senC_d1 = extractStatsDayAll(1,'sen')
senAll_d3, senNF_d3, senC_d3 = extractStatsDayAll(3,'sen')

  
sen_feedback[:,0] # Sensitivity, feedback group, day1
sen_control[:,0]

sen_feedback[:,1]

# Checked 29April, legit. V2 and V3 analysis corresponds.

#%% 
def make4Bars(wanted_measure,title,ylabel):
    '''
    Comparing day 1 and 3, NF and controls, using the statsDayAll structure and extract func.
    
    Connected lines bar plot
    
    '''
    all_d1, NF_d1, C_d1 = extractStatsDayAll(1,wanted_measure)
    all_d3, NF_d3, C_d3 = extractStatsDayAll(3,wanted_measure)
    
    print('NF mean day 1: ',round(np.mean(NF_d1),3))
    print('NF mean day 3: ',round(np.mean(NF_d3),3))
    print('C mean day 1: ',round(np.mean(C_d1),3))
    print('C mean day 3: ',round(np.mean(C_d3),3))
    
    y_min = np.min([np.min(all_d1),np.min(all_d3)])
    y_max = np.max([np.max(all_d1),np.max(all_d3)])
    
    # Connect lines
    plt.figure(random.randint(0,100))
    plt.ylabel(ylabel)
    plt.ylim([y_min-0.05,y_max+0.05])
#    plt.ylim([y_min-5,y_max+5])
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
        
    return plt, t_fb, t_c

def make2Bars(wanted_measure,title,ylabel):
    '''
    Comparing day 2, NF and controls, using the statsDayAll structure and extract func.    
    '''
    
    all_d2, NF_d2, C_d2 = extractStatsDayAll(2,wanted_measure)
    
    print('NF mean day 2: ',round(np.nanmean(NF_d2),3))
    print('C mean day 2: ',round(np.mean(C_d2),3))
    
    y_min = np.nanmin(all_d2)
    y_max = np.nanmax(all_d2)
    
    # Connect lines
    plt.figure(random.randint(0,100))
    plt.ylabel(ylabel)
    plt.ylim([y_min-0.01,y_max+0.01])
#    plt.ylim([y_min-5,y_max+5])
    plt.xticks([1,2],['NF day 2','Control day 2'])
    plt.title(title)
    
    plt.bar(1,np.nanmean(NF_d2),color=(0,0,0,0),edgecolor='tomato')
    plt.bar(2,np.mean(C_d2),color=(0,0,0,0),edgecolor='dodgerblue')
    
    plt.scatter(np.full(11,1),NF_d2,color='tomato')
    plt.scatter(np.full(11,2),C_d2,color='dodgerblue')
    
    # Omit one subject from C?
    t = stats.ttest_ind(NF_d2,C_d2,nan_policy='omit')
        
    return plt, t, all_d2

#%%
pl, t_fb, t_c = make4Bars('rt','Overall response time','Response time (s)')
pl, t_fb, t_c = make4Bars('rt_lure','Response time for lures','Response time (s)')
pl, t_fb, t_c = make4Bars('rt_nlure','Response time for non lures','Response time (s)')
pl, t_fb, t_c = make4Bars('nkeypress','Number of total keypresses','Number keypresses')
pl, t_fb, t_c = make4Bars('nan','Number of total NaNs','Number ') # Not saved

pl, t_fb, t_c = make4Bars('sen','x','x')
pl, t_fb, t_c = make4Bars('spec','x','x')
pl, t_fb, t_c = make4Bars('acc','x','x')

#diff_fb_p=stats.ttest_rel(sen_feedback[:,0],sen_feedback[:,1])
#diff_c_p=stats.ttest_rel(sen_control[:,0],sen_control[:,1])

# Investigate day 2, using make2Bars and StatsDayAll (NOT block-wise)
pl, t, sen_all_d2 = make2Bars('sen','Sensitivity day 2','Sensitivity') 
pl, t, acc_all_d2 = make2Bars('acc','Accuracy day 2','Accuracy') 
pl, t, rt_all_d2 = make2Bars('rt','RT day 2','RT') 
pl, t = make2Bars('nkeypress','Number of total keypresses','Number keypresses') 

# Check whether this looks reasonable by checking one day 2 file manually. DONE.

#%% Is good decoding accuracy correlated with a good day 2 behavioral response?
# Omit 11 
sen_all_d2_c = np.copy(sen_all_d2)
sen_all_d2_c = sen_all_d2_c[~np.isnan(sen_all_d2_c)]

# Acc
acc_all_d2_c = np.copy(acc_all_d2)
acc_all_d2_c = acc_all_d2_c[~np.isnan(acc_all_d2_c)]

# RT
rt_all_d2_c = np.copy(rt_all_d2)
rt_all_d2_c = rt_all_d2_c[~np.isnan(rt_all_d2_c)]

subsAll_c = np.copy(subsAll) # From EEG inv 18April
subsAll_c = np.delete(subsAll_c, 2)

plt.scatter(subsAll_c,sen_all_d2_c)
plt.ylabel('Sensitivity day 2')
plt.xlabel('Real time decoding accuracy, bias corrected')
plt.title('Sensitivity day 2 vs. RT decoding accuracy, N=21')

plt.scatter(subsAll_c,acc_all_d2_c)
plt.ylabel('Accuracy day 2')
plt.xlabel('Real time decoding accuracy, bias corrected')
plt.title('Accuracy day 2 vs. RT decoding accuracy, N=21')

# Plot RT vs decoding acc
plt.scatter(subsAll_c,rt_all_d2_c)
plt.ylabel('Response time day 2')
plt.xlabel('Real time decoding accuracy, bias corrected')
plt.title('Response time day 2 vs. RT decoding accuracy, N=21')

stats.linregress(acc_all_d2_c,subsAll_c)




np.corrcoef(sen_all_d2_c,subsAll_c)

stats.linregress(sen_all_d2_c,subsAll_c)

np.corrcoef(acc_all_d2_c,subsAll_c)

#%% Plot sensitivity block-wise, with individual lines for subjects, day1
subsAll, subsNF, subsC = extractBehVal(1,'sen')

colorC = sns.color_palette("Blues",11)
colorNF = sns.color_palette("Reds",11)

# Plot NF subjects
plt.figure(20)
for j in range(len(subsNF)):
    plt.plot(subsNF[j],color=colorNF[j],linewidth=0.8)
    
plt.plot(np.mean(subsNF,axis=0),label='Mean NF group',color='red',linewidth=2.0)

# Plot C subjects
for i in range(len(subsC)):
    plt.plot(subsC[i],color=colorC[i],linewidth=0.8)
    
plt.plot(np.mean(subsC,axis=0),label='Mean control group',color='blue',linewidth=2.0)

    
plt.plot(np.mean(subsAll,axis=0),label='Mean all participants',color='black',linewidth=2.0)
plt.title('Sensitivity across blocks, all participants, day 1')
plt.xticks(np.arange(0,16),[str(item) for item in np.arange(1,17)])
plt.xlabel('Block number, day 1')
plt.ylabel('Sensitivity')
plt.legend()

#%% sensitivity block-wise, with individual lines for subjects, day3
subsAll, subsNF, subsC = extractBehVal(3,'sen')

colorC = sns.color_palette("Blues",11)
colorNF = sns.color_palette("Reds",11)

# Plot NF subjects
plt.figure(21)
for j in range(len(subsNF)):
    plt.plot(subsNF[j],color=colorNF[j],linewidth=0.8)
    
plt.plot(np.mean(subsNF,axis=0),label='Mean NF group',color='red',linewidth=2.0)

# Plot C subjects
for i in range(len(subsC)):
    plt.plot(subsC[i],color=colorC[i],linewidth=0.8)
    
plt.plot(np.mean(subsC,axis=0),label='Mean control group',color='blue',linewidth=2.0)

    
plt.plot(np.mean(subsAll,axis=0),label='Mean all participants',color='black',linewidth=2.0)
plt.title('Sensitivity across blocks, all participants, day 3')
plt.xticks(np.arange(0,16),[str(item) for item in np.arange(1,17)])
plt.xlabel('Block number, day 3')
plt.ylabel('Sensitivity')
plt.legend()

#%% spec block-wise, with individual lines for subjects, day1
subsAll, subsNF, subsC = extractBehVal(1,'spec')

colorC = sns.color_palette("Blues",11)
colorNF = sns.color_palette("Reds",11)

# Plot NF subjects
plt.figure(23)
for j in range(len(subsNF)):
    plt.plot(subsNF[j],color=colorNF[j],linewidth=0.8)
    
plt.plot(np.mean(subsNF,axis=0),label='Mean NF group',color='red',linewidth=2.0)

# Plot C subjects
for i in range(len(subsC)):
    plt.plot(subsC[i],color=colorC[i],linewidth=0.8)
    
plt.plot(np.mean(subsC,axis=0),label='Mean control group',color='blue',linewidth=2.0)

    
plt.plot(np.mean(subsAll,axis=0),label='Mean all participants',color='black',linewidth=2.0)
plt.title('Specificity across blocks, all participants, day 1')
plt.xticks(np.arange(0,16),[str(item) for item in np.arange(1,17)])
plt.xlabel('Block number, day 1')
plt.ylabel('Specificity')
plt.legend()


#%% spec block-wise, with individual lines for subjects, day3
subsAll, subsNF, subsC = extractBehVal(3,'spec')

colorC = sns.color_palette("Blues",11)
colorNF = sns.color_palette("Reds",11)

# Plot NF subjects
plt.figure(24)
for j in range(len(subsNF)):
    plt.plot(subsNF[j],color=colorNF[j],linewidth=0.8)
    
plt.plot(np.mean(subsNF,axis=0),label='Mean NF group',color='red',linewidth=2.0)

# Plot C subjects
for i in range(len(subsC)):
    plt.plot(subsC[i],color=colorC[i],linewidth=0.8)
    
plt.plot(np.mean(subsC,axis=0),label='Mean control group',color='blue',linewidth=2.0)

    
plt.plot(np.mean(subsAll,axis=0),label='Mean all participants',color='black',linewidth=2.0)
plt.title('Specificity across blocks, all participants, day 3')
plt.xticks(np.arange(0,16),[str(item) for item in np.arange(1,17)])
plt.xlabel('Block number, day 3')
plt.ylabel('Specificity')
plt.legend()


#%% acc block-wise, with individual lines for subjects, day1
subsAll, subsNF, subsC = extractBehVal(1,'acc')

colorC = sns.color_palette("Blues",11)
colorNF = sns.color_palette("Reds",11)

# Plot NF subjects
plt.figure(23)
for j in range(len(subsNF)):
    plt.plot(subsNF[j],color=colorNF[j],linewidth=0.8)
    
plt.plot(np.mean(subsNF,axis=0),label='Mean NF group',color='red',linewidth=2.0)

# Plot C subjects
for i in range(len(subsC)):
    plt.plot(subsC[i],color=colorC[i],linewidth=0.8)
    
plt.plot(np.mean(subsC,axis=0),label='Mean control group',color='blue',linewidth=2.0)

    
plt.plot(np.mean(subsAll,axis=0),label='Mean all participants',color='black',linewidth=2.0)
plt.title('Accuracy across blocks, all participants, day 1')
plt.xticks(np.arange(0,16),[str(item) for item in np.arange(1,17)])
plt.xlabel('Block number, day 1')
plt.ylabel('Accuracy')
plt.legend()

#%% acc block-wise, with individual lines for subjects, day1
subsAll, subsNF, subsC = extractBehVal(3,'acc')

colorC = sns.color_palette("Blues",11)
colorNF = sns.color_palette("Reds",11)

# Plot NF subjects
plt.figure(23)
for j in range(len(subsNF)):
    plt.plot(subsNF[j],color=colorNF[j],linewidth=0.8)
    
plt.plot(np.mean(subsNF,axis=0),label='Mean NF group',color='red',linewidth=2.0)

# Plot C subjects
for i in range(len(subsC)):
    plt.plot(subsC[i],color=colorC[i],linewidth=0.8)
    
plt.plot(np.mean(subsC,axis=0),label='Mean control group',color='blue',linewidth=2.0)

    
plt.plot(np.mean(subsAll,axis=0),label='Mean all participants',color='black',linewidth=2.0)
plt.title('Accuracy across blocks, all participants, day 3')
plt.xticks(np.arange(0,16),[str(item) for item in np.arange(1,17)])
plt.xlabel('Block number, day 3')
plt.ylabel('Accuracy')
plt.legend()



















