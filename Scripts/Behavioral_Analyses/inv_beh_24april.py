# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:12:38 2019

@author: Greta
"""

import pickle
from matplotlib import pyplot as plt
import os
import statistics
import numpy as np
from scipy import stats


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

stats.ttest_rel(acc_feedback[:,0],acc_feedback[:,1])
stats.ttest_rel(acc_control[:,0],acc_control[:,1])


#%% T tests

# Sensitivity
diff_fb=np.mean(sen_feedback[:,1]-sen_feedback[:,0])
diff_c=np.mean(sen_control[:,1]-sen_control[:,0])

diff_fb_p=stats.ttest_rel(sen_feedback[:,0],sen_feedback[:,1])
diff_c_p=stats.ttest_rel(sen_control[:,0],sen_control[:,1])


# Acc
diff_fb_acc=np.mean(acc_feedback[:,1]-acc_feedback[:,0])
diff_c_acc=np.mean(acc_control[:,1]-acc_control[:,0])

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

