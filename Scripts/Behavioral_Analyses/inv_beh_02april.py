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


saveDir = 'P:\\closed_loop_data\\beh_analysis\\'
os.chdir(saveDir)


# Extract values from dict
#dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
#
#wanted_keys = ("alpha_fromfile_overall","alpha_fromfile_run")
#
#result = dictfilt(pkl_arr[0], wanted_keys)
#
#g = list(result.values())
    
#%%
with open('Beh_subjID_02.pkl', "rb") as fin:
    sub02 = (pickle.load(fin))[0]

with open('Beh_subjID_03.pkl', "rb") as fin:
    sub03 = (pickle.load(fin))[0]

with open('Beh_subjID_15.pkl', "rb") as fin:
    sub15 = (pickle.load(fin))[0]

with open('Beh_subjID_07.pkl', "rb") as fin:
    sub07 = (pickle.load(fin))[0]
    
with open('Beh_subjID_08.pkl', "rb") as fin:
    sub08 = (pickle.load(fin))[0]
    
with open('Beh_subjID_11.pkl', "rb") as fin:
    sub11 = (pickle.load(fin))[0]

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

f15=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f15[count,:]=computeStats('15',day)
    




#%% Initial sensitivity measures
# Sensitivity
#f07_TP = sub07['no_Inhibitions_nonlure_day_1']
#f07_FN = sub07['inhibitions_nonlure_day_1']
#
#(f07_TP)/(f07_TP+f07_FN)
#
#f07_TP3 = sub07['no_Inhibitions_nonlure_day_3']
#f07_FN3 = sub07['inhibitions_nonlure_day_3']
#
#(f07_TP3)/(f07_TP3+f07_FN3)
#
## Specificity
#f07_TN = sub07['inhibitions_lure_day_1']
#f07_FP = sub07['no_Inhibitions_lure_day_1']
#
#(f07_TN)/(f07_TN+f07_FP)
#
#f07_TN3 = sub07['inhibitions_lure_day_3']
#f07_FP3 = sub07['no_Inhibitions_lure_day_3']
#
#(f07_TN3)/(f07_TN3+f07_FP3)

# Successrate, inhibitions_lure + no_inhibitions_nonlure/all mouse clicks?

# Accuracy, TP + TN/all



#%% Plots for RT surrounding lures
    
c1=sub07['surrounding_CR_Lst_day_1']
c3=sub07['surrounding_CR_Lst_day_3']
f1=sub07['surrounding_FR_Lst_day_1']
f3=sub07['surrounding_FR_Lst_day_3']

c4=sub07['surrounding_CR_Lst_day_4']
c5=sub07['surrounding_CR_Lst_day_5']
f4=sub07['surrounding_FR_Lst_day_4']
f5=sub07['surrounding_FR_Lst_day_5']

c13 = [statistics.mean(k) for k in zip(c1, c3)]
c45 = [statistics.mean(k) for k in zip(c4, c5)]

f13 = [statistics.mean(k) for k in zip(f1, f3)]
f45 = [statistics.mean(k) for k in zip(f4, f5)]

ticks = ['-3','-2','-1','lure','1','2','3']

plt.figure()
plt.plot(c13,color='green')
plt.plot(f13,color='red')
plt.title('Day 1 and 3, sub07')
plt.xticks(np.arange(0,7,1),ticks)

plt.figure(2)
plt.plot(c45,color='green')
plt.plot(f45,color='red')
plt.title('Day 4 and 5, sub07')
plt.xticks(np.arange(0,7,1),ticks)