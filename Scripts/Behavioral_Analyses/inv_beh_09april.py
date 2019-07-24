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


#saveDir = 'P:\\closed_loop_data\\beh_analysis\\'
saveDir = 'C:\\Users\\sofha\\Documents\\closed_loop\\beh_analysis\\'
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
f22=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f22[count,:]=computeStats('22',day)
f21=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f21[count,:]=computeStats('21',day)
f23=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f23[count,:]=computeStats('23',day)
f25=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f25[count,:]=computeStats('25',day)
f27=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f27[count,:]=computeStats('30',day)
f30=np.zeros((4,4))
for count,day in enumerate(dayLst):
    f30[count,:]=computeStats('30',day)

sub_fb=[f07,f08,f26,f27,f30,f11,f13,f14,f16,f19,f22]
sub_c=[f17,f18,f23,f15,f21,f25,f31,f34,f32,f33,f24]
#%%





sen_feedback=np.asarray([f[:,0] for f in sub_fb])
sen_control=np.asarray([f[:,0] for f in sub_c])
spec_feedback=np.asarray([f[:,1] for f in sub_fb])
spec_control=np.asarray([f[:,1] for f in sub_c])
fpr_feedback=np.asarray([f[:,2] for f in sub_fb])
fpr_control=np.asarray([f[:,2] for f in sub_c])

plt.figure(1)
plt.bar(1,np.mean(sen_feedback[:,0]))
plt.bar(2,np.mean(sen_feedback[:,1]))
plt.bar(3,np.mean(sen_control[:,0]))
plt.bar(4,np.mean(sen_control[:,1]))
plt.figure(3)
plt.bar(1,np.mean(spec_feedback[:,0]))
plt.bar(2,np.mean(spec_feedback[:,1]))
plt.bar(3,np.mean(spec_control[:,0]))
plt.bar(4,np.mean(spec_control[:,1]))

plt.figure(2)
plt.bar(1,np.mean(sen_feedback[:,2]))
plt.bar(2,np.mean(sen_feedback[:,3]))
plt.bar(3,np.mean(sen_control[:,2]))
plt.bar(4,np.mean(sen_control[:,3]))

plt.figure(5)
plt.bar(1,np.mean(fpr_feedback[:,0]))
plt.bar(2,np.mean(fpr_feedback[:,1]))
plt.bar(3,np.mean(fpr_control[:,0]))
plt.bar(4,np.mean(fpr_control[:,1]))

from scipy import stats
diff_fb=np.mean(sen_feedback[:,1]-sen_feedback[:,0])
diff_c=np.mean(sen_control[:,1]-sen_control[:,0])
diff_fb_p=stats.ttest_rel(sen_feedback[:,0],sen_feedback[:,1])
diff_c_p=stats.ttest_rel(sen_control[:,0],sen_control[:,1])

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