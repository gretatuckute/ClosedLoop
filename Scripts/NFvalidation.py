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

scriptsDir = 'C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\'

#%% Plot styles
plt.style.use('seaborn-notebook')

matplotlib.rc('font', **font)

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['grid.alpha'] = 0.3

#%% Variables
subjID_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

subjID_NF = ['07','08','11','13','14','16','19','22','26','27','30']
subjID_C = ['15','17','18','21','23','24','25','31','32','33','34']

n_it = 5

#%%
os.chdir('P:\\closed_loop_data\\offline_analysis_pckl\\')

d2_all = {}

for subj in subjID_all:
    with open('08May_subj_'+subj+'.pkl', "rb") as fin:
         d2_all[subj] = (pickle.load(fin))[0]

#%%
def extractVal2(wkey):
    subsAll = []
    subsNF = []
    subsC = []
    
    for key, value in d2_all.items():
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
            subsNF.append(subsNF_result)
            
        if len(subsC_result) == 1:
            subsC.append(subsC_result)
    
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

def blockAlpha():
    
    alphaAll = []
    accAll = []
    clfoAll = []
    
    
    subsAll_a, subsNF_a, subsC_a = extractVal2('ALPHA_test')
    subsAll_clf, subsNF_clf, subsC_clf = extractVal2('CLFO_test')
    
    
    for idx, item in subsAll_a:
        above_a = len(np.where((np.array(item)>0.5))[0])/len(item)
        print(subjID_all[idx],above_a)

        
    # above_clfo = len(np.where((np.array(subsAll_clf)>0))[0])/len(subsAll_clf)

    
    # Alpha and clf output and alpha accuracy per block
    a_per_block = np.zeros((n_it*4))
    acc_per_block = np.zeros((n_it*4))
    clfo_per_block = np.zeros((n_it*4))
    k = 0
    
    for b in range(n_it*4):
        a_per_block[b] = (np.mean(subsAll_a[k:k+50])) # Mean alpha per block
        acc_per_block[b] = (len(np.where((np.array(subsAll_a[k:k+50])>0.5))[0])/50)
        clfo_per_block[b] = (np.mean(subsAll_clf[k:k+50])) 
        k += 50
        
    return a_per_block, acc_per_block, clfo_per_block



def plotAlphaVSbeh():
    
    # Extract wanted behavioral measure
    subsAll_sen, subsNF_sen, subsC_sen = extractStatsBlockDay2('sen')
    
    a_per_block, acc_per_block, clfo_per_block = blockAlpha()
    
    NFBlocks = np.sort(np.concatenate([np.arange(12,8+n_it*8,8),np.arange(13,8+n_it*8,8),np.arange(14,8+n_it*8,8),np.arange(15,8+n_it*8,8)]))
    
    senBlocksNF = (np.copy(senBlocks))[NFBlocks]
    accBlocksNF = (np.copy(accBlocks))[NFBlocks]
    specBlocksNF = (np.copy(specBlocks))[NFBlocks]
    RTBlocksNF = (np.copy(RTBlocks))[NFBlocks]
    
    # Plot
    plt.step(np.arange(1,n_it*4+1),a_per_block,where='post',label='alpha',linewidth=4.0)
    plt.step(np.arange(1,n_it*4+1),clfo_per_block,where='post',label='clf output',linewidth=4.0)
    plt.step(np.arange(1,n_it*4+1),acc_per_block,where='post',label='decoding acc',linewidth=4.0)
    
    
    plt.xticks(np.arange(1,n_it*4+1),[str(item) for item in np.arange(1,n_it*4+1)])
    plt.step(np.arange(1,n_it*4+1),senBlocksNF,where='post',label='sensitivity')
    plt.step(np.arange(1,n_it*4+1),specBlocksNF,where='post',label='specificity')
    plt.step(np.arange(1,n_it*4+1),accBlocksNF,where='post',label='accuracy')
    plt.step(np.arange(1,n_it*4+1),RTBlocksNF,where='post',label='RT')
    
    plt.legend()
    
    
    
    
#%% Check correlation between NF matched alpha subject and that particular subject

# Delta beh output vs correlation of real and yoked clf output

def corrControl():
    # Rearrange to I can make paired test
    NF_group = ['07','08','11','13','14','16','19','22','26','27','30']
    C_group = ['17','18','15','24','21','33','25','32','34','23','31']
    
    d_match = {}
    
    for i in range(len(NF_group)):
        d_match[NF_group[i]] = C_group[i]
    
    
    # for key, value in d_all.items():
        
    for k_match, val_match in d_match.items():
        # print(k_match)
        for key, val in d_all.items():
            # print(key)
            if k_match == key:
                print(k_match)
                # Compare two alpha files. Check wheter corr between alpha and clf output is 
                
                print(val_match) # Take this subject, which is the key! extract alpha file from values
                
                
                alpha_control = []
                # Extract alpha from here
                for name,item in d_all[val_match].items():
                    if name == 'subjID':
                        alpha_control.append(item) # Appending to a list 
                    
                print(val) # extract alpha file here too 
    
    
#%% Classifier output pre (and post?) FR and CR
os.chdir(scriptsDir)
from responseTime_func import extractCat

saveDir = 'P:\\closed_loop_data\\beh_analysis\\' 
EEGDir = 'P:\\closed_loop_data\\offline_analysis_pckl\\' 

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
    
    # Find a limit, i.e. which is the closest FR or CR to end of the list 
    
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
    
    return np.nanmean(post3_FR), np.nanmean(post3_CR)

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

#%%
# Errorbars

sem_FR_NF = np.std(preFR_NF)/np.sqrt(10)
sem_CR_NF = np.std(preCR_NF)/np.sqrt(10)

sem_FR_C = np.std(preFR_C)/np.sqrt(11)
sem_CR_C = np.std(preCR_C)/np.sqrt(11)


#%% Plot
plt.figure(random.randint(0,100))
plt.ylabel('Mean classifier 3 trials post lure')
plt.xticks([1,2,3,4],['NF FR','NF CR','Control FR','Control CR'])# 'Control day 1, part 2', 'Control day 3, part 2'])
# plt.title(title)

plt.scatter(np.full(10,1),preFR_NF,color='lightsalmon')
plt.scatter(np.full(10,2),preCR_NF,color='lightsalmon')
plt.scatter(np.full(11,3),preFR_C,color='powderblue')
plt.scatter(np.full(11,4),preCR_C,color='powderblue')

for i in range(10):
    plt.plot([(np.full(10,1))[i],(np.full(10,2))[i]], [(preFR_NF)[i],(preCR_NF)[i]],color='lightsalmon')

for i in range(11):
    plt.plot([(np.full(11,3))[i],(np.full(11,4))[i]], [(preFR_C)[i],(preCR_C)[i]],color='powderblue')
    
plt.plot([(np.full(1,1)),(np.full(1,2))], [(np.mean(preFR_NF)),np.mean(preCR_NF)],color='black')
plt.plot([(np.full(1,3)),(np.full(1,4))], [(np.mean(preFR_C)),np.mean(preCR_C)],color='black')

(_, caps, _) = plt.errorbar(np.full(1,1),np.mean(preFR_NF),yerr=sem_FR_NF, capsize=8, color='black',elinewidth=2,barsabove=True)
for cap in caps:
    cap.set_markeredgewidth(2)
(_, caps, _) = plt.errorbar(np.full(1,2),np.mean(preCR_NF),yerr=sem_CR_NF, capsize=8, color='black',elinewidth=2,barsabove=True)
for cap in caps:
    cap.set_markeredgewidth(2)

(_, caps, _) = plt.errorbar(np.full(1,3),np.mean(preFR_C),yerr=sem_FR_C, capsize=8, color='black',elinewidth=2,barsabove=True)
for cap in caps:
    cap.set_markeredgewidth(2)
(_, caps, _) = plt.errorbar(np.full(1,4),np.mean(preCR_C),yerr=sem_FR_C, capsize=8, color='black',elinewidth=2,barsabove=True)
for cap in caps:
    cap.set_markeredgewidth(2)
    
    
    





#%%



print(stats.ttest_ind(preFR_NF,preCR_NF,nan_policy='omit'))
print(stats.ttest_ind(preFR_C,preCR_C,nan_policy='omit'))

#%%    Draft for pre and post analysis
        
block_len = 50
# I need info for when a lure was correctly rejected, and FR (not rejected). Match this with the clf output
                
clfo30 = np.copy(clf_output_test)
a30 = np.copy(alpha_test)

lureLst30 = np.copy(lureLst2)

# Only extract vals for the NF blocks
e_mock = np.arange((8+n_it*8)*block_len)
nf_blocks_idx = np.concatenate([e_mock[600+n*400:800+n*400] for n in range(n_it)]) # Neurofeedback blocks 
lureLstNF = lureLst30[nf_blocks_idx]
    
# There are a total of 5*8*6 = 240 lures during day 2
# Sorry, a lot of lists.. 

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


for count,trial in enumerate(lureLstNF):
    if trial == 'FR':
        on_FR.append(clfo30[count])
        pre1_FR.append(clfo30[count-1])
        pre2_FR.append(clfo30[count-2])
        pre3_FR.append(clfo30[count-3])
        
        post1_FR.append(clfo30[count+1])
        post2_FR.append(clfo30[count+2])
        post3_FR.append(clfo30[count+3])
        
    if trial == 'CR':
        on_CR.append(clfo30[count])
        pre1_CR.append(clfo30[count-1])
        pre2_CR.append(clfo30[count-2])
        pre3_CR.append(clfo30[count-3])
        
        post1_CR.append(clfo30[count+1])
        post2_CR.append(clfo30[count+2])
        post3_CR.append(clfo30[count+3])
        

np.mean(pre1_FR)

np.mean(pre1_CR)

np.mean(post3_FR)
np.mean(post3_CR)

    
    
    
    
    
    



