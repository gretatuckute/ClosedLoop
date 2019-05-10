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

scriptsDir = 'C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\'
os.chdir(scriptsDir)
from responseTime_func import extractCat

saveDir = 'P:\\closed_loop_data\\beh_analysis\\' 
EEGDir = 'P:\\closed_loop_data\\offline_analysis_pckl\\' 

#%% Plot styles
plt.style.use('seaborn-notebook')

matplotlib.rc('font', **font)

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['grid.alpha'] = 0.5

#%% Variables
subjID_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

subjID_NF = ['07','08','11','13','14','16','19','22','26','27','30']
subjID_C = ['15','17','18','21','23','24','25','31','32','33','34']

n_it = 5

#%%
os.chdir('P:\\closed_loop_data\\offline_analysis_pckl\\')

d2_all = {}

for subj in subjID_all:
    with open('09May_subj_'+subj+'.pkl', "rb") as fin:
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

statsBlockDay2_all = {}

# Extract stats for all subjects
for idx,subjID in enumerate(subjID_all):
    statsDay, statsBlock, statsBlock_day2 = computeStats(subjID)
    
    statsBlockDay2_all[subjID] = statsBlock_day2 


#%%

def blockAlpha():

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
    
    # NF alpha
    for idx, item in enumerate(subsNF_a):
        k = 0
        for b in range(n_it*4):
            a_per_block_NF[idx,b] = (np.mean(item[k:k+50])) # Mean alpha per block
            acc_per_block_NF[idx,b] = len(np.where((np.array(item[k:k+50])>0.5))[0])/50
            k += 50
            
    # C alpha
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
    Returns averaged stats per block for NF subjects and C subjects
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

    
def plotMatchedAlphavsBeh():
    '''
    Does not average across participants, but plots the participant with its matched control participant.
    '''
    NFBlocks_idx = np.sort(np.concatenate([np.arange(12,8+n_it*8,8),np.arange(13,8+n_it*8,8),np.arange(14,8+n_it*8,8),np.arange(15,8+n_it*8,8)]))

    subsAll_beh, subsNF_beh, subsC_beh = extractStatsBlockDay2('acc')
    
    behBlocksNF_all = [] # Only NF blocks extracted
    for idx, item in enumerate(subsAll_beh):
        behBlocksNF = (np.copy(item))[NFBlocks_idx]
        behBlocksNF_all.append(behBlocksNF)
    
    # Extract alpha, accuracy and clf output per block
    a_per_block_all, acc_per_block_all, clfo_per_block_all = blockMatchedAlpha() 
    
    # For alpha control
    subsAll_a, subsNF_a, subsC_a = extractVal2('ALPHA_test')
    
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
                # Compare two alpha files. Check wheter corr between alpha and clf output is 
                
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
    
    # Manually add the last value as the same as the last?
    
    # Plot NF subject vs matched control, with the alpha shown (the NF person's alpha)
    fig,ax=plt.subplots()
    
    for axis in [ax.xaxis]:
        axis.set(ticks=np.arange(1.5,n_it*4+1), ticklabels=[str(item) for item in np.arange(1,n_it*4+1)])
    
    #plt.xticks(np.arange(1,n_it*4+2),[str(item) for item in np.arange(1,n_it*4+1)])
    # Plotting alpha of the NF subject, e.g. subj 7, which was also used for subj 17
    plt.step(np.arange(1,n_it*4+2),np.append(zscore(matchedAlphaBlock[7][0]),(zscore(matchedAlphaBlock[7][0])[-1:])),where='post',label='Alpha value (shown for both subjects)',linewidth=2.0, color='black')
    # plt.step(np.arange(1,n_it*4+1),zscore(clfoBlockNF_avg),where='post',label='clf output NF',linewidth=4.0)
    plt.step(np.arange(1,n_it*4+2),np.append(zscore(matchedBeh[7][0]),zscore(matchedBeh[7][0])[-1:]),where='post',label='Behavioral accuracy, NF subject',linewidth=1.0,color='tomato')
    plt.step(np.arange(1,n_it*4+2),np.append(zscore(matchedBeh[7][1]),zscore(matchedBeh[7][1])[-1:]),where='post',label='Behavioral accuracy, matched control subject',linewidth=1.0,color='dodgerblue')
    plt.xlabel('NF block number, day 2')
    plt.ylabel('Z-scored units')
    plt.legend()
    plt.title('Behavioral accuracy per block for NF subject and matched control')
    
    # Reality check
    # han = matchedAccBlock[7][0] # Must be the NF person of d_match index 7, i.e. subj 22
    # np.mean(han) # Matches with the RT accuracy for subj 22 in d_all2
    
def blockMatchedAlpha():
    NFBlocks_idx = np.sort(np.concatenate([np.arange(12,8+n_it*8,8),np.arange(13,8+n_it*8,8),np.arange(14,8+n_it*8,8),np.arange(15,8+n_it*8,8)]))

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
            a_per_block_all = a_per_block_all[]
            
            acc_per_block_all[idx,b] = len(np.where((np.array(item[k:k+50])>0.5))[0])/50
            k += 50    
            
    # clf output all
    for idx, item in enumerate(subsAll_clf):
        k = 0
        for b in range(n_it*4):
            clfo_per_block_all[idx,b] = (np.mean(item[k:k+50])) # Mean alpha per block
            k += 50  
            
    return a_per_block_all, acc_per_block_all, clfo_per_block_all
            
    


      
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

    
    
    
    
    
    



