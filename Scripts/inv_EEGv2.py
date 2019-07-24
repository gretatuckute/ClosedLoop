# -*- coding: utf-8 -*-
"""
This script is based on inv_09May.py, and investigates the new EEG pipeline, 
using 450 samples for 19 subjects, and 550 samples for the first 3 subjects. 
The .pckl files are from 9th of May, and the script used is offline_analysis_FUNC_v1.py.

v2 implements error rate.

@author: Greta
"""

import pickle
import matplotlib
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import mne
import matplotlib.cm as cm
import statsmodels.stats.multitest as sm
from scipy import stats



scriptsDir = 'C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\'
os.chdir(scriptsDir)

from variables import *
from responseTime_func import outputStableLureIdx

#%% Pipeline with updated LORO and ouput alpha and clf output values

# d_all2 = {}

# for subj in subjID_all:
#     with open(EEGDir+'09May_subj_'+subj+'.pkl', "rb") as fin:
#          d_all2[subj] = (pickle.load(fin))[0]

# fname = 'd_all2.pkl'         
# with open(npyDir+fname, 'wb') as fout:
#      pickle.dump(d_all2, fout)

#%% ######## FUNCTIONS ########
def extractVal(wanted_key):
    '''
    Extract values from EEG dict, d_all, or d_all2 (if using analyses from 09May)
    '''
    subsAll = []
    subsNF = []
    subsC = []
    
    for key, value in d_all2.items():
        subsNF_result = []
        subsC_result = []
        
        for k, v in value.items():        
            if k == wanted_key:                
                subsAll.append(v)
                
                if key in subjID_NF:
                    subsNF_result.append(v)
                if key in subjID_C:
                    subsC_result.append(v)
        
        if len(subsNF_result) == 1:
            subsNF.append(subsNF_result)
            
        if len(subsC_result) == 1:
            subsC.append(subsC_result)
    
    meanAll = np.mean(subsAll)
    meanNF = np.mean(subsNF)
    meanC = np.mean(subsC)

    subsNF = [item for sublist in subsNF for item in sublist]
    subsC = [item for sublist in subsC for item in sublist]

    return np.asarray(subsAll), np.asarray(subsNF), np.asarray(subsC), meanAll, meanNF, meanC

def extractMNE(wanted_key):
    subsAll = []
    for key, value in d_all2.items():
        for k, v in value.items():        
            if k == wanted_key:                
                subsAll.append(v)
    return subsAll


def extractEvokeds(epochsArray):
    '''Function that outputs evokeds
    '''
    evokeds = [epochsArray[name].average() for name in ('scene','face')]
    return evokeds, evokeds[0],evokeds[1] # 0 for scenes, 1 for faces

#%% Extract RT acc, corr and non-corr
subsAll, subsNF, subsC, meanAll, meanNF, meanC = extractVal('RT_test_acc_corr')
subsAll_uncor, subsNF_uncor, subsC_uncor, meanAll_uncor, meanNF_uncor, meanC_uncor = extractVal('RT_test_acc_uncorr')

#%% RT pipeline analysis. RT decoding accuracy corrected.
plt.figure(5)
plt.scatter(np.arange(0,len(subsNF),1),1-subsNF,color='tomato') # NF subjects
plt.scatter(np.arange(len(subsC),len(subsAll),1),1-subsC,color='dodgerblue') # C subjects
plt.xticks(np.arange(0,len(subsAll),1),sub_axis)
plt.xlabel('Subject ID')
plt.ylabel('Decoding error rate')
plt.title('Real-time decoding error rate')
NF_mean = [meanNF]*len(subsNF)
C_mean = [meanC]*len(subsC)

plt.plot(np.arange(0,len(subsNF)),NF_mean, label='Mean NF',color='tomato')
plt.plot(np.arange(len(subsNF),len(subsAll)),C_mean, label='Mean control',color='dodgerblue')
plt.legend()
plt.grid(color='gainsboro',linewidth=0.5)


fig,ax = plt.subplots()
ax.grid(color='gainsboro',linewidth=0.5,zorder=0)
plt.bar(np.arange(0,len(subsNF),1),(1-subsNF),edgecolor='tomato',color='white',zorder=2,linewidth=0.5) # NF subjects
plt.bar(np.arange(len(subsC),len(subsAll),1),1-subsC,edgecolor='dodgerblue',color='white',zorder=2,linewidth=0.5) # C subjects
plt.xticks(np.arange(0,len(subsAll),1),[str(item) for item in np.arange(1,len(subsAll)+1,1)],zorder=5)
plt.xlabel('Participants')
plt.ylabel('Decoding error rate')
plt.title('Real-time decoding error rate')
NF_mean = np.asarray([meanNF]*11)
C_mean = np.asarray([meanC]*11)
plt.ylim(0.25,0.6)

plt.hlines(1-NF_mean,-0.5,11-0.5,label='Mean NF',color='tomato',zorder=4,linestyles='dashed',linewidth=2)
plt.hlines(1-C_mean,11-0.5,22-0.5,label='Mean control',color='dodgerblue',zorder=4,linestyles='dashed',linewidth=2)
plt.hlines(0.5,xmin=-0.5,xmax=21.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2)
plt.legend(loc='upper right')



#%% Plot RT accuracies cor vs uncor
fig,ax = plt.subplots()
ax.grid(color='gainsboro',linewidth=0.5,zorder=0)
plt.scatter(np.arange(0,11,1),(1-subsNF),color='tomato',zorder=3,label='NF corrected') # NF subjects
plt.scatter(np.arange(0,11,1),(1-subsNF_uncor),color='brown',zorder=3,label='NF uncorrected') # NF subjects

plt.scatter(np.arange(11,22,1),1-subsC,color='dodgerblue',zorder=3,label='Control corrected') # C subjects
plt.scatter(np.arange(11,22,1),1-subsC_uncor,color='navy',zorder=3,label='Control uncorrected') # C subjects

plt.xticks(np.arange(0,22,1),[str(item) for item in np.arange(1,22+1,1)])
plt.xlabel('Participants')
plt.ylabel('Decoding error rate')
plt.title('Real-time decoding error rate \nClassifier bias corrected vs. uncorrected')

NF_mean_uncor = np.asarray([meanNF_uncor]*11)
C_mean_uncor = np.asarray([meanC_uncor]*11)

plt.hlines(1-NF_mean,-0.5,11-0.5,color='tomato',zorder=4,linestyles='dashed',linewidth=2)
plt.hlines(1-NF_mean_uncor,-0.5,11-0.5,color='brown',zorder=4,linestyles='dashed',linewidth=2)

plt.hlines(1-C_mean,11-0.5,22-0.5,color='dodgerblue',zorder=4,linestyles='dashed',linewidth=2)
plt.hlines(1-C_mean_uncor,11-0.5,22-0.5,color='navy',zorder=4,linestyles='dashed',linewidth=2)


plt.ylim(0.27,0.55)
plt.legend(loc='upper right')

#%% Plot all face vs scene accuracies for RT (corrected) - could make this for LOBO, or for LORO (need to run another iteration of the script to save the face/scene accs)
subsAll_s, subsNF_s, subsC_s, meanAll_s, meanNF_s, meanC_s = extractVal('RT_scene_acc')
subsAll_f, subsNF_f, subsC_f, meanAll_f, meanNF_f, meanC_f = extractVal('RT_face_acc')

fig,ax = plt.subplots()
ax.grid(color='gainsboro',linewidth=0.5,zorder=0)
plt.scatter(np.arange(0,11,1),(1-subsNF_f),color='tomato',zorder=3,label='Face') # NF subjects
plt.scatter(np.arange(0,11,1),(1-subsNF_s),color='dodgerblue',zorder=3,label='Scene') # NF subjects

plt.scatter(np.arange(11,22,1),1-subsC_f,color='tomato',zorder=3) # C subjects
plt.scatter(np.arange(11,22,1),1-subsC_s,color='dodgerblue',zorder=3) # C subjects

plt.xticks(np.arange(0,22,1),[str(item) for item in np.arange(1,22+1,1)])
plt.xlabel('Participants')
plt.ylabel('Decoding error rate')
plt.title('Real-time decoding error rate \nFace vs. scene')

s_mean = np.asarray([meanAll_s]*len(subsAll))
f_mean = np.asarray([meanAll_f]*len(subsAll))

plt.hlines(1-f_mean,-0.5,22-0.5,color='tomato',zorder=4,linestyles='dashed',linewidth=2)
plt.hlines(1-s_mean,-0.5,22-0.5,color='dodgerblue',zorder=4,linestyles='dashed',linewidth=2)

plt.ylim(0.20,0.52)
plt.legend(loc='upper right')

#%% Overall per RT run accuracy, corrected - colored based on NF and C 
subsAll_run, subsNF_run, subsC_run, meanAll_run, meanNF_run, meanC_run = extractVal('RT_test_acc_corr_run')

subsAll_run = 1-subsAll_run
subsNF_run = 1-subsNF_run
subsC_run = 1-subsC_run

cmReds=cm.get_cmap("Reds")
cmReds1 = cmReds(np.linspace(0.5, 1, 100))
cmBlues=cm.get_cmap("Blues")

normNF = matplotlib.colors.Normalize(vmin=np.min(1-subsNF_RT_acc), vmax=np.max(1-subsNF_RT_acc))
normC = matplotlib.colors.Normalize(vmin=np.min(1-subsC_RT_acc), vmax=np.max(1-subsC_RT_acc))

# Individual runs, mean
run1=np.mean(np.asarray([subsAll_run[f][0] for f in range(len(subsAll_run))]))
run2=np.mean(np.asarray([subsAll_run[f][1] for f in range(len(subsAll_run))]))
run3=np.mean(np.asarray([subsAll_run[f][2] for f in range(len(subsAll_run))]))
run4=np.mean(np.asarray([subsAll_run[f][3] for f in range(len(subsAll_run))]))
run5=np.mean(np.asarray([subsAll_run[f][4] for f in range(len(subsAll_run))]))

run_means = [run1]+[run2]+[run3]+[run4]+[run5] # Reality check: Same as np.mean(subsAll)

fig,ax = plt.subplots()
ax.grid(color='gainsboro',linewidth=0.5,zorder=0)

# NF
for count,entry in enumerate(subsNF_run):
    plt.plot(np.arange(1,6),subsNF_run[count],c=cmReds(normNF(1-subsNF_RT_acc[count])),linewidth=0.5,zorder=3)
    plt.scatter(np.arange(1,6),subsNF_run[count],c=cmReds(normNF(1-subsNF_RT_acc[count])),zorder=3)
# C
for count,entry in enumerate(subsC_run):
    plt.plot(np.arange(1,6),subsC_run[count],c=cmBlues(normC(1-subsC_RT_acc[count])),linewidth=0.5,zorder=2)
    plt.scatter(np.arange(1,6),subsC_run[count],c=cmBlues(normC(1-subsC_RT_acc[count])),zorder=2)

plt.plot(np.arange(1,6),run_means,linestyle='dashed',color='black',label='Mean error rate per run',linewidth=2,zorder=3)
plt.xticks(np.arange(1,6),['1','2','3','4','5']) 
plt.xlabel('Run number')
plt.ylabel('Decoding error rate')
plt.title('Real-time decoding error rate per run')
plt.legend()
plt.ylim(0.20,0.6)

#%% Plot offline train LORO accuracies, bias corrected, stable
subsAll_LORO, subsNF_LORO, subsC_LORO, meanAll_LORO, meanNF_LORO, meanC_LORO = extractVal('LORO_stable_acc_corr')

fig,ax = plt.subplots()
ax.grid(color='gainsboro',linewidth=0.5,zorder=0)
plt.bar(np.arange(0,11,1),(1-subsNF_LORO),edgecolor='tomato',color='white',zorder=2,linewidth=0.5) # NF subjects
plt.bar(np.arange(11,22,1),1-subsC_LORO,edgecolor='dodgerblue',color='white',zorder=2,linewidth=0.5) # C subjects
plt.xticks(np.arange(0,22,1),[str(item) for item in np.arange(1,22+1,1)],zorder=5)
plt.xlabel('Participants')
plt.ylabel('Decoding error rate')
plt.title('Classifier error rate \nLeave-one-run-out cross-validation') #########
NF_mean_LORO = np.asarray([meanNF_LORO]*len(subsNF))
C_mean_LORO = np.asarray([meanC_LORO]*len(subsC))
plt.ylim(0.15,0.6)

plt.hlines(1-NF_mean_LORO,-0.5,11-0.5,label='Mean NF',color='tomato',zorder=4,linestyles='dashed',linewidth=2)
plt.hlines(1-C_mean_LORO,11-0.5,len(subsAll)-0.5,label='Mean control',color='dodgerblue',zorder=4,linestyles='dashed',linewidth=2)
plt.hlines(0.5,xmin=-0.5,xmax=21.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2)
plt.legend(loc='upper right')

#%% Plot offline train LOBO accuracies, corrected, stable
subsAll_LOBO, subsNF_LOBO, subsC_LOBO, meanAll_LOBO, meanNF_LOBO, meanC_LOBO = extractVal('LOBO_stable_train_acc_corr')

fig,ax = plt.subplots()
ax.grid(color='gainsboro',linewidth=0.5,zorder=0)
plt.bar(np.arange(0,11,1),(1-subsNF_LOBO),edgecolor='tomato',color='white',zorder=2,linewidth=0.5) # NF subjects
plt.bar(np.arange(11,22,1),1-subsC_LOBO,edgecolor='dodgerblue',color='white',zorder=2,linewidth=0.5) # C subjects
plt.xticks(np.arange(0,22,1),[str(item) for item in np.arange(1,22+1,1)],zorder=5)
plt.xlabel('Participants')
plt.ylabel('Decoding error rate')
plt.title('Classifier error rate \nLeave-one-block-out cross-validation') #########
NF_mean_LOBO = np.asarray([meanNF_LOBO]*len(subsNF))
C_mean_LOBO = np.asarray([meanC_LOBO]*len(subsC))
plt.ylim(0.15,0.6)

plt.hlines(1-NF_mean_LOBO,-0.5,11-0.5,label='Mean NF',color='tomato',zorder=4,linestyles='dashed',linewidth=2)
plt.hlines(1-C_mean_LOBO,11-0.5,22-0.5,label='Mean control',color='dodgerblue',zorder=4,linestyles='dashed',linewidth=2)
plt.hlines(0.5,xmin=-0.5,xmax=21.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2)
plt.legend(loc='upper right')

#%% ######### MNE ############
#%% Extract evoked responses (averaged)
MNEstable_all = extractMNE('MNE_stable_blocks_SSP')
y_stable_all = extractMNE('MNE_y_stable_blocks')

#%% Append scene and face evoked to lists
MNEscene_all = []
MNEface_all = []
MNEevoked_all = []

c = 0
for sub in MNEstable_all:
    # Crop the first 3 subjects from 110 samples to 90 samples
    if c <= 2:
        print(c)
        evokeds, scene_avg, face_avg = extractEvokeds(sub)
        scene_avg.crop(tmin=-0.1,tmax=0.79)
        face_avg.crop(tmin=-0.1,tmax=0.79)
        MNEscene_all.append(scene_avg)
        MNEface_all.append(face_avg)
        MNEevoked_all.append(evokeds)
        c += 1
    else:
        print(c)
        evokeds, scene_avg, face_avg = extractEvokeds(sub)
        MNEscene_all.append(scene_avg)
        MNEface_all.append(face_avg)
        MNEevoked_all.append(evokeds)
        c += 1

#%% Check whether MNE category labeling corresponds
face_avg_all = []
scene_avg_all = []
c = 0
for idx, subj in enumerate(MNEstable_all):
    if c <= 2:
        g1 = subj.get_data()
        g1 = g1[:,:,:90]
        y_stable = y_stable_all[idx]       
        g3 = g1[y_stable==True] # Faces = 1
        g4 = g1[y_stable==False] # Scenes = 0
        
        g3a = np.mean(g3,axis=0) #faces 
        g4a = np.mean(g4,axis=0)
        
        face_avg_all.append(g3a)
        scene_avg_all.append(g4a)
        c+=1
    else:
        g1 = subj.get_data()
        y_stable = y_stable_all[idx]       
        g3 = g1[y_stable==True] # Faces = 1
        g4 = g1[y_stable==False] # Scenes = 0
        
        g3a = np.mean(g3,axis=0) #faces 
        g4a = np.mean(g4,axis=0)
        
        face_avg_all.append(g3a)
        scene_avg_all.append(g4a)
        c+=1
    
plt.figure(14)
plt.plot(np.mean(face_avg_all,axis=0).T[:,7],color='red') # Corresponds to e_dict['Face'] as below
plt.plot(np.mean(scene_avg_all,axis=0).T[:,7],color='blue')
        
e_dict = {}
e_dict['Scene'] = [item[0] for item in MNEevoked_all] # Rearranges alphabetically
e_dict['Face'] = [item[1] for item in MNEevoked_all]

mne.viz.plot_compare_evokeds(e_dict,picks=[7],colors=['r','b'],\
                             truncate_xaxis=False,title='Scene vs face ERP, meaned across all participants',\
                             show_sensors=True,show_legend=True,truncate_yaxis=False,ci=False)
      
#%% Plotting evoked responses for 1 subject, individually
MNEstable_all[17]['face'].average().plot(spatial_colors=True, time_unit='s',picks=[7])
MNEstable_all[0]['scene'].average().plot(spatial_colors=True, time_unit='s',picks=[7])

mne.viz.plot_compare_evokeds(MNEscene_all[:9],picks=[6,7,12,13,22]) # Plotting all the individual evoked arrays (up to 10)
mne.viz.plot_compare_evokeds(MNEface_all[:2],picks=[6,7,12,13,22]) # Plotting all the individual evoked arrays (up to 10)

# If using 01, 02, Oz, PO3 and PO4. 6,7,12,13,22. These values are not z-scored. Standardize when extracting in offline_analysis.

#%% 
colorAll = sns.color_palette("Set2",44)

# Plot of the SSP projectors, 1 subject
MNEstable_all[20].average().plot_projs_topomap()
# Consider adding p_variance values to the plot.

# Plot the topomap of the power spectral density across epochs.
MNEstable_all[1].plot_psd_topomap(proj=False)

# Category-wise
MNEstable_all[0]['face'].plot_psd_topomap(proj=False)
MNEstable_all[0]['scene'].plot_psd_topomap(proj=True)

# Plot topomap (possibiity of adding specific times)
MNEstable_all[0].average().plot_topomap(proj=True,times=np.linspace(0.05, 0.15, 5))
MNEstable_all[0].average().plot_topomap(proj=False)

# Grand averaging
MNEevoked_scene = mne.grand_average(MNEscene_all)
MNEevoked_face = mne.grand_average(MNEface_all)

#%% Stats
ch_idx = [6,7,12,13,22]

scene_5chs = np.zeros((22,90))
dat_5chs = np.zeros((22,5,90))
for idx,evokedentry in enumerate(MNEscene_all):
    dat = evokedentry._data
    dat5chs = dat[ch_idx]
    dat_5chs[idx] = dat5chs
    dat5chs_mean = np.mean(dat5chs,axis=0)
    print(dat5chs_mean.shape)
    scene_5chs[idx] = dat5chs_mean
    
face_5chs = np.zeros((22,90))
dat_5chs_face = np.zeros((22,5,90))
for idx,evokedentry in enumerate(MNEface_all):
    dat = evokedentry._data
    dat5chs = dat[ch_idx]
    dat_5chs_face[idx] = dat5chs
    dat5chs_mean = np.mean(dat5chs,axis=0)
    face_5chs[idx] = dat5chs_mean

p_vals = []
for el in range(0,90):
    t = stats.ttest_rel(scene_5chs[:,el],face_5chs[:,el])
    p_vals.append(t[1])

plt.figure(50)
plt.scatter(np.arange(0,90),p_vals)
plt.figure(52)
plt.plot(np.arange(0,90),p_vals)

# plt.xticks(np.arange(0,95,5),[str(item) for item in np.arange(-100,850,50)])

sig_p = []
for p in p_vals:
    if p < 0.05:
        sig_p.append(p)
    else:
        sig_p.append(np.nan)

log_pvals = -(np.log10(p_vals))

cor_alpha = 0.05/90
cor_alphalog = -(np.log10(cor_alpha))

ax,fig = plt.subplots()
plt.plot(log_pvals)
plt.xlabel('Samples')
plt.ylabel('-log(p)')
plt.hlines(cor_alphalog,0,90,color='red')

bools, p_adj, x, x2 = sm.multipletests(p_vals,method='bonferroni')

#%% Use MNE cluster permutation
X_input = [face_5chs,scene_5chs]
X_3Dinput = [dat_5chs,dat_5chs_face]
# Reshape to [22, 90, 5]
dat_5chs_r = dat_5chs.transpose(0, 2, 1)
dat_5chs_face_r = dat_5chs_face.transpose(0, 2, 1)

tfce = dict(start=0.2, step=.2)

# Make a simple test with mean of 2D input
face_mean = np.mean(face_5chs, axis=0)
scene_mean = np.mean(scene_5chs, axis=0)

# Visualize the means
plt.figure(100)
plt.plot(face_mean)
plt.plot(scene_mean)

F, c1, c_pvals1, H01 = mne.stats.permutation_cluster_test([face_mean,scene_mean])

# Create artificial data
noise = np.random.normal(0,0.2,[22,90])
noise2 = np.random.normal(0,0.2,[22,30])

cond1 = np.ones((22,90)) + noise

flat = np.ones((22,30)) + noise2
peak = np.full((22,30),10) + noise2

cond2 = np.hstack((flat,peak,flat))

cond1_3d = np.dstack((cond1,cond1,cond1))
cond2_3d = np.dstack((cond2,cond2,cond2))

F, c1, c_pvals1, H01 = mne.stats.permutation_cluster_test([cond1,cond2],n_permutations=100,threshold=10,tail=1) # Finds one cluster

# The above test corresponds to 
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test([face_5chs,scene_5chs],threshold=0.1,n_permutations=1000,tail=1)#,threshold=3) 

times = np.arange(0,90)

plt.subplot(211)
plt.plot(np.arange(0,90),face_mean)
plt.plot(np.arange(0,90),scene_mean)

plt.subplot(212)
for i_c, c in enumerate(clusters):
    c = c[0]
    
    h = plt.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
    
# Artificial data
T_obs1, clusters1, cluster_p_values1, H01 = mne.stats.permutation_cluster_test([cond1,cond2],n_permutations=1000,tail=1,threshold=3) 


plt.subplot(211)
plt.plot(np.arange(0,90),np.mean(cond1,axis=0))
plt.plot(np.arange(0,90),np.mean(cond2,axis=0))

plt.subplot(212)
for i_c, c in enumerate(clusters1):
    c = c[0]
    
    h = plt.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)    


#%% Artificial data, 3d
F, c1, c_pvals1, H01 = mne.stats.spatio_temporal_cluster_test([cond1_3d,cond2_3d])
# Finds the one, big cluster

# If 2d input:
F, c1, c_pvals1, H01 = mne.stats.permutation_cluster_test(X_input,tfce,tail=1,n_permutations=1000)
sig_points = c_pvals1.reshape(F.shape).T < .05

plt.figure(10)
plt.plot(c_pvals1)

plt.figure(10)
plt.plot(sig_points)





X_3D_trans = [dat_5chs_face_r,dat_5chs_r]
F, c, c_pvals, H0 = mne.stats.spatio_temporal_cluster_test(X_3D_trans,threshold=tfce)





# 2d using epochsarray
t1 = MNEstable_all[2]['face'].get_data()
t2 = MNEstable_all[2]['scene'].get_data()

t1t = t1.transpose(0, 2, 1)
t2t = t2.transpose(0, 2, 1)

t1t_oc = t1t[:,:,ch_idx]
t2t_oc = t2t[:,:,ch_idx]

# If using 3d:
# observations x time x space
# Extract data: transpose because the cluster test requires channels to be last
# In this case, inference is done over items. In the same manner, we could
# also conduct the test over, e.g., subjects.
Fobs1, clusters1, clusters_pval1, H01=mne.stats.spatio_temporal_cluster_test([t1t_oc,t2t_oc],n_permutations=100)





Fobs1, clusters1, clusters_pval1, H01 = mne.stats.permutation_cluster_test([t1[:,6,:],t2[:,6,:]],threshold=tfce,tail=1)
plt.plot(clusters_pval1)

Fobs1, clusters1, clusters_pval1, H01=mne.stats.spatio_temporal_cluster_test(t1t,t2t,n_permutations=100)

significant_points = clusters_pval1.reshape(Fobs1.shape).T < .05

print(str(significant_points.sum()) + " points selected by TFCE ...")
plt.plot(significant_points)


#%%
# Get evoked
scene_evoked_allSubs = MNEevoked_scene._data
face_evoked_allSubs = MNEevoked_face._data

# Means
scene_evoked_mean = np.mean(scene_evoked_allSubs,axis=0)
face_evoked_mean = np.mean(face_evoked_allSubs,axis=0)

# Plot joint
ts_args = {}
ts_args['truncate_yaxis'] = False

fig1 = MNEevoked_scene.plot_joint(title=None)
fig1.savefig(figDir+'sceneERP.pdf',bbox_inches = "tight")

fig2 = MNEevoked_face.plot_joint(title=None)
fig2.savefig(figDir+'faceERP.pdf',bbox_inches = "tight")

# If manually adding a sensor plot
MNEevoked_scene.plot_sensors(show_names=True)

# Evokeds based on conditions
title = 'Subject \nscene vs face'

# Plot evoked across all channels, comparing two categories, 1 subject
mne.viz.plot_evoked_topo(MNEevoked_all[17], title=title, background_color='w') # color=colors

# Compare two categories
mne.viz.plot_compare_evokeds(MNEevoked_all,title=title,show_sensors=True,cmap=None,colors=['b','r'])#,ci=True)
# When multiple channels are passed, this function combines them all, to get one time course for each condition. 

# For each subject separately, no CI
mne.viz.plot_compare_evokeds(MNEevoked_all[17],title=title,show_sensors=True,ci=True,picks=[7],colors=['b','r'])


#%% Evoked responses across all participants with CI 95% bootstrapped confidence interval, based on categories

# Create dict where with key "Scene" for all participants' evoked responses to scenes.
e_dict = {}
e_dict['Scene'] = [item[0] for item in MNEevoked_all] # Rearranges alphabetically
e_dict['Face'] = [item[1] for item in MNEevoked_all]

mne.viz.plot_compare_evokeds(e_dict,picks=[6,7,12,13,22],colors=['r','b'],\
                             truncate_xaxis=False,title='Scene vs face ERP, meaned across all participants',\
                             show_sensors=True,show_legend=True,truncate_yaxis=False,ci=True)

mne.viz.plot_compare_evokeds(e_dict,picks=[6,7,12,13,22],colors=['r','b'],\
                             truncate_xaxis=False,title=' ',\
                             show_sensors=True,show_legend=True,truncate_yaxis=False,ci=True)


mne.viz.plot_compare_evokeds(e_dict,picks=[6,7,12,13,22])


# Plot for individual subjects, based on category, e.g. face here
dict_face = e_dict['Face']

for item in dict_face[6:7]:
    mne.viz.plot_compare_evokeds(item,picks=[6,7,12,13,22])

# Plot averaged evoked for each subject, no CI
for item in MNEevoked_all[6:8]:
    mne.viz.plot_compare_evokeds(item,picks=[6,7,12,13,22],colors=['r','b'],\
        truncate_xaxis=False,title=None,\
        show_sensors=True,show_legend=True,truncate_yaxis=False,ci=True)
    # Save and add to big plot
    # No 2 (by default is the face category)


# Plot grand averaged ERPs for both categories
MNEevoked_face.plot(picks=[6,7,12,13,22])
MNEevoked_scene.plot(picks=[6,7,12,13,22])

#%% Sort epochs based on categories
MNEepochs_all = []
for idx,subj in enumerate(MNEstable_all):
    sorted_epochsarray = [MNEstable_all[idx][name] for name in ('scene','face')]
    MNEepochs_all.append(sorted_epochsarray)

#%% Plot evoked averages for individual subjects with CI
for idx,subj in enumerate(MNEstable_all):
    single_sub = MNEstable_all[idx].get_data()
   
    evoked_array_c0 = []
    evoked_array_c1 = []

    for catidx,cat in enumerate(y_stable_all[idx]):
        if cat == 0:
            evoked_array_c0.append(mne.EvokedArray(single_sub[catidx], info_fs100,tmin=-0.1,comment=cat)) # Scenes 0
            print
        if cat == 1:
            evoked_array_c1.append(mne.EvokedArray(single_sub[catidx], info_fs100,tmin=-0.1,comment=cat))

    e_dict_single = {}
    e_dict_single['Scene'] = evoked_array_c0
    e_dict_single['Face'] = evoked_array_c1

    fig = mne.viz.plot_compare_evokeds(e_dict_single,picks=[6,7,12,13,22],colors=['r','b'],\
        truncate_xaxis=False,title='Participant ' + str(subjID_all[idx]),\
        show_sensors=False,show_legend=True,truncate_yaxis=False,ci=True)


    fig.savefig('C://Users//Greta//Desktop//closed_loop//RESULTS//thesis_plots//EEG_done//single_ERPs//evoked_'+str(subjID_all[idx])+'.png',dpi=180)

#%%
# Appending all entries in the overall epochsarray as single evoked arrays of shape (n_channels, n_times) 
g2 = MNEstable_all[0][0].get_data()
evoked_array = [mne.EvokedArray(entry, info_fs100,tmin=-0.1) for entry in g2]

# Appending all entries in the overall epochsarray as single evoked arrays of shape (n_channels, n_times) - category info added as comments
events_list = y_stable_blocks # LOAD THIS FROM SOMEWHERE!!!!!
event_id = dict(scene=0, face=1)
n_epochs = len(events_list)
events_list = [int(i) for i in events_list]

evoked_array2 = []
for idx,cat in enumerate(events_list):
    evoked_array2.append(mne.EvokedArray(g2[idx], info_fs100,tmin=-0.1,comment=cat))
    
mne.viz.plot_compare_evokeds(evoked_array2[:10],picks=[7]) # Plotting all the individual evoked arrays (up to 10)

# Testing the best way to plot individual evoked arrays
evoked_array2[0].plot(picks=[7])

# Creating a dict of lists: Condition 0 and condition 1 with evoked arrays.
evoked_array_c0 = []
evoked_array_c1 = []

for idx,cat in enumerate(events_list):
    if cat == 0:
        evoked_array_c0.append(mne.EvokedArray(g2[idx], info_fs100,tmin=-0.1,comment=cat)) # Scenes 0
    if cat == 1:
        evoked_array_c1.append(mne.EvokedArray(g2[idx], info_fs100,tmin=-0.1,comment=cat)) # Faces 1

e_dict={}
e_dict['0'] = evoked_array_c0
e_dict['1'] = evoked_array_c1
# Could create these e_dicts for several people, and plot the means. Or create an e_dict
# with the evokeds for each person, and make the "overall" mean with individual evoked means across subjects.


#%% Scene vs face ERP (evoked grand average), meaned across all participants. No CI.

# Compare mean of evoked response across all participants. Comparing categories.
e_dict = {}
e_dict['Scene'] = MNEevoked_scene # Grand averages
e_dict['Face'] = MNEevoked_face

mne.viz.plot_compare_evokeds(e_dict,picks=[6,7,12,13,22],colors=['r','b'],\
                             truncate_xaxis=False,title='Scene vs face ERP, meaned across all participants',\
                             show_sensors=True,show_legend=True,truncate_yaxis=False) #cmap='viridis'

#%%
# Investigate lure vs non lure ERPs

# E.g. for subject 07

b = MNEstable_all[1].get_data()

lureStable = outputStableLureIdx('08')

b_lure = b[lureStable==1]
b_lure_avg = np.mean(b_lure,axis=0)
plt.plot(b_lure_avg[6,:])
b_nonlure = b[lureStable==0]

evoked_lure = []
evoked_nonlure = []

for idx,cat in enumerate(lureStable):
    if cat == 0:
        evoked_nonlure.append(mne.EvokedArray(b[idx], info_fs100, tmin=-0.1,comment=cat)) # nonlures 0
    if cat == 1:
        evoked_lure.append(mne.EvokedArray(b[idx], info_fs100,tmin=-0.1,comment=cat)) # lure 1



e_dict_s = {}
e_dict_s['nonlure'] = evoked_nonlure
e_dict_s['lure'] = evoked_lure

mne.viz.plot_compare_evokeds(e_dict_s,picks=[6,7,12,13,22],colors=['r','b'],\
        truncate_xaxis=False,title='Participant ' + str(subjID_all[idx]),\
        show_sensors=False,show_legend=True,truncate_yaxis=False,ci=True)

mne.viz.plot_compare_evokeds(e_dict_s,picks=[6])

# True lure labeling

#%% Save lure figs for individuals 
grand_e_dict = {}


for idx,subj in enumerate(MNEstable_all):
    single_sub = MNEstable_all[idx].get_data()
    print(idx)
    lureStable = outputStableLureIdx(subjID_all[idx])

    evoked_lure = []
    evoked_nonlure = []

    for catidx,cat in enumerate(lureStable):
        print(catidx)
        if cat == 0:
            evoked_nonlure.append(mne.EvokedArray(single_sub[catidx], info_fs100,tmin=-0.1,comment=cat)) # nonlure 0
        if cat == 1:
            evoked_lure.append(mne.EvokedArray(single_sub[catidx], info_fs100,tmin=-0.1,comment=cat)) # lure 1

    e_dict_s = {}
    e_dict_s['Non-lure'] = evoked_nonlure
    e_dict_s['Lure'] = evoked_lure
    
    # grand_e_dict['Non-lure'] = [e_dict_s['Non-lure']] # Rearranges alphabetically
    # grand_e_dict['Lure'] = [e_dict_s['Lure']]

    fig = mne.viz.plot_compare_evokeds(e_dict_s,picks=[6,7,12,13,22],colors=['r','g'],truncate_xaxis=False,title='Participant ' + str(subjID_all[idx]),show_sensors=False,show_legend=True,truncate_yaxis=False,ci=True)
    fig.savefig('C://Users//Greta//Desktop//closed_loop//RESULTS//thesis_plots//EEG_done//single_ERPs//lure_evoked_'+str(subjID_all[idx])+'.png',dpi=180)

mne.viz.plot_compare_evokeds(grand_e_dict,picks=[6,7,12,13,22])


#%%
# Make animation
fig,anim = evokeds[0].animate_topomap(times=np.linspace(0.00, 0.79, 100),butterfly=True)
# Save animation
fig,anim = evokeds[0].animate_topomap(times=np.linspace(0.00, 0.79, 50),frame_rate=10,blit=False)
anim.save('Brainmation.gif', writer='imagemagick', fps=10)