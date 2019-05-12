# -*- coding: utf-8 -*-
"""
This script is based on inv_18apr.py, and investigates the new EEG pipeline, 
using 450 samples for 19 subjects, and 550 samples for the first 3 subjects. 
The .pckl files are from 18th of April, and the script used is offline_analysis_FUNC.py,
together with the parallelisation script for cluster-use.

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

subjID_NF = ['07','08','11','13','14','16','19','22','26','27','30']
subjID_C = ['15','17','18','21','23','24','25','31','32','33','34']

# For plots, comparing NF and C
color = ['tomato']*11 + ['dodgerblue']*11
color_uncor = ['brown']*11 + ['navy']*11
sub_axis = subjID_NF + subjID_C
sub_axis_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

#%% New pipeline: offline_analysis_FUNC, 18 April
os.chdir('P:\\closed_loop_data\\offline_analysis_pckl\\')

d_all = {}

# Append all subject dictionaries to an overall dict, d_all
subs550 = ['07','08','11']
subs450 = ['13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

for subj in subs550:
    with open('18April_550_subj_'+subj+'.pkl', "rb") as fin:
         d_all[subj] = (pickle.load(fin))[0]

for subj in subs450:
    with open('18April_subj_'+subj+'.pkl', "rb") as fin:
         d_all[subj] = (pickle.load(fin))[0] 

#%% Pipeline with updated LORO and ouput alpha and clf output values

d_all2 = {}

for subj in sub_axis_all:
    with open('09May_subj_'+subj+'.pkl', "rb") as fin:
         d_all2[subj] = (pickle.load(fin))[0]

#%% Test difference between old and new LORO CV
subsAll, subsNF, subsC, meanAll, meanNF, meanC = extractVal('LORO_stable_acc_corr')         
subsAll2, subsNF2, subsC2, meanAll2, meanNF2, meanC2 = extractVal('LORO_stable_acc_corr')         

#%% Check GROUP assignments (checked 23 April - OK)
dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
GROUP_keys = ('ALPHA_correlation','GROUP')

allSubs_GROUP = []
for key, value in d_all.items():
    result = dictfilt(value, GROUP_keys)
    print(key,result)
    g = list(result.values())    
    allSubs_GROUP.append(g)

#%% Extract values from EEG dict, d_all

def extractVal(wanted_key):
    '''
    Remember to change from d_all to d_all2 (if using analyses from 09May)
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

    return subsAll, subsNF, subsC, meanAll, meanNF, meanC

#%%
subsAll, subsNF, subsC, meanAll, meanNF, meanC = extractVal('RT_test_acc_corr')
subsAll_uncor, subsNF_uncor, subsC_uncor, meanAll_uncor, meanNF_uncor, meanC_uncor = extractVal('RT_test_acc_uncorr')

# Save subsAll for beh comparsion
# np.save('subsAll_RT_acc.npy',subsAll)

#%% RT pipeline analysis. RT decoding accuracy corrected.

plt.figure(5)
plt.scatter(np.arange(0,len(subsNF),1),subsNF,color='tomato') # NF subjects
plt.scatter(np.arange(len(subsC),len(subsAll),1),subsC,color='dodgerblue') # C subjects
plt.xticks(np.arange(0,len(subsAll),1),sub_axis)
plt.xlabel('Subject ID')
plt.ylabel('Decoding accuracy')
plt.title('Real-time decoding accuracy (NF blocks)')
NF_mean = [meanNF]*len(subsNF)
C_mean = [meanC]*len(subsC)

plt.plot(np.arange(0,len(subsNF)),NF_mean, label='Mean NF group',color='tomato')
plt.plot(np.arange(len(subsNF),len(subsAll)),C_mean, label='Mean Control group',color='dodgerblue')
plt.legend()

# Barplot
plt.figure(6)
plt.bar(0, meanNF,color=(0,0,0,0),edgecolor='tomato',width=0.1)
plt.bar(0.2, meanC,color=(0,0,0,0),edgecolor='dodgerblue',width=0.1)
plt.ylabel('RT decoding accuracy (NF blocks)')
plt.xticks([0,0.2],['NF group','Control group'])
plt.ylim([0.5,0.73])

plt.scatter(np.zeros((len(subsNF))),subsNF,color='tomato')
plt.scatter(np.full(len(subsC),0.2),subsC,color='dodgerblue')

#%% Plot RT accuracies cor vs uncor

plt.figure(7)
plt.scatter(np.arange(0,len(subsNF),1),subsNF,color='tomato') # NF
plt.scatter(np.arange(0,len(subsNF_uncor),1),subsNF_uncor,color='brown') # NF

plt.scatter(np.arange(len(subsC),len(subsAll),1),subsC,color='dodgerblue') # C
plt.scatter(np.arange(len(subsC_uncor),len(subsAll_uncor),1),subsC_uncor,color='navy') # C

plt.xticks(np.arange(0,len(subsAll),1),sub_axis)
plt.xlabel('Subject ID')
plt.ylabel('Decoding accuracy')
plt.title('Real-time decoding accuracy (NF blocks)\n Classifier bias corrected vs uncorrected')

NF_mean = [meanNF]*len(subsNF)
C_mean = [meanC]*len(subsC)

NF_mean_uncor = [meanNF_uncor]*len(subsNF)
C_mean_uncor = [meanC_uncor]*len(subsC)

plt.plot(np.arange(0,len(subsNF)),NF_mean, color='tomato',label='Bias corrected')
plt.plot(np.arange(len(subsNF),len(subsAll)),C_mean,color='dodgerblue',label='Bias corrected')

plt.plot(np.arange(0,len(subsNF)),NF_mean_uncor, color='brown',label='Bias uncorrected')
plt.plot(np.arange(len(subsNF),len(subsAll)),C_mean_uncor, color='navy',label='Bias uncorrected')
plt.legend()

#%% Plot all face vs scene accuracies for RT (corrected)

subsAll_s, subsNF_s, subsC_s, meanAll_s, meanNF_s, meanC_s = extractVal('RT_scene_acc')
subsAll_f, subsNF_f, subsC_f, meanAll_f, meanNF_f, meanC_f = extractVal('RT_face_acc')

# Continuous subject ID x-axis
plt.figure(8)
plt.scatter(np.arange(0,len(subsAll_s),1),subsAll_s,color='seagreen')#,label='Bias corrected')
plt.scatter(np.arange(0,len(subsAll_f),1),subsAll_f,color='hotpink')#,label='Bias uncorrected')
plt.xticks(np.arange(0,len(subsAll),1),sub_axis_all) 
plt.xlabel('Subject ID')
plt.ylabel('Decoding accuracy')
plt.title('Real-time decoding accuracy (NF blocks)\n Scene vs face decoding accuracy')

s_mean = [meanAll_s]*len(subsAll)
f_mean = [meanAll_f]*len(subsAll)

plt.plot(np.arange(0,len(subsAll_s)),s_mean,color='seagreen',label='Scene decoding accuracy')
plt.plot(np.arange(0,len(subsAll_f)),f_mean,color='hotpink',label='Face decoding accuracy')
plt.legend()

# subject ID x-axis based on NF and C
plt.figure(9)
# Scene
plt.scatter(np.arange(0,len(subsNF_s),1),subsNF_s,color='seagreen',marker='^')#,label='NF group') # NF subjects
plt.scatter(np.arange(len(subsC_s),len(subsAll_s),1),subsC_s,color='seagreen')#,label='Control group') # C subjects
# Face
plt.scatter(np.arange(0,len(subsNF_f),1),subsNF_f,color='hotpink',marker='^') # NF subjects
plt.scatter(np.arange(len(subsC_f),len(subsAll_f),1),subsC_f,color='hotpink') # C subjects

plt.xticks(np.arange(0,len(subsAll),1),sub_axis) # SUBAXIS
plt.xlabel('Subject ID')
plt.ylabel('Decoding accuracy')
plt.title('Real-time decoding accuracy (NF blocks)\n Scene vs face decoding accuracy')

plt.plot(np.arange(0,len(subsAll_s)),s_mean ,color='seagreen',label='Scene decoding accuracy')
plt.plot(np.arange(0,len(subsAll_f)),f_mean, color='hotpink',label='Face decoding accuracy')
plt.legend()

#%% Overall per RT run accuracy, corrected
subsAll_run, subsNF_run, subsC_run, meanAll_run, meanNF_run, meanC_run = extractVal('RT_test_acc_corr_run')

# cmap = plt.get_cmap('hsv')
# colors = [cmap(i) for i in np.linspace(0, 1, 22)]

colorAll = sns.color_palette("Set2",22)

# Individual runs, mean
run1=np.mean(np.asarray([subsAll_run[f][0] for f in range(len(subsAll_run))]))
run2=np.mean(np.asarray([subsAll_run[f][1] for f in range(len(subsAll_run))]))
run3=np.mean(np.asarray([subsAll_run[f][2] for f in range(len(subsAll_run))]))
run4=np.mean(np.asarray([subsAll_run[f][3] for f in range(len(subsAll_run))]))
run5=np.mean(np.asarray([subsAll_run[f][4] for f in range(len(subsAll_run))]))

run_means = [run1]+[run2]+[run3]+[run4]+[run5] # Reality check: Same as np.mean(subsAll)

plt.figure(10)
for c,entry in enumerate(subsAll_run):
    #plt.plot(np.arange(1,6),subsAll_run[c],color=colorAll[c],linewidth=0.3)
    plt.scatter(np.arange(1,6),subsAll_run[c],color=colorAll[c])
plt.plot(np.arange(1,6),run_means,linestyle='-',color='black',label='Mean accuracy per run',linewidth=2)
plt.xticks(np.arange(1,6),['1','2','3','4','5']) 
plt.xlabel('NF run number')
plt.ylabel('Decoding accuracy')
plt.title('Real-time decoding accuracy for each NF run')
plt.legend()

#%% Plot offline train LORO accuracies, bias corrected, stable
subsAll_LORO, subsNF_LORO, subsC_LORO, meanAll_LORO, meanNF_LORO, meanC_LORO = extractVal('LORO_stable_acc_corr')

# np.save('subsAll_LORO_09May.npy',subsAll_LORO)
# np.save('subsNF_LORO_09May.npy',subsNF_LORO)
# np.save('subsC_LORO_09May.npy',subsC_LORO)

plt.figure(11)
plt.scatter(np.arange(0,len(subsNF_LORO),1),subsNF_LORO,color='tomato') # NF subjects
plt.scatter(np.arange(len(subsC_LORO),len(subsAll_LORO),1),subsC_LORO,color='dodgerblue') # C subjects

plt.xticks(np.arange(0,len(subsAll_LORO),1),sub_axis)
plt.xlabel('Subject ID')
plt.ylabel('Decoding accuracy')

NF_mean_LORO = [meanNF_LORO]*len(subsNF)
C_mean_LORO = [meanC_LORO]*len(subsC)
plt.title('Offline decoding accuracy (stable blocks)\n Leave one run out cross-validation',fontsize=13)

plt.plot(np.arange(0,len(subsNF_LORO)),NF_mean_LORO, label='NF group', color='tomato')
plt.plot(np.arange(len(subsNF_LORO),len(subsAll_LORO)),C_mean_LORO, label='Control group', color='dodgerblue')
plt.legend()


#%% Plot offline train LOBO accuracies, corrected, stable
subsAll_LOBO, subsNF_LOBO, subsC_LOBO, meanAll_LOBO, meanNF_LOBO, meanC_LOBO = extractVal('LOBO_stable_train_acc_corr')

# np.save('subsNF_LOBO.npy',subsNF_LOBO)
# np.save('subsC_LOBO.npy',subsC_LOBO)


plt.figure(12)
plt.scatter(np.arange(0,len(subsNF_LOBO),1),subsNF_LOBO,color='tomato') # NF subjects
plt.scatter(np.arange(len(subsC_LOBO),len(subsAll_LOBO),1),subsC_LOBO,color='dodgerblue') # C subjects
plt.xticks(np.arange(0,len(subsAll_LOBO),1),sub_axis)
plt.xlabel('Subject ID')
plt.ylabel('Decoding accuracy')

NF_mean_LOBO = [meanNF_LOBO]*len(subsNF_LOBO)
C_mean_LOBO = [meanC_LOBO]*len(subsC_LOBO)
plt.title('Leave one block out')

plt.title('Offline decoding accuracy (stable blocks)\n Leave one block out cross-validation')

plt.plot(np.arange(0,len(subsNF_LOBO)),NF_mean_LOBO, label='NF group', color='tomato')
plt.plot(np.arange(len(subsNF_LOBO),len(subsAll_LOBO)),C_mean_LOBO, label='Control group', color='dodgerblue')
plt.legend()

#%% ######### MNE ############

#%% Function that outputs evokeds
def extractMNE(wanted_key):
    subsAll = []
    # subsNF = []
    # subsC = []
    
    for key, value in d_all.items():
        # subsNF_result = []
        # subsC_result = []
        
        for k, v in value.items():        
            if k == wanted_key:                
                subsAll.append(v)
                
                # if key in subjID_NF:
                #     subsNF_result.append(v)
                # if key in subjID_C:
                #     subsC_result.append(v)
        
        # if len(subsNF_result) == 1:
        #     subsNF.append(subsNF_result)
            
        # if len(subsC_result) == 1:
        #     subsC.append(subsC_result)
    
    # meanAll = np.mean(subsAll)
    # meanNF = np.mean(subsNF)
    # meanC = np.mean(subsC)

    return subsAll


def extractEvokeds(epochsArray):
    evokeds = [epochsArray[name].average() for name in ('scene','face')]
    return evokeds, evokeds[0],evokeds[1] # 0 for scenes, 1 for faces


#%% Extract evoked responses (averaged)
    
MNEstable_all = extractMNE('MNE_stable_blocks_SSP')

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
    
#%% 
# Plotting evoked responses for 1 subject, individually
MNEstable_all[0]['face'].average().plot(spatial_colors=True, time_unit='s',picks=[7])
MNEstable_all[0]['scene'].average().plot(spatial_colors=True, time_unit='s',picks=[7])

mne.viz.plot_compare_evokeds(MNEscene_all[:9],picks=[6,7,12,13,22]) # Plotting all the individual evoked arrays (up to 10)
mne.viz.plot_compare_evokeds(MNEface_all[:9],picks=[6,7,12,13,22]) # Plotting all the individual evoked arrays (up to 10)

# If using 01, 02, Oz, PO3 and PO4. 6,7,12,13,22. These values are not z-scored. Standardize when extracting in offline_analysis.

# from mne.decoding import Scaler
# scale_test=Scaler(scalings='mean').fit_transform(MNEstable_all[0][2].get_data())


#%% 
colorAll = sns.color_palette("Set2",44)

# Plot of the SSP projectors, 1 subject
MNEstable_all[0].average().plot_projs_topomap()
# Consider adding p_variance values to the plot.

# Plot the topomap of the power spectral density across epochs.
MNEstable_all[0].plot_psd_topomap(proj=True)

# Category-wise
MNEstable_all[0]['face'].plot_psd_topomap(proj=True)
MNEstable_all[0]['scene'].plot_psd_topomap(proj=True)

# Plot topomap (possibiity of adding specific times)
MNEstable_all[0].average().plot_topomap(proj=True,times=np.linspace(0.05, 0.15, 5))

# Grand averaging
MNEevoked_scene = mne.grand_average(MNEscene_all)
MNEevoked_face = mne.grand_average(MNEface_all)

MNEevoked_scene.plot_joint()
MNEevoked_face.plot_joint()

# If manually adding a sensor plot
MNEevoked_scene.plot_sensors(show_names=True)

# Evokeds based on conditions
colors = 'blue', 'red'
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
e_dict['Scene'] = [item[0] for item in MNEevoked_all]
e_dict['Face'] = [item[1] for item in MNEevoked_all]

mne.viz.plot_compare_evokeds(e_dict,picks=[6,7,12,13,22],colors=['r','b'],\
                             truncate_xaxis=False,title='Scene vs face ERP, meaned across all participants',\
                             show_sensors=True,show_legend=True,truncate_yaxis=False,ci=True)

#%% Sort epochs based on categories

MNEepochs_all = []
for idx,subj in enumerate(MNEstable_all):
    sorted_epochsarray = [MNEstable_all[idx][name] for name in ('scene','face')]
    MNEepochs_all.append(sorted_epochsarray)


#%% ??

sorted_epochsarray = [MNEstable_all[0][name] for name in ('scene','face')]

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
        print
    if cat == 1:
        evoked_array_c1.append(mne.EvokedArray(g2[idx], info_fs100,tmin=-0.1,comment=cat)) # Faces 1

e_dict={}
e_dict['0'] = evoked_array_c0
e_dict['1'] = evoked_array_c1
# Could create these e_dicts for several people, and plot the means. Or create an e_dict
# with the evokeds for each person, and make the "overall" mean with individual evoked means across subjects.


#%% Scene vs face ERP (evoked grand average), meaned across all participants

# Compare mean of evoked response across all participants. Comparing categories.
e_dict = {}
e_dict['Scene'] = MNEevoked_scene # Grand averages
e_dict['Face'] = MNEevoked_face

mne.viz.plot_compare_evokeds(e_dict,picks=[6,7,12,13,22],colors=['r','b'],\
                             truncate_xaxis=False,title='Scene vs face ERP, meaned across all participants',\
                             show_sensors=True,show_legend=True,truncate_yaxis=False) #cmap='viridis'


#%% Currently not relevant

# Plot image
MNEstable_all[0].plot_image()


# Noise covariance plot - not really sure what to make of this (yet)
# noise_cov = mne.compute_covariance(stable_blocksSSP_plot)
# fig = mne.viz.plot_cov(noise_cov, stable_blocksSSP_plot.info) 

#%%
# Make animation
fig,anim = evokeds[0].animate_topomap(times=np.linspace(0.00, 0.79, 100),butterfly=True)
# Save animation
fig,anim = evokeds[0].animate_topomap(times=np.linspace(0.00, 0.79, 50),frame_rate=10,blit=False)
anim.save('Brainmation.gif', writer='imagemagick', fps=10)