# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:28:03 2019

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
plt.style.use('seaborn-pastel')

font = {'family' : 'sans-serif',
       'weight' : 1,
       'size'   : 12}

matplotlib.rc('font', **font)

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

#%% Old files
os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\pckl\\')

with open('04April_V2_subj_13.pkl', "rb") as fin:
    sub13 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_14.pkl', "rb") as fin:
    sub14 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_15.pkl', "rb") as fin:
    sub15 = (pickle.load(fin))[0] 
    
#%% Write a function that outputs evokeds
    
def extractEvokeds(epochsArray):
    evokeds = [epochsArray[name].average() for name in ('scene','face')]
    return evokeds[0],evokeds[1] # 0 for scenes, 1 for faces

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
         
    
#%% Check GROUP assignments (checked 23 April - OK)
dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
GROUP_keys = ('ALPHA_correlation','GROUP')

allSubs_GROUP = []
for key, value in d_all.items():
    result = dictfilt(value, GROUP_keys)
    print(key,result)
    g = list(result.values())    
    allSubs_GROUP.append(g)
    
#%% RT pipeline # Function in next section.

RT_keys = ('RT_train_acc','RT_correct_NFtest_pred','RT_test_acc_corr','RT_test_acc_corr_run','RT_scene_acc','RT_face_acc','RT_test_acc_uncorr','RT_test_acc_uncorr_run')

subjIDs_NF = ['07','08','11','13','14','16','19','22','26','27','30']
subjIDs_C = ['15','17','18','21','23','24','25','31','32','33','34']

#subsNF = ['7','8','26','27','30','11','13','14','16','19','22']
#subsC = ['17','18','23','31','34','15','21','24','25','32','33']
   
# Extract chosen keys in key, value pairs (not in random, unordered order)

subsNF_RT = []
subsC_RT = []

for key, value in d_all.items():
    subsNF_result = []
    subsC_result = []
    
    for k, v in value.items():        
        if k in RT_keys:
            if key in subsNF:
                subsNF_result.append([k,v])
            if key in subsC:
                subsC_result.append([k,v])
    
    if len(subsNF_result) > 1:
        subsNF_RT.append(subsNF_result)
        
    if len(subsC_result) > 1:
        subsC_RT.append(subsC_result)

RT_test_acc = [] # First NF and then C
for sub in subsNF_RT:
    for lst in sub:
        if lst[0] == 'RT_test_acc_corr':
            RT_test_acc.append(lst[1])

for sub in subsC_RT:
    for lst in sub:
        if lst[0] == 'RT_test_acc_corr':
            RT_test_acc.append(lst[1])
            
# Random order
#subsNF_RT = []
#subsC_RT = []
#for key, value in d_all.items():
#    result = dictfilt(value, RT_keys)
#    g = list(result.values()) 
#    
#    if key in subsNF:
#        subsNF_RT.append(g)
#    
#    if key in subsC:
#        subsC_RT.append(g)

#%%

def extractVal(wanted_key):
    subsAll = []
    subsNF = []
    subsC = []
    
    for key, value in d_all.items():
        subsNF_result = []
        subsC_result = []
        
        for k, v in value.items():        
            if k == wanted_key:                
                subsAll.append(v)
                
                if key in subjIDs_NF:
                    subsNF_result.append(v)
                if key in subjIDs_C:
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

#%% Plot only RT accuracies
color = ['tomato']*11 + ['dodgerblue']*11
sub_axis = subjIDs_NF + subjIDs_C

plt.figure(5)
plt.scatter(np.arange(0,len(subsAll),1),subsAll,color=color)
plt.xticks(np.arange(0,len(subsAll),1),sub_axis)
plt.xlabel('Subject ID')
plt.ylabel('RT decoding accuracy (NF blocks)')
#NF_mean = [np.mean(RT_test_acc[0:len(subsNF_RT)])]*len(subsNF_RT)
#C_mean = [np.mean(RT_test_acc[len(subsNF_RT):])]*len(subsC_RT)
NF_mean = [meanNF]*len(subsNF)
C_mean = [meanC]*len(subsC)

plt.plot(np.arange(0,len(subsNF)),NF_mean, label='MeanNF', linestyle='--',color='tomato')
plt.plot(np.arange(len(subsNF),len(subsAll)),C_mean, label='MeanC', linestyle='--',color='dodgerblue')

plt.figure(6)
plt.bar(0, meanNF,color=(0,0,0,0),edgecolor='tomato',width=0.1)
plt.bar(0.2, meanC,color=(0,0,0,0),edgecolor='dodgerblue',width=0.1)
plt.ylabel('RT decoding accuracy (NF blocks)')
plt.xticks([0,0.2],['NF group','Control group'])
plt.ylim([0.5,0.73])

plt.scatter(np.zeros((len(subsNF))),subsNF,color='tomato')
plt.scatter(np.full(len(subsC),0.2),subsC,color='dodgerblue')

#%% Plot RT accuracies cor vs uncor
color = ['tomato']*11 + ['dodgerblue']*11
color_uncor = ['brown']*11 + ['navy']*11
sub_axis = subjIDs_NF + subjIDs_C

plt.figure(7)
plt.scatter(np.arange(0,len(subsAll),1),subsAll,color=color)#,label='Bias corrected')
plt.scatter(np.arange(0,len(subsAll_uncor),1),subsAll_uncor,color=color_uncor)#,label='Bias uncorrected')
plt.xticks(np.arange(0,len(subsAll),1),sub_axis)
plt.xlabel('Subject ID')
plt.ylabel('RT decoding accuracy (NF blocks)')

NF_mean = [meanNF]*len(subsNF)
C_mean = [meanC]*len(subsC)

NF_mean_uncor = [meanNF_uncor]*len(subsNF)
C_mean_uncor = [meanC_uncor]*len(subsC)

plt.plot(np.arange(0,len(subsNF)),NF_mean, linestyle='--',color='tomato',label='Bias corrected')
plt.plot(np.arange(len(subsNF),len(subsAll)),C_mean, linestyle='--',color='dodgerblue',label='Bias corrected')

plt.plot(np.arange(0,len(subsNF)),NF_mean_uncor, linestyle='--',color='brown',label='Bias uncorrected')
plt.plot(np.arange(len(subsNF),len(subsAll)),C_mean_uncor, linestyle='--',color='navy',label='Bias uncorrected')
plt.legend()

#%% Plot all face vs scene accuracies for RT (corrected)
sub_axis_all = ['07','08','11','13','14','15','16','17','18','19','21','22','23','24','25','26','27','30','31','32','33','34']

subsAll_s, subsNF_s, subsC_s, meanAll_s, meanNF_s, meanC_s = extractVal('RT_scene_acc')
subsAll_f, subsNF_f, subsC_f, meanAll_f, meanNF_f, meanC_f = extractVal('RT_face_acc')

# Continuous subject ID x-axis
plt.figure(8)
plt.scatter(np.arange(0,len(subsAll_s),1),subsAll_s,color='seagreen')#,label='Bias corrected')
plt.scatter(np.arange(0,len(subsAll_f),1),subsAll_f,color='hotpink')#,label='Bias uncorrected')
plt.xticks(np.arange(0,len(subsAll),1),sub_axis_all) # SUBAXIS ALL CONTINUOUS
plt.xlabel('Subject ID')
plt.ylabel('RT decoding accuracy (NF blocks)')

s_mean = [meanAll_s]*len(subsAll)
f_mean = [meanAll_f]*len(subsAll)

plt.plot(np.arange(0,len(subsAll_s)),s_mean, linestyle='--',color='seagreen',label='Scene decoding accuracy')
plt.plot(np.arange(0,len(subsAll_f)),f_mean, linestyle='--',color='hotpink',label='Face decoding accuracy')
plt.legend()

# subject ID x-axis based on NF and C
plt.figure(9)
plt.scatter(np.arange(0,len(subsNF_s),1),subsNF_s,color='seagreen') # NF subjects
plt.scatter(np.arange(len(subsNF_s),len(subsAll_s),1),subsC_s,color='seagreen') # C subjects

plt.scatter(np.arange(0,len(subsNF_f),1),subsNF_f,color='hotpink') # NF subjects
plt.scatter(np.arange(len(subsNF_f),len(subsAll_f),1),subsC_f,color='hotpink') # C subjects

plt.xticks(np.arange(0,len(subsAll),1),sub_axis) # SUBAXIS
plt.xlabel('Subject ID')
plt.ylabel('RT decoding accuracy (NF blocks)')

s_mean = [meanAll_s]*len(subsAll)
f_mean = [meanAll_f]*len(subsAll)

plt.plot(np.arange(0,len(subsAll_s)),s_mean, linestyle='--',color='seagreen',label='Scene decoding accuracy')
plt.plot(np.arange(0,len(subsAll_f)),f_mean, linestyle='--',color='hotpink',label='Face decoding accuracy')
plt.legend()

#%% Overall per RT run accuracy, corrected
subsAll_run, subsNF_run, subsC_run, meanAll_run, meanNF_run, meanC_run = extractVal('RT_test_acc_corr_run')

cmap = plt.get_cmap('hsv')
colors = [cmap(i) for i in np.linspace(0, 1, 22)]

# Individual runs, mean
run1=np.mean(np.asarray([subsAll_run[f][0] for f in range(len(subsAll_run))]))
run2=np.mean(np.asarray([subsAll_run[f][1] for f in range(len(subsAll_run))]))
run3=np.mean(np.asarray([subsAll_run[f][2] for f in range(len(subsAll_run))]))
run4=np.mean(np.asarray([subsAll_run[f][3] for f in range(len(subsAll_run))]))
run5=np.mean(np.asarray([subsAll_run[f][4] for f in range(len(subsAll_run))]))

run_means = [run1]+[run2]+[run3]+[run4]+[run5]

plt.figure(10)
for c,entry in enumerate(subsAll_run):
#    plt.plot(np.arange(1,6),subsAll_run[c],color=colors[c],linestyle='-')
    plt.scatter(np.arange(1,6),subsAll_run[c],color=colors[c])
plt.plot(np.arange(1,6),run_means,linestyle='-',color='black',label='Mean accuracy per run')
plt.xticks(np.arange(1,6),['1','2','3','4','5']) 
plt.xlabel('NF run number')
plt.ylabel('RT decoding accuracy (NF blocks)')
plt.legend()



#%% Plot offline train LORO accuracies, corrected, stable
subsAll_t, subsNF_t, subsC_t, meanAll_t, meanNF_t, meanC_t = extractVal('LORO_stable_acc_corr')

color = ['tomato']*11 + ['dodgerblue']*11
sub_axis = subjIDs_NF + subjIDs_C

plt.figure(11)
plt.scatter(np.arange(0,len(subsAll_t),1),subsAll_t,color=color)
plt.xticks(np.arange(0,len(subsAll_t),1),sub_axis)
plt.xlabel('Subject ID')
plt.ylabel('Offline decoding accuracy (stable blocks), bias corrected')
#NF_mean = [np.mean(RT_test_acc[0:len(subsNF_RT)])]*len(subsNF_RT)
#C_mean = [np.mean(RT_test_acc[len(subsNF_RT):])]*len(subsC_RT)
NF_mean_t = [meanNF_t]*len(subsNF)
C_mean_t = [meanC_t]*len(subsC)
plt.title('Leave one run out')

plt.plot(np.arange(0,len(subsNF_t)),NF_mean_t, label='MeanNF', linestyle='--',color='tomato')
plt.plot(np.arange(len(subsNF_t),len(subsAll_t)),C_mean_t, label='MeanC', linestyle='--',color='dodgerblue')


#%% Plot offline train LORO accuracies, corrected, stable
subsAll_t, subsNF_t, subsC_t, meanAll_t, meanNF_t, meanC_t = extractVal('LOBO_stable_train_acc_corr')

color = ['tomato']*11 + ['dodgerblue']*11
sub_axis = subjIDs_NF + subjIDs_C

plt.figure(12)
plt.scatter(np.arange(0,len(subsAll_t),1),subsAll_t,color=color)
plt.xticks(np.arange(0,len(subsAll_t),1),sub_axis)
plt.xlabel('Subject ID')
plt.ylabel('Offline decoding accuracy (stable blocks), bias corrected')
#NF_mean = [np.mean(RT_test_acc[0:len(subsNF_RT)])]*len(subsNF_RT)
#C_mean = [np.mean(RT_test_acc[len(subsNF_RT):])]*len(subsC_RT)
NF_mean_t = [meanNF_t]*len(subsNF)
C_mean_t = [meanC_t]*len(subsC)
plt.title('Leave one block out')

plt.plot(np.arange(0,len(subsNF_t)),NF_mean_t, label='MeanNF', linestyle='--',color='tomato')
plt.plot(np.arange(len(subsNF_t),len(subsAll_t)),C_mean_t, label='MeanC', linestyle='--',color='dodgerblue')






#%% Extract MNE dict objects
wanted_keys = ('MNE_RT_epochs_fb_avg','MNE_RT_epochs_fb_nonavg','MNE_stable_blocks_SSP')

# Append to list
allSubs_scene = []
allSubs_face = []


for sub in allSubs_MNE:
    stable = sub[0]
    
    scene_avg, face_avg = extractEvokeds(stable)
    allSubs_scene.append(scene_avg)
    allSubs_face.append(face_avg)
    
mne.viz.plot_compare_evokeds(allSubs_scene,picks=[6,7,12,13,22]) # Plotting all the individual evoked arrays (up to 10)
mne.viz.plot_compare_evokeds(allSubs_face,picks=[6,7,12,13,22]) # Plotting all the individual evoked arrays (up to 10)

# If using 01, 02, Oz, PO3 and PO4. 6,7,12,13,22. These values are not z-scored. Standardize when extracting in offline_analysis.


from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP, PSDEstimator)


scale_test=Scaler(scalings='mean').fit_transform(allSubs_MNE[0][2].get_data())



# What is going on w. sub 15    
# allSubs_MNE[2][2]['face'].average().plot(spatial_colors=True, time_unit='s',picks=[7])

#%%
# Plot showing where the correct predictions are located pr. run.

n_it=5

pred_run = np.reshape(sub13n['RT_correct_NFtest_pred'],[n_it,200])

for run in range(n_it):
    plt.figure(run)
    plt.bar(np.arange(200),pred_run[run,:])
    
    
#%% Plot based on categories
allSubs_MNE[0][0]['face'].average().plot(spatial_colors=True, time_unit='s')#,picks=[7])
allSubs_MNE[0][0]['scene'].average().plot(spatial_colors=True, time_unit='s')

# Plot of the SSP projectors
allSubs_MNE[0][0].average().plot_projs_topomap()
# Consider adding p_variance values to the plot.

# Plot the topomap of the power spectral density across epochs.
stable_blocksSSP_plot.plot_psd_topomap(proj=True)

stable_blocksSSP_plot['face'].plot_psd_topomap(proj=True)
stable_blocksSSP_plot['scene'].plot_psd_topomap(proj=True)

# Plot topomap (possibiity of adding specific times)
allSubs_MNE[0][0].average().plot_topomap(proj=True)
allSubs_MNE[0][0].average().plot_topomap(proj=True,times=np.linspace(0.05, 0.15, 5))

# Plot joint topomap and evoked ERP
allSubs_MNE[0][0].average().plot_joint()

# If manually adding a sensor plot
stable_blocksSSP_plot.plot_sensors(show_names=True)

# Noise covariance plot - not really sure what to make of this (yet)
noise_cov = mne.compute_covariance(stable_blocksSSP_plot)
fig = mne.viz.plot_cov(noise_cov, stable_blocksSSP_plot.info) 

# Generate list of evoked objects from condition names
evokeds = [allSubs_MNE[0][0][name].average() for name in ('scene','face')]

colors = 'blue', 'red'
title = 'Subject \nscene vs face'

# Plot evoked across all channels, comparing two categories
mne.viz.plot_evoked_topo(evokeds, color=colors, title=title, background_color='w')

# Compare two categories
mne.viz.plot_compare_evokeds(evokeds,title=title,show_sensors=True,cmap=None,colors=['b','r'])#,ci=True)
# When multiple channels are passed, this function combines them all, to get one time course for each condition. 

mne.viz.plot_compare_evokeds(evokeds,title=title,show_sensors=True,ci=True,picks=[7],colors=['b','r'])

# Make animation
fig,anim = evokeds[0].animate_topomap(times=np.linspace(0.00, 0.79, 100),butterfly=True)
# Save animation
fig,anim = evokeds[0].animate_topomap(times=np.linspace(0.00, 0.79, 50),frame_rate=10,blit=False)
anim.save('Brainmation.gif', writer='imagemagick', fps=10)


# Sort epochs based on categories
sorted_epochsarray = [allSubs_MNE[0][0][name] for name in ('scene','face')]

# Plot image
stable_blocksSSP_plot.plot_image()

# Appending all entries in the overall epochsarray as single evoked arrays of shape (n_channels, n_times) 
g2 = allSubs_MNE[0][0].get_data()
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
# Could create these e_dicts for several people, and plot the means. Or create an e_dict with the evokeds for each person, and make the "overall" mean with individual evoked means across subjects.


mne.viz.plot_compare_evokeds(e_dict,ci=0.4,picks=[7],colors=['b','r'])#,title=title,show_sensors=True,cmap='viridis',ci=True)
mne.viz.plot_compare_evokeds(e_dict,ci=0.8,picks=[7],colors=['b','r'])#,title=title,show_sensors=True,cmap='viridis',ci=True)