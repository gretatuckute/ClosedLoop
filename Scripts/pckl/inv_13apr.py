# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:28:03 2019

@author: Greta
"""
import pickle
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import mne

#%% Old files

os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\pckl\\')

with open('04April_V2_subj_13.pkl', "rb") as fin:
    sub13 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_14.pkl', "rb") as fin:
    sub14 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_15.pkl', "rb") as fin:
    sub15 = (pickle.load(fin))[0] 
    
#%% New pipeline: offline_analysis_wMNE_v2.2
os.chdir('P:\\closed_loop_data\\offline_analysis_pckl\\')

with open('13April_test_13.pkl', "rb") as fin:
    sub13n = (pickle.load(fin))[0] 
    
with open('13April_test_14.pkl', "rb") as fin:
    sub14n = (pickle.load(fin))[0] 
    
with open('13April_test_15.pkl', "rb") as fin:
    sub15n = (pickle.load(fin))[0] 
    
#%% Testing how to make MNE plots for more subjects
# Extract MNE dict objects
dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
wanted_keys = ('MNE_RT_epochs_fb_avg','MNE_RT_epochs_fb_nonavg','MNE_stable_blocks_SSP')

subLst = [sub13n,sub14n,sub15n]

allSubs_MNE = []
for sub in subLst:
    result = dictfilt(sub, wanted_keys)
    print(result)
    g = list(result.values())    
    allSubs_MNE.append(g)

# CHECK THIS. Order: stable, nonavg and avg


#%% Write a function that outputs evokeds
    
def extractEvokeds(epochsArray):
    evokeds = [epochsArray[name].average() for name in ('scene','face')]
    return evokeds[0],evokeds[1] # 0 for scenes, 1 for faces


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