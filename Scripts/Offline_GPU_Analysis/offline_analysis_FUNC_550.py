#
# Author: Greta Tuckute, 18 Mar 2019
# 
# Script for analyzing RT NF accuracy (offline RT pipeline - identical to RT paradigm)
# Script for computing the training accuracy based on stable blocks 
#
# Output:
#
#
# Edits:
# 04 April 2019, new path scripts, extract 450 samples, confusion matrices for training, 
# Difference between V2 and V3 save pckl file is that in V3 all files are names RT in front, if it has something to do with the RT offline analysis.
#wMNE_v2.2. has new dict save names. Add correlation with alpha file. 

#%% Imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from sklearn import metrics
import numpy as np
from scipy.stats import zscore
import mne
import pickle
import argparse
import os

#%% Validate that these correspond to the GPU RT analysis scripts
#os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts')

from EEG_classification import sigmoid, testEpoch,trainLogReg_cross_offline,trainLogReg_cross,trainLogReg_cross2,trainLogReg
from EEG_analysis_RT import preproc1epoch, create_info_mne, applySSP,average_stable
from EEG_analysis_offline import extractEpochs_tmin,extractCat,extractAlpha,applySSP_forplot

#%%

def findSubjectFiles(subjID,manual_n_it=False):
    ''' Loads an EEG file, a marker file, a category file, and an alpha value file (real-time classifier output values) from a chosen subject.
    
    # Input
    - subjID: string, subject ID containing two digits.
    - manual_n_it: int, manual entry for number of iterations
    
    # Output
    - EEGfile, markerFile, idxFile, alphaFile: strings of the names of the files.
    - n_it: int, number of iterations. Default is 5 (corresponds to 5 neurofeedback runs)
    
    '''
    
    if subjID == '07': 
        EEGfile = 'subject_07EEG_.csv'
        markerFile = 'subject_07marker_.csv'
    
    if subjID == '08': 
        EEGfile = 'subject_08EEG_.csv'
        markerFile = 'subject_08marker_.csv'
    
    if subjID == '11': 
        EEGfile = 'subject_11EEG_.csv'
        markerFile = 'subject_11marker_.csv'

    if subjID == '13': 
        EEGfile = 'subject_13_EEG_03-21-19_11-02.csv'
        markerFile = 'subject_13_marker_03-21-19_11-02.csv'
    
    if subjID == '14': 
        EEGfile = 'subject_14_EEG_03-21-19_09-48.csv'
        markerFile = 'subject_14_marker_03-21-19_09-48.csv'
        
    if subjID == '15': 
        EEGfile = 'subject_15_EEG_03-21-19_12-43.csv'
        markerFile = 'subject_15_marker_03-21-19_12-43.csv'

    if subjID == '16': 
        EEGfile = 'subject_16_EEG_03-28-19_08-43.csv'
        markerFile = 'subject_16_marker_03-28-19_08-43.csv'
        
    if subjID == '17': 
        EEGfile = 'subject_17_EEG_03-28-19_15-37.csv'
        markerFile = 'subject_17_marker_03-28-19_15-37.csv'
    
    if subjID == '18': 
        EEGfile = 'subject_18_EEG_03-28-19_14-08.csv'
        markerFile = 'subject_18_marker_03-28-19_14-08.csv'
        
    if subjID == '19': 
        EEGfile = 'subject_19_EEG_03-21-19_14-34.csv'
        markerFile = 'subject_19_marker_03-21-19_14-34.csv'
        
    if subjID == '21': 
        EEGfile = 'subject_21_EEG_04-02-19_08-45.csv'
        markerFile = 'subject_21_marker_04-02-19_08-45.csv'

    if subjID == '22': 
        EEGfile = 'subject_22_EEG_04-04-19_09-15.csv'
        markerFile = 'subject_22_marker_04-04-19_09-15.csv'
        
    if subjID == '23': 
        EEGfile = 'subject_23_EEG_04-02-19_16-48.csv'
        markerFile = 'subject_23_marker_04-02-19_16-48.csv'
        
    if subjID == '24': 
        EEGfile = 'subject_24_EEG_04-09-19_15-45.csv'
        markerFile = 'subject_24_marker_04-09-19_15-45.csv'
    
    if subjID == '25': 
        EEGfile = 'subject_25_EEG_04-02-19_14-09.csv'
        markerFile = 'subject_25_marker_04-02-19_14-09.csv'
        
    if subjID == '26': 
        EEGfile = 'subject_26_EEG_04-09-19_08-48.csv'
        markerFile = 'subject_26_marker_04-09-19_08-48.csv'
        
    if subjID == '27': 
        EEGfile = 'subject_27_EEG_03-28-19_12-45.csv'
        markerFile = 'subject_27_marker_03-28-19_12-45.csv'
        
    if subjID == '30': 
        EEGfile = 'subject_30_EEG_04-04-19_15-58.csv'
        markerFile = 'subject_30_marker_04-04-19_15-58.csv'
        
    if subjID == '31': 
        EEGfile = 'subject_31_EEG_04-11-19_08-40.csv'
        markerFile = 'subject_31_marker_04-11-19_08-40.csv'
        
    if subjID == '32': 
        EEGfile = 'subject_32_EEG_04-11-19_15-50.csv'
        markerFile = 'subject_32_marker_04-11-19_15-50.csv'
        
    if subjID == '33': 
        EEGfile = 'subject_33_EEG_04-09-19_17-06.csv'
        markerFile = 'subject_33_marker_04-09-19_17-06.csv'
        
    if subjID == '34': 
        EEGfile = 'subject_34_EEG_04-11-19_14-29.csv'
        markerFile = 'subject_34_marker_04-11-19_14-29.csv'  
    
    idxFile = 'createIndices_' + subjID + '_day_2.csv'
    alphaFile = 'alpha_subjID_' + subjID + '.csv' 
    
    if manual_n_it is None:
        n_it = 5
    if manual_n_it is not None:
        n_it = manual_n_it
        
    print(EEGfile)
        
    return EEGfile, markerFile, idxFile, alphaFile, n_it

#data_dir = 'P:\\closed_loop_data\\'+subjID+'\\'
#os.chdir(data_dir)

def analyzeOffline(subjID):
    '''
    
    
    '''
    
    # Initialize conditions for preprocessing of epochs (preproc1epoch from EEG_analysis_RT)
    reject_ch = 1 # Rejection of nine predefined channels
    reject = None # Rejection of channels, either manually defined or based on MNE analysis
    mne_reject = 0 
    flat = None # Input for MNE rejection
    bad_channels = None # Input for manual rejection of channels
    opt_detrend = 1 # Temporal EEG detrending (linear)
    
    if reject_ch == 1:
        n_channels = 23
    if reject_ch == 0:
        n_channels = 32
    
    # Initialize conditions for SSP rejection (applySSP from EEG_analysis_RT)
    threshold = 0.1  # SSP projection variance threshold
    
    
    # Initialize dictionary for saving outputs
    d = {}
    d['subjID'] = subjID
    
    plot_MNE = 0
    
    EEGfile, markerFile, idxFile, alphaFile, n_it = findSubjectFiles(subjID,manual_n_it=None)
    
    #%% Test RT alpha per run
    alpha,marker = extractAlpha(alphaFile)
    above_chance = len(np.where((np.array(alpha)>0.5))[0])/len(alpha) 
    
    alpha_per_run = np.zeros((n_it))
    j = 0
    
    for ii in range(n_it):
        alpha_per_run[ii] = len(np.where((np.array(alpha[j:j+200])>0.5))[0])/200 
        j += 200
    
    d['ALPHA_fromfile_overall'] = above_chance
    d['ALPHA_fromfile_run'] = alpha_per_run

    #%% Extract epochs from EEG data
    prefilter = 0
    n_samples_fs500 = 550 # Number of samples to extract for each epoch, sampling frequency 500
    n_samples_fs100 = int(550/5) # Number of samples, sampling frequency 100 (resampled)
    
    e = extractEpochs_tmin(EEGfile,markerFile,prefilter=prefilter,marker1=0,n_samples=n_samples_fs500)
    cat = extractCat(idxFile,exp_type='fused')
    
    # MNE info files
    info_fs500 = create_info_mne(reject_ch=0,sfreq=500)
    info_fs100 = create_info_mne(reject_ch=1,sfreq=100)

    #%% Run NF RT offline analysis
    
    stable_blocks0 = e[:600,:,:] # First run
    stable_blocks1 = np.zeros((600,n_channels,n_samples_fs100)) 
    
    y = np.array([int(x) for x in cat])
    y_run = y[:600]
    
    pred_prob_train = np.zeros((n_it*600,2))
    pred_prob_test = np.zeros((n_it*200,2)) # Prediction probability
    pred_prob_test_corr = np.zeros((n_it*200,2)) # Prediction probability, corrected for classifier bias
    alpha_test = np.zeros((n_it*200))
    clf_output_test = np.zeros((n_it*200))
    train_acc = np.zeros(n_it)
    c_test = 0
    c = 0
    offset = 0
    y_pred_test1 = np.zeros(5*200)
    y_test_feedback = np.concatenate((y[600:800],y[1000:1200],y[1400:1600],y[1800:2000],y[2200:2400]))
    Y_train = np.zeros((n_it,600))
    Y_run = np.zeros((n_it,600))
    
    val = 1 # Offset validation
    offset_pred = 0 # Offset prediction
    offset_Pred = []
    # Score = np.zeros((n_it,3))
    epochs_fb = np.zeros((200,23,n_samples_fs100)) # Epochs feedback
    
    for b in range(n_it):
        for t in range(stable_blocks0.shape[0]):
            epoch = stable_blocks0[t,:,:]
            epoch = preproc1epoch(epoch,info_fs500,SSP=False,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
            stable_blocks1[t+offset,:,:] = epoch
            c += 1
        projs1,stable_blocksSSP1 = applySSP(stable_blocks1,info_fs100,threshold=threshold) # Apply SSP on stable blocks
        
        # Average training blocks after SSP
        stable_blocksSSP1 = average_stable(stable_blocksSSP1)
    
        if val == 1: # Test for offset classifier bias
            if b > 0: # Runs after first run
                stable_blocksSSP1_append = np.append(stable_blocksSSP1,epochs_fb,axis=0)
                fb_start = (b-1)*200
                y_run_append = np.append(y_run,y_test_feedback[fb_start:fb_start+200],axis=0)
                clf,offset_pred = trainLogReg_cross2(stable_blocksSSP1_append,y_run_append)
            else:
                clf,offset_pred = trainLogReg_cross(stable_blocksSSP1,y_run) # First run
                
            offset_Pred.append(offset_pred)
            #Score[b,:]=score
            
            print('Offset estimated to '+str(offset_pred))
            
        else:
            clf = trainLogReg(stable_blocksSSP1,y_run)
    
        Y_run[b,:]=y_run
        y_pred_train1 = np.zeros(len(y_run))
        offset_pred = np.min([np.max([offset_pred,-0.25]),0.25])/2 # Limit offset correction to abs(0.125)
        
        # Training accuracy based on the RT classifier
        for t in range(len(y_run)):
            epoch_t = stable_blocksSSP1[t,:,:]
            pred_prob_train[t,:],y_pred_train1[t] = testEpoch(clf,epoch_t)
            
        Y_train[b,:]=y_pred_train1
        train_acc[b] = len(np.where(np.array(y_pred_train1[0:len(y_run)])==np.array(y_run))[0])/len(y_run)
        
        stable_blocks1 = stable_blocks1[200:,:,:]
        stable_blocks1 = np.concatenate((stable_blocks1,np.zeros((200,n_channels,n_samples_fs100))),axis=0)
        
        s_begin = 800+b*400
        offset = 400 # Indexing offset for EEG data and y
        stable_blocks0 = e[s_begin:s_begin+200,:,:]
        y_run = np.concatenate((y_run[200:],y[s_begin:s_begin+200]))
        
        # Test accuracy of RT epochs
        for t in range(200):
            print('Testing epoch number: ',c)
            
            epoch = e[c,:,:]
            epoch = preproc1epoch(epoch,info_fs500,projs=projs1,SSP=True,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
            
            if t > 0:
                epoch = (epoch+epoch_prev)/2
            
            pred_prob_test[c_test,:],y_pred_test1[c_test] = testEpoch(clf,epoch)
    
            # Correct the prediction bias offset
            pred_prob_test_corr[c_test,0] = np.min([np.max([pred_prob_test[c_test,0]+offset_pred,0]),1]) 
            pred_prob_test_corr[c_test,1] = np.min([np.max([pred_prob_test[c_test,1]-offset_pred,0]),1])
    
            clf_output = pred_prob_test_corr[c_test,int(y[c])]-pred_prob_test_corr[c_test,int(y[c]-1)]
            clf_output_test[c_test] = clf_output # Save classifier output for correlation checks
            alpha_test[c_test] = sigmoid(clf_output) # Convert corrected classifier output to an alpha value using a sigmoid transfer function
            
            epoch_prev = epoch
    
            epochs_fb[t,:,:] = epoch
            c += 1
            c_test += 1
        
    above_chance_offline = len(np.where((np.array(alpha_test[:c_test])>0.5))[0])/len(alpha_test[:c_test])
    print('Above chance alpha (corrected): ' + str(above_chance_offline))
    
    score = metrics.accuracy_score(y_test_feedback[:c_test], y_pred_test1[:c_test]) 
    print('Accuracy score (uncorrected): ' + str(score))
    
    # Test score per run
    a_per_run = np.zeros((n_it))
    score_per_run = np.zeros((n_it))
    j = 0
    
    for run in range(n_it):
        score_per_run[run] = metrics.accuracy_score(y_test_feedback[run*200:(run+1)*200], y_pred_test1[run*200:(run+1)*200]) 
        a_per_run[run] = (len(np.where((np.array(alpha_test[j:j+200])>0.5))[0])/200)
        j += 200
    
    print('Alpha, corrected, above chance per run: ' + str(a_per_run))
    print('Score, uncorrected, per run: ' + str(score_per_run))
    
    
    d['RT_train_acc'] = train_acc
    d['RT_test_acc_corr'] = above_chance_offline
    d['RT_test_acc_corr_run'] = a_per_run
    d['RT_test_acc_uncorr'] = score
    d['RT_test_acc_uncorr_run'] = score_per_run

    #%% Compare RT alpha with RT offline alpha
    alphan = np.asarray(alpha)
    alpha_corr = (np.corrcoef(alphan[1:], alpha_test[:999]))[0][1] # Shifted with one
    d['ALPHA_correlation'] = alpha_corr
    if alpha_corr >= 0.98:
        d['GROUP'] = 1 # 1 for NF group
    if alpha_corr < 0.98:
        d['GROUP'] = 0 # 0 for control group
        
    # Classifier output correlation has to be checked offline, because clf_output values are computed offline (RT pipeline)
    #np.corrcoef(clf_output_test, clf_output_test13)
    
    #plt.plot(alphan)
    #plt.plot(alpha_test)
    #
    ## Compare alphas from file and offline classification 
    #plt.plot(np.arange(999),alphan[1:]) #blue, en foran 
    #plt.plot(np.arange(1000),alpha_test)
    #
    ## Run 1
    #plt.plot(np.arange(200),alphan[1:201]) #starts at 0.5
    #plt.plot(np.arange(200),alpha_test[0:200]) #matches


    #%% Confusion matrices
    n_it_trials = n_it*200
    
    correct = (y_test_feedback[:n_it_trials]==y_pred_test1[:n_it_trials])
    
    alpha_test_c = np.copy(alpha_test)
    alpha_test_c[alpha_test_c > 0.5] = True # 1 is a correctly predicted 
    alpha_test_c[alpha_test_c < 0.5] = False
    
    alpha_predcat = np.argmax(pred_prob_test_corr,axis=1)
            
    conf_uncorr = confusion_matrix(y_test_feedback[:n_it_trials],y_pred_test1[:n_it_trials])
    conf_corr = confusion_matrix(y_test_feedback[:n_it_trials],alpha_predcat[:n_it_trials])
    
    # Separate into scenes and faces accuracy  
    scene_acc = conf_corr[0,0]/(conf_corr[0,0]+conf_corr[0,1])
    face_acc = conf_corr[1,1]/(conf_corr[1,0]+conf_corr[1,1])
    
    d['RT_correct_NFtest_pred'] = correct
    d['RT_conf_corr'] = conf_corr
    d['RT_conf_uncorr'] = conf_uncorr
    
    d['RT_scene_acc'] = scene_acc
    d['RT_face_acc'] = face_acc
    
    # Training confusion matrices
    conf_train = []
    for b in range(n_it):
        conf_train.append(confusion_matrix(Y_run[b,:], Y_train[b,:]))
        
    d['RT_conf_train'] = conf_train

    #%% Training on stable blocks only - leave one block out CV
    offset_pred_lst = []
    c_test = 0
    
    no_sb = 8+4*n_it # Number stable blocks
    block_len = 50
    
    pred_prob_test = np.zeros((no_sb*block_len,2)) # Prediction probability test. Block length of 50 trials
    pred_prob_test_corr = np.zeros((no_sb*block_len,2)) # Prediction probability test, corrected for bias
    alpha_test = np.zeros((no_sb*block_len)) # Alpha values for stable blocks
    
    stable_blocks_fbrun = np.concatenate([e[400+n*400:600+n*400] for n in range(n_it)]) # Stable blocks feedback run
    y_stable_blocks_fbrun = np.concatenate([y[400+n*400:600+n*400] for n in range(n_it)])
    
    stable_blocks = np.concatenate((e[:400,:,:], stable_blocks_fbrun))
    y_stable_blocks = np.concatenate((y[:400], y_stable_blocks_fbrun))
    y_pred = np.zeros(no_sb*block_len) 

    for sb in range(no_sb):
        val_indices = range(sb*block_len,(sb+1)*block_len) # Validation block index
        stable_blocks_val = stable_blocks[val_indices]
        y_val = y_stable_blocks[val_indices]
        
        stable_blocks_train = np.delete(stable_blocks, val_indices, axis=0)
        y_train = np.delete(y_stable_blocks, val_indices)
        stable_blocks_train_prep = np.zeros((len(y_train),23,n_samples_fs100))
        
        for t in range(stable_blocks_train.shape[0]):
            epoch = stable_blocks_train[t,:,:]
            epoch = preproc1epoch(epoch,info_fs500,SSP=False,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
            stable_blocks_train_prep[t,:,:] = epoch
    
        projs1,stable_blocksSSP_train = applySSP(stable_blocks_train_prep,info_fs100,threshold=threshold)
        
        # Average after SSP correction
        stable_blocksSSP_train = average_stable(stable_blocksSSP_train)
        clf,offset_pred = trainLogReg_cross_offline(stable_blocksSSP_train,y_train) #cur. in EEG_classification
        offset_pred = np.min([np.max([offset_pred,-0.25]),0.25])/2
        offset_pred_lst.append(offset_pred)
        
        # Test epochs in validation block. Preprocessing and testing epoch-wise
        for t in range(block_len):
            epoch = stable_blocks_val[t,:,:]
            epoch = preproc1epoch(epoch,info_fs500,projs=projs1,SSP=True,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
            
            if t > 0:
                epoch = (epoch+epoch_prev)/2
            
            pred_prob_test[c_test,:],y_pred[c_test] = testEpoch(clf,epoch)
            
            # Correct the prediction bias offset
            pred_prob_test_corr[c_test,0] = np.min([np.max([pred_prob_test[c_test,0]+offset_pred,0]),1]) 
            pred_prob_test_corr[c_test,1] = np.min([np.max([pred_prob_test[c_test,1]-offset_pred,0]),1])
            
            clf_output = pred_prob_test_corr[c_test,int(y_val[t])]-pred_prob_test_corr[c_test,int(y_val[t]-1)]
            alpha_test[c_test] = sigmoid(clf_output)
            
            epoch_prev = epoch
            
            c_test += 1
            
        print('No c_test: ' + str(c_test) + 'out of ' + str(no_sb*block_len))
        
    above_chance_train = len(np.where((np.array(alpha_test[:c_test])>0.5))[0])/len(alpha_test[:c_test])
    print('Above chance alpha train (corrected): ' + str(above_chance_train))    
    
    score = metrics.accuracy_score(y_stable_blocks, y_pred) 
    
    d['LOBO_stable_train_offsets'] = offset_pred_lst
    d['LOBO_stable_train_acc_corr'] = above_chance_train
    d['LOBO_stable_train_acc_uncorr'] = score

    #%% Extract data for MNE plots
    
    # Perform preprocessing and SSP on all the stable blocks.
    stable_blocks_plot = np.zeros((len(y_stable_blocks),n_channels,n_samples_fs100)) 
    
    for t in range(stable_blocks.shape[0]):
        epoch = stable_blocks[t,:,:]
        epoch = preproc1epoch(epoch,info_fs500,SSP=False,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
        stable_blocks_plot[t,:,:] = epoch
    
    projs1,stable_blocksSSP_plot,p_variance = applySSP_forplot(stable_blocks_plot,info_fs100,threshold=threshold,add_events=y_stable_blocks)
    
    d['MNE_stable_blocks_SSP'] = stable_blocksSSP_plot
    d['MNE_stable_blocks_SSP_projvariance'] = p_variance
    d['MNE_y_stable_blocks'] = y_stable_blocks


    if plot_MNE == True:
        print('Making a lot of MNE plots!')
    
        #%% Adding events manually to epochs *also implemented in applySSP_forplot*
        stable_blocksSSP_get = stable_blocksSSP_plot.get_data()
        events_list = y_stable_blocks
        event_id = dict(scene=0, face=1)
        n_epochs = len(events_list)
        events_list = [int(i) for i in events_list]
        events = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int),events_list]
        
        epochs_events = mne.EpochsArray(stable_blocksSSP_get, info_fs100, events=events, tmin=-0.1, event_id=event_id,baseline=None)
        
        epochs_events['face'].average().plot()
        epochs_events['scene'].average().plot()
        
        #%%
        # Overall average plot, all channels 
        stable_blocksSSP_plot.average().plot(spatial_colors=True, time_unit='s')#,picks=[7]) 
        
        # Plot based on categories
        stable_blocksSSP_plot['face'].average().plot(spatial_colors=True, time_unit='s')#,picks=[7])
        stable_blocksSSP_plot['scene'].average().plot(spatial_colors=True, time_unit='s')
        
        # Plot of the SSP projectors
        stable_blocksSSP_plot.average().plot_projs_topomap()
        # Consider adding p_variance values to the plot.
        
        
        # Plot the topomap of the power spectral density across epochs.
        stable_blocksSSP_plot.plot_psd_topomap(proj=True)
        
        stable_blocksSSP_plot['face'].plot_psd_topomap(proj=True)
        stable_blocksSSP_plot['scene'].plot_psd_topomap(proj=True)
        
        # Plot topomap (possibiity of adding specific times)
        stable_blocksSSP_plot.average().plot_topomap(proj=True)
        stable_blocksSSP_plot.average().plot_topomap(proj=True,times=np.linspace(0.05, 0.15, 5))
        
        # Plot joint topomap and evoked ERP
        stable_blocksSSP_plot.average().plot_joint()
        
        # If manually adding a sensor plot
        stable_blocksSSP_plot.plot_sensors(show_names=True)
        
        # Noise covariance plot - not really sure what to make of this (yet)
        noise_cov = mne.compute_covariance(stable_blocksSSP_plot)
        fig = mne.viz.plot_cov(noise_cov, stable_blocksSSP_plot.info) 
        
        # Generate list of evoked objects from condition names
        evokeds = [stable_blocksSSP_plot[name].average() for name in ('scene','face')]
        
        colors = 'blue', 'red'
        title = 'Subject \nscene vs face'
        
        # Plot evoked across all channels, comparing two categories
        mne.viz.plot_evoked_topo(evokeds, color=colors, title=title, background_color='w')
        
        # Compare two categories
        mne.viz.plot_compare_evokeds(evokeds,title=title,show_sensors=True,cmap='viridis')#,ci=True)
        # When multiple channels are passed, this function combines them all, to get one time course for each condition. 
        
        mne.viz.plot_compare_evokeds(evokeds,title=title,show_sensors=True,cmap='viridis',ci=True,picks=[7])
        
        # Make animation
        fig,anim = evokeds[0].animate_topomap(times=np.linspace(0.00, 0.79, 100),butterfly=True)
        # Save animation
        fig,anim = evokeds[0].animate_topomap(times=np.linspace(0.00, 0.79, 50),frame_rate=10,blit=False)
        anim.save('Brainmation.gif', writer='imagemagick', fps=10)
        
        
        # Sort epochs based on categories
        sorted_epochsarray = [stable_blocksSSP_plot[name] for name in ('scene','face')]
        
        # Plot image
        stable_blocksSSP_plot.plot_image()
        
        # Appending all entries in the overall epochsarray as single evoked arrays of shape (n_channels, n_times) 
        g2 = stable_blocksSSP_plot.get_data()
        evoked_array = [mne.EvokedArray(entry, info_fs100,tmin=-0.1) for entry in g2]
        
        # Appending all entries in the overall epochsarray as single evoked arrays of shape (n_channels, n_times) - category info added as comments
        evoked_array2 = []
        for idx,cat in enumerate(events_list):
            evoked_array2.append(mne.EvokedArray(g2[idx], info_fs100,tmin=-0.1,comment=cat))
            
        mne.viz.plot_compare_evokeds(evoked_array2[:10]) # Plotting all the individual evoked arrays (up to 10)
        
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
        
        
        mne.viz.plot_compare_evokeds(e_dict,ci=0.4,picks=[7])#,title=title,show_sensors=True,cmap='viridis',ci=True)
        mne.viz.plot_compare_evokeds(e_dict,ci=0.8,picks=[7])#,title=title,show_sensors=True,cmap='viridis',ci=True)
        
        # Comparing
        mne.viz.plot_compare_evokeds(evokeds,title=title,show_sensors=True,picks=[7],ci=False)
        mne.viz.plot_compare_evokeds(e_dict,title=title,show_sensors=True,picks=[7],ci=True)#,,ci=True)
        
        # Based on categories, the correct epochs are added to the evoked_array_c0 and c1
    
        #%% Manually check whether the MNE events correspond to the actual categories
        
        g1 = stable_blocksSSP_plot.get_data()
        
        g3 = g1[y_stable_blocks==True] # Faces = 1
        g4 = g1[y_stable_blocks==False] # Scenes = 0
        
        g3a = np.mean(g3,axis=0)
        g4a = np.mean(g4,axis=0)
        
        plt.figure(4)
        plt.plot(g3a.T[:,7]) # Corresponds to: stable_blocksSSP_plot['face'].average().plot(spatial_colors=True, time_unit='s')
        plt.plot(g4a.T[:,7])
        
        epochsg3 = mne.EpochsArray(g3, info=info_fs100)
        epochsg4 = mne.EpochsArray(g4, info=info_fs100)
        
        plt.figure(6)
        epochsg3.average().plot(spatial_colors=True, time_unit='s',picks=[7]) # Corresponds to: stable_blocksSSP_plot['face'].average().plot(spatial_colors=True, time_unit='s',picks=[7])
        epochsg4.average().plot(spatial_colors=True, time_unit='s',picks=[7]) 
        
        # Plot topomap
        a3 = epochsg3.average()
        a3.plot_topomap(times=np.linspace(0.05, 0.15, 5))
        
        # Plot of evoked ERP and topomaps, in one plot!
        a3.plot_joint()
        
        # Control the y axis
        a3.plot(ylim=dict(eeg=[-2000000, 2000000]))
        a3.plot(ylim=dict(eeg=[-2000000, 2000000]))
    
    #%% Confusion matrices - stable blocks accuracy, LOBO
    
    # y_stable_blocks has the correct y vals, y_pred has the predicted vals. For uncorrected prediction.
    
    # Uncorrected
    correct = (y_stable_blocks == y_pred)
    conf_train_stable_uncorr = confusion_matrix(y_stable_blocks,y_pred)
    
    # Separate into scenes and faces accuracy  
    scene_acc_uncorr = conf_train_stable_uncorr[0,0]/(conf_train_stable_uncorr[0,0]+conf_train_stable_uncorr[0,1])
    face_acc_uncorr = conf_train_stable_uncorr[1,1]/(conf_train_stable_uncorr[1,0]+conf_train_stable_uncorr[1,1])
    
    # Corrected
    alpha_test_c = np.copy(alpha_test)
    alpha_test_c[alpha_test_c > 0.5] = True # 1 is a correctly predicted 
    alpha_test_c[alpha_test_c < 0.5] = False
    
    alpha_predcat = np.argmax(pred_prob_test_corr,axis=1)
    conf_train_stable = confusion_matrix(y_stable_blocks,alpha_predcat)
    
    # Separate into scenes and faces accuracy  
    scene_acc = conf_train_stable[0,0]/(conf_train_stable[0,0]+conf_train_stable[0,1])
    face_acc = conf_train_stable[1,1]/(conf_train_stable[1,0]+conf_train_stable[1,1])
    
    d['LOBO_stable_conf_uncorr'] = conf_train_stable_uncorr
    d['LOBO_stable_scene_acc_uncorr'] = scene_acc_uncorr
    d['LOBO_stable_face_acc_uncorr'] = face_acc_uncorr
    
    d['LOBO_stable_conf_corr'] = conf_train_stable
    d['LOBO_stable_scene_acc_corr'] = scene_acc
    d['LOBO_stable_face_acc_corr'] = face_acc
    
    #%% Training accuracy, training on stable and NF - leave one block out CV. Accuracy can be based on either stable+NF, only stable or only NF blocks
    offset_pred_lst = []
    c_test = 0
    
    no_b = 8+8*n_it # Number blocks total
    pred_prob_test = np.zeros((no_b*block_len,2))
    pred_prob_test_corr = np.zeros((no_b*block_len,2))
    alpha_test = np.zeros((no_b*block_len))
    y_pred = np.zeros(no_b*block_len) 
    
    for b in range(no_b):
        val_indices = range(b*block_len,(b+1)*block_len)
        blocks_val = e[val_indices]
        y_val = y[val_indices]
        
        blocks_train = np.delete(e, val_indices,axis=0)
        y_train = np.delete(y, val_indices)
        blocks_train_prep = np.zeros((len(y_train),23,n_samples_fs100))
        
        for t in range(blocks_train_prep.shape[0]):
            epoch = blocks_train[t,:,:]
            epoch = preproc1epoch(epoch,info_fs500,SSP=False,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
            blocks_train_prep[t,:,:] = epoch
    
        projs1,blocksSSP_train = applySSP(blocks_train_prep,info_fs100,threshold=threshold)
        
        # Average after SSP correction
        blocksSSP_train = average_stable(blocksSSP_train)
        clf,offset_pred = trainLogReg_cross_offline(blocksSSP_train,y_train) #cur. in EEG_classification
        offset_pred = np.min([np.max([offset_pred,-0.25]),0.25])/2
        offset_pred_lst.append(offset_pred)
        
        # Test epochs in left out block
        for t in range(block_len):
            epoch = blocks_val[t,:,:]
            epoch = preproc1epoch(epoch,info_fs500,projs=projs1,SSP=True,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
            
            if t > 0:
                epoch = (epoch+epoch_prev)/2
                
            pred_prob_test[c_test,:],y_pred[c_test] = testEpoch(clf,epoch)
            
            # Correct the prediction bias offset
            pred_prob_test_corr[c_test,0] = np.min([np.max([pred_prob_test[c_test,0]+offset_pred,0]),1]) 
            pred_prob_test_corr[c_test,1] = np.min([np.max([pred_prob_test[c_test,1]-offset_pred,0]),1])
            
            clf_output = pred_prob_test_corr[c_test,int(y_val[t])]-pred_prob_test_corr[c_test,int(y_val[t]-1)]
            alpha_test[c_test] = sigmoid(clf_output)
            
            epoch_prev = epoch
            
            c_test += 1
            
        print('No c_test: ' + str(c_test) + ' out of ' + str(no_b*block_len))
            
    above_chance_train = len(np.where((np.array(alpha_test[:c_test])>0.5))[0])/len(alpha_test[:c_test])
    print('Above chance alpha train (corrected) on both stable and NF: ' + str(above_chance_train))
    
    # Separate in stable only, and stable + NF
    e_mock = np.arange((8+n_it*8)*block_len)
    stable_blocks_fbrun = np.concatenate([e_mock[400+n*400:600+n*400] for n in range(n_it)]) # Stable blocks feedback run
    stable_blocks_idx = np.concatenate((e_mock[:400],stable_blocks_fbrun))
    
    a = alpha_test[stable_blocks_idx] 
    
    above_chance_stable = len(np.where((np.array(a)>0.5))[0])/len(a)
    print('Above chance alpha train (corrected) on stable blocks: ' + str(above_chance_stable))
    
    #nf_blocks_idx = np.concatenate([e_mock[600+n*400:800+n*400] for n in range(n_it)]) # Neurofeedback blocks 
    #a2 = alpha_test[nf_blocks_idx] 
    #above_chance_nf = len(np.where((np.array(a2)>0.5))[0])/len(a2)
    #print('Above chance alpha train (corrected) on NF blocks: ' + str(above_chance_nf))
    
    d['LOBO_all_train_offsets'] = offset_pred_lst
    d['LOBO_all_train_acc'] = above_chance_train # All blocks included
    d['LOBO_all_train_acc_stable_test'] = above_chance_stable # Trained on both stable and NF, only tested on stable
    #d['train_acc_nf_test'] = above_chance_nf # Trained on both stable and NF, only tested on NF
    
    #%% Confusion matrices - training on stable+NF blocks, testing on stable+NF, stable or NF blocks
    
    # Uncorrected, all (stable + NF)
    correct = (y == y_pred)
    conf_train_all_uncorr = confusion_matrix(y,y_pred)
    
    # Separate into scenes and faces accuracy  
    scene_acc_uncorr = conf_train_all_uncorr[0,0]/(conf_train_all_uncorr[0,0]+conf_train_all_uncorr[0,1])
    face_acc_uncorr = conf_train_all_uncorr[1,1]/(conf_train_all_uncorr[1,0]+conf_train_all_uncorr[1,1])
    
    # Corrected, all
    alpha_test_c = np.copy(alpha_test)
    alpha_test_c[alpha_test_c > 0.5] = True # 1 is a correctly predicted 
    alpha_test_c[alpha_test_c < 0.5] = False
    
    alpha_predcat = np.argmax(pred_prob_test_corr,axis=1)
    conf_train_all = confusion_matrix(y,alpha_predcat)
    
    # Separate into scenes and faces accuracy  
    scene_acc = conf_train_all[0,0]/(conf_train_all[0,0]+conf_train_all[0,1])
    face_acc = conf_train_all[1,1]/(conf_train_all[1,0]+conf_train_all[1,1])
    
    d['LOBO_all_conf_uncorr'] = conf_train_all_uncorr
    d['LOBO_all_scene_acc_uncorr'] = scene_acc_uncorr
    d['LOBO_all_face_acc_uncorr'] = face_acc_uncorr
    
    d['LOBO_all_conf_corr'] = conf_train_all
    d['LOBO_all_scene_acc_corr'] = scene_acc
    d['LOBO_all_face_acc_corr'] = face_acc
    
    #%% Training on stable blocks only - leave one run out CV
    offset_pred_lst = []
    c_test = 0
    
    no_sb = 8+4*n_it # Number stable blocks
    block_len = 50
    
    pred_prob_test = np.zeros((no_sb*block_len,2)) # Prediction probability test. Block length of 50 trials
    pred_prob_test_corr = np.zeros((no_sb*block_len,2)) # Prediction probability test, corrected for bias
    alpha_test = np.zeros((no_sb*block_len)) # Alpha values for stable blocks
    
    stable_blocks_fbrun = np.concatenate([e[400+n*400:600+n*400] for n in range(n_it)]) # Stable blocks feedback run
    y_stable_blocks_fbrun = np.concatenate([y[400+n*400:600+n*400] for n in range(n_it)])
    
    stable_blocks = np.concatenate((e[:400,:,:], stable_blocks_fbrun))
    y_stable_blocks = np.concatenate((y[:400], y_stable_blocks_fbrun))
    
    y_pred = np.zeros(no_sb*block_len) 
    
    for r in range(n_it+1): # 6 runs
        print('Run no: ',r)
        if r == 0: # First run
            val_indices = range(0,400) # First run
            stable_blocks_val = stable_blocks[val_indices]
            y_val = y_stable_blocks[val_indices]
            
        if r > 0: 
            val_indices = range((r+1)*200,((r+1)*200)+200) # Validation block index
            stable_blocks_val = stable_blocks[val_indices]
            y_val = y_stable_blocks[val_indices]
        
        stable_blocks_train = np.delete(stable_blocks, val_indices, axis=0)
        y_train = np.delete(y_stable_blocks, val_indices)
        stable_blocks_train_prep = np.zeros((len(y_train),23,n_samples_fs100))
        
        for t in range(stable_blocks_train.shape[0]):
            epoch = stable_blocks_train[t,:,:]
            epoch = preproc1epoch(epoch,info_fs500,SSP=False,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
            stable_blocks_train_prep[t,:,:] = epoch
    
        projs1,stable_blocksSSP_train = applySSP(stable_blocks_train_prep,info_fs100,threshold=threshold)
        
        # Average after SSP correction
        stable_blocksSSP_train = average_stable(stable_blocksSSP_train)
        clf,offset_pred = trainLogReg_cross_offline(stable_blocksSSP_train,y_train) #cur. in EEG_classification
        offset_pred = np.min([np.max([offset_pred,-0.25]),0.25])/2
        offset_pred_lst.append(offset_pred)
        
        # Test epochs in validation run. Preprocessing and testing epoch-wise
        for t in range(len(val_indices)):
            epoch = stable_blocks_val[t,:,:]
            epoch = preproc1epoch(epoch,info_fs500,projs=projs1,SSP=True,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
            
            if t > 0:
                epoch = (epoch+epoch_prev)/2
            
            pred_prob_test[c_test,:],y_pred[c_test] = testEpoch(clf,epoch)
            
            # Correct the prediction bias offset
            pred_prob_test_corr[c_test,0] = np.min([np.max([pred_prob_test[c_test,0]+offset_pred,0]),1]) 
            pred_prob_test_corr[c_test,1] = np.min([np.max([pred_prob_test[c_test,1]-offset_pred,0]),1])
            
            clf_output = pred_prob_test_corr[c_test,int(y_val[t])]-pred_prob_test_corr[c_test,int(y_val[t]-1)]
            alpha_test[c_test] = sigmoid(clf_output)
            
            epoch_prev = epoch
            
            c_test += 1
            
        print('No c_test: ' + str(c_test) + ' out of ' + str(no_sb*block_len))
            
    above_chance_train = len(np.where((np.array(alpha_test[:c_test])>0.5))[0])/len(alpha_test[:c_test])
    print('Above chance alpha train (corrected): ' + str(above_chance_train))    
    
    score = metrics.accuracy_score(y_stable_blocks, y_pred) 
    
    d['LORO_stable_train_offsets_stable'] = offset_pred_lst
    d['LORO_stable_acc_corr'] = above_chance_train
    d['LORO_stable_acc_uncorr'] = score    
        
    #%% Extract RT epochs (non-averaged) for plots and analysis
    stable_blocks0 = e[:600,:,:] # First run
    stable_blocks1 = np.zeros((600,n_channels,n_samples_fs100)) 
    
    y = np.array([int(x) for x in cat])
    y_run = y[:600]
    
    c_test = 0
    c = 0
    offset = 0
    y_pred_test1 = np.zeros(5*200)
    y_test_feedback = np.concatenate((y[600:800],y[1000:1200],y[1400:1600],y[1800:2000],y[2200:2400]))
    
    epochs_fb_nonavg = np.zeros((1000,23,n_samples_fs100)) # Epochs feedback
#    epochs_fb_avg = np.zeros((1000,23,n_samples_fs100)) # Epochs feedback
    
    for b in range(n_it):
        for t in range(stable_blocks0.shape[0]):
            epoch = stable_blocks0[t,:,:]
            epoch = preproc1epoch(epoch,info_fs500,SSP=False,reject=None,mne_reject=0,reject_ch=reject_ch,flat=None,bad_channels=None,opt_detrend=1)
            stable_blocks1[t+offset,:,:] = epoch
            c += 1
            
        projs1,stable_blocksSSP1 = applySSP(stable_blocks1,info_fs100,threshold=threshold) # Apply SSP on stable blocks
            
        stable_blocks1 = stable_blocks1[200:,:,:]
        stable_blocks1 = np.concatenate((stable_blocks1,np.zeros((200,n_channels,n_samples_fs100))),axis=0)
        
        s_begin = 800+b*400
        offset = 400 # Indexing offset for EEG data and y
        stable_blocks0 = e[s_begin:s_begin+200,:,:]
        y_run = np.concatenate((y_run[200:],y[s_begin:s_begin+200]))
        
        # Append RT epochs 
        for t in range(200):
            print('Epoch number: ',c)
            epoch1 = e[c,:,:]
            epoch1 = preproc1epoch(epoch1,info_fs500,projs=projs1,SSP=True,reject=reject,mne_reject=mne_reject,reject_ch=reject_ch,flat=flat,bad_channels=bad_channels,opt_detrend=opt_detrend)
            
#            For averaging epochs:
#            if t == 0:
#                epoch_avg = epoch1
#            
#            if t > 0:
#                epoch_avg = (epoch1+epoch_prev)/2 # Checked again 17 April that this is legit
#            
#            epoch_prev = epoch_avg
#    
#            epochs_fb_avg[c_test,:,:] = epoch_avg
            
            epochs_fb_nonavg[c_test,:,:] = epoch1
    
            c += 1
            c_test += 1
        
            
    # Create MNE objects
    events_list = y_test_feedback
    event_id = dict(scene=0, face=1)
    n_epochs = len(events_list)
    events_list = [int(i) for i in events_list]
    events = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int),events_list]
    
    eRT_nonavg = mne.EpochsArray(epochs_fb_nonavg, info=info_fs100, events=events,event_id=event_id,tmin=-0.1,baseline=None)
#    eRT_avg = mne.EpochsArray(epochs_fb_avg, info=info_fs100, events=events,event_id=event_id,tmin=-0.1,baseline=None)
    
    d['MNE_RT_epochs_fb_nonavg'] = eRT_nonavg
#    d['MNE_RT_epochs_fb_avg'] = eRT_avg
    d['MNE_y_test_feedback'] = y_test_feedback
    
    
    if plot_MNE == True:
        # Creating a dict of lists: Condition 0 and condition 1 with evoked arrays.
        evoked_array_c0 = []
        evoked_array_c1 = []
        
        eRT_get = eRT_nonavg.get_data()
        
        for idx,cat in enumerate(events_list):
            if cat == 0:
                evoked_array_c0.append(mne.EvokedArray(eRT_get[idx], info_fs100,tmin=-0.1,comment=cat)) # Scenes 0
                print
            if cat == 1:
                evoked_array_c1.append(mne.EvokedArray(eRT_get[idx], info_fs100,tmin=-0.1,comment=cat)) # Faces 1
        
        e_dict={}
        e_dict['0'] = evoked_array_c0
        e_dict['1'] = evoked_array_c1
        
        #colors = 'red', 'blue'
        #mne.viz.plot_compare_evokeds(e_dict,ci=0.95,picks=[7],colors=colors)
    
    #%% Save pckl file
    pkl_arr = [d]
    
    print('Finished running test and train analyses for subject: ' + str(subjID))
    
    # PICKLE TIME
    fname = '18April_550_subj_'+str(subjID)+'.pkl'
    with open(fname, 'wb') as fout:
        pickle.dump(pkl_arr, fout)
    
    
