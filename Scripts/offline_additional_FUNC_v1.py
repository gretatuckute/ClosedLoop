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

# FUNC v1.0 Script offline analysis FUNC V1.0 saves the clf output and alpha test values for each subject. 
# Also outputs new LORO values (5 CV folds). Training on stable and NF omitted. Adds the first 3 subjects in the same script, but uses
# 550 samples.

#%% Imports
from sklearn.metrics import confusion_matrix

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
    
    if subjID in ['07','08','11']:
        n_samples_fs500 = 550 # Number of samples to extract for each epoch, sampling frequency 500
        n_samples_fs100 = int(550/5) # Number of samples, sampling frequency 100 (resampled)
    else:
        n_samples_fs500 = 450 
        n_samples_fs100 = int(450/5) 
    
    e = extractEpochs_tmin(EEGfile,markerFile,prefilter=prefilter,marker1=0,n_samples=n_samples_fs500)
    cat = extractCat(idxFile,exp_type='fused')
    
    # MNE info files
    info_fs500 = create_info_mne(reject_ch=0,sfreq=500)
    info_fs100 = create_info_mne(reject_ch=1,sfreq=100)

    #%% Run NF RT offline analysis
    coef_lst = []
    stable_blocks0 = e[:600,:,:] # Fi+st run
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
        
        # Save clf _coefs
        coefs = clf.coef_
        coef_r = np.reshape(coefs, [23,n_samples_fs100])
        coef_lst.append(coef_r)

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
    
    d['RT_test_acc_corr'] = above_chance_offline
    d['RT_test_acc_corr_run'] = a_per_run
    d['RT_coefs'] = coef_lst

    #%% Analyze RT session block-wise
    d['ALPHA_test'] = alpha_test
    d['CLFO_test'] = clf_output_test

 
    #%% Stable blocks - train on first run.
    c_test = 0
    
    no_sb_fb = (4*n_it)-4-4 # Number stable blocks in fb run minus train run
    block_len = 50
    
    pred_prob_test = np.zeros((no_sb_fb*block_len,2)) # Prediction probability test. Block length of 50 trials
    pred_prob_test_corr = np.zeros((no_sb_fb*block_len,2)) # Prediction probability test, corrected for bias
    alpha_test = np.zeros((no_sb_fb*block_len)) # Alpha values for stable blocks
    
    stable_blocks_fbrun = np.concatenate([e[400+n*400:600+n*400] for n in range(n_it)]) # Stable blocks feedback run
    y_stable_blocks_fbrun = np.concatenate([y[400+n*400:600+n*400] for n in range(n_it)])
    
    y_pred = np.zeros(no_sb_fb*block_len) 
    
    # Only use the stable blocks from fb runs
    
    for r in range(n_it-2): # 5 runs - 1
        print('Run no: ',r)
        # First fb run
        stable_blocks_train = stable_blocks_fbrun[:400]
        y_train = y_stable_blocks_fbrun[:400]
        
        val_indices = range((r+2)*200,((r+2)*200)+200) # Validation block index
        stable_blocks_val = stable_blocks_fbrun[val_indices]
        y_val = y_stable_blocks_fbrun[val_indices]
        
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
            
        print('No c_test: ' + str(c_test))
            
    above_chance_train = len(np.where((np.array(alpha_test[:c_test])>0.5))[0])/len(alpha_test[:c_test])
    print('Above chance alpha train (corrected): ' + str(above_chance_train))    
    
    score = metrics.accuracy_score(y_stable_blocks_fbrun[400:], y_pred) 
    
    d['LORO_first_acc_corr'] = above_chance_train
    d['LORO_first_acc_uncorr'] = score    
        
   #%% Stable blocks - train on last run.
    c_test = 0
    
    pred_prob_test = np.zeros((no_sb_fb*block_len,2)) # Prediction probability test. Block length of 50 trials
    pred_prob_test_corr = np.zeros((no_sb_fb*block_len,2)) # Prediction probability test, corrected for bias
    alpha_test = np.zeros((no_sb_fb*block_len)) # Alpha values for stable blocks
    
    stable_blocks_fbrun = np.concatenate([e[400+n*400:600+n*400] for n in range(n_it)]) # Stable blocks feedback run
    y_stable_blocks_fbrun = np.concatenate([y[400+n*400:600+n*400] for n in range(n_it)])
    
    y_pred = np.zeros(no_sb_fb*block_len) 

    for r in range(n_it-2): # 5 runs - 1
        print('Run no: ',r)
        # Last fb run
        stable_blocks_train = stable_blocks_fbrun[600:]
        y_train = y_stable_blocks_fbrun[600:]
        
        val_indices = range((r)*200,((r)*200)+200) # Validation block index
        stable_blocks_val = stable_blocks_fbrun[val_indices]
        y_val = y_stable_blocks_fbrun[val_indices]
        
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
            
        print('No c_test: ' + str(c_test))
            
    above_chance_train = len(np.where((np.array(alpha_test[:c_test])>0.5))[0])/len(alpha_test[:c_test])
    print('Above chance alpha train (corrected): ' + str(above_chance_train))    
    
    score = metrics.accuracy_score(y_stable_blocks_fbrun[:600], y_pred) 
    
    d['LORO_last_acc_corr'] = above_chance_train
    d['LORO_last_acc_uncorr'] = score    
    
    #%% Save pckl file
    pkl_arr = [d]
    
    print('Finished running test and train analyses for subject: ' + str(subjID))
    
    # PICKLE TIME
    fname = '18Jun_subj_'+str(subjID)+'.pkl'
    with open(fname, 'wb') as fout:
        pickle.dump(pkl_arr, fout)
    
    
