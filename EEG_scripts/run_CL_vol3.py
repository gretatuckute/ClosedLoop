# Imports
import numpy as np
import mne
import pylsl
import warnings
from pylsl import StreamOutlet,StreamInfo,resolve_byprop,local_clock
import time
import os 

os.chdir('C:\\Users\\nicped\\Documents\\GitLab\\project\\Python_Scripts')
from EEG_functions_CL import preproc1epoch, create_info_mne, computeSSP,applySSP,removeEpochs,average_stable
from EEG_preproc_overview import extractCat
from EEG_classification import sigmoid, trainLogReg,testEpoch,trainLogReg_cross,trainLogReg_cross2
from stream_functions import *
gamer_dir = "C:\\Users\\nicped\\Documents\\GitLab\\project\\SUBJECTS\\"+subject_id+"\\"

# Manual input of subject ID for streaming and logging
subject_id = '19' 

# Extract experimental categories 
y = extractCat(gamer_dir+"createIndices_"+subject_id+"_day_2"+".csv",exp_type='fused')
y = np.array([int(x) for x in y])

#%% Tracking of print outputs. Saved as a log file to the subject ID folder
Transcript.start(gamer_dir+'stream_logfile_subject'+subject_id+time.strftime('%m-%d-%y_%H-%M')+'.log')

#%% Look for a recent stream from the Psychopy experimental script
marker_not_present = 1 # Whether to look for a marker stream from other scripts
timeout = 1 # Time to look for stream in seconds 

while marker_not_present:
    t = local_clock()
    streams = resolve_byprop('type', 'Markers', timeout=timeout)
    for i in range (len(streams)):
        lsl_created = (streams[i].created_at())
        if np.abs(lsl_created - t) < 20: # Make sure that the stream has been created within the last 20 seconds
            marker_not_present = 0
            print("Stream from Psychopy experimental script found")

#%% Start sampling, preprocessing and analyzing EEG data real-time in three phases:
# 1) Stable (EEG data during stable blocks used for training the decoding classifier. Size: 600 most recent stable trials)
# 2) Train (training of the decoding classifier, based on the specific run number)
# 3) Feedback 


fs=500
inlet_EEG,store_EEG=read_EEG_stream(fs=fs,max_buf=2)
inlet_marker,store_marker=read_marker_stream(stream_name ='PsychopyExperiment20')#'MyMarkerStream3'
info_outlet = StreamInfo('alphaStream','Markers',1,0,'float32','myuniquesourceid23441')
outlet_alpha =StreamOutlet(info_outlet)
info_fs500=create_info_mne(reject_ch=0,sfreq=500)
info_fs100=create_info_mne(reject_ch=1,sfreq=100)

look_for_trigger=1
C=1
t_latency=0.05 # s
baseline_tmin=-0.1
tmax=0.800
fs_new=100
n_time=int((tmax-baseline_tmin)*fs_new)
state='stable'
excess_EEG=[]
excess_EEG_time=[]
excess_marker=[]
excess_marker_time=[]
clear_stream(inlet_marker)
clear_stream(inlet_EEG)
time.sleep(1.1) # wait one epoch
n_channels=23
stable_blocks=np.zeros((600,n_channels,n_time))
stable_blocks0=np.zeros((200,n_channels,n_time))
marker_all=[]
n_run=0
reject=None#dict(eeg=100)
flat=None#dict(eeg=5)
threshold=0.1
y_run=y[0:600]
n_trials=2400#len(y)
marker=[0]
alphaAll=[]
epochs_fb=np.zeros((200,n_channels,n_time))
y_feedback=np.concatenate((y[600:800],y[1000:1200],y[1400:1600],y[1800:2000],y[2200:2400]))
while marker[0]+1<n_trials:

    #if rej>10:
        #popupmsg("More than 5 errors")
        #messagebox.showinfo("Data problems", "High number of mismatch")
        #messagebox.showwarning("Warning","Warning message")
        #sg.Popup('Error', 'Too many mismatches')
    if state=='stable':

        epoch,state,marker,excess_EEG,excess_EEG_time,excess_marker,excess_marker_time,look_for_trigger=get_epoch(inlet_EEG,inlet_marker,store_EEG,store_marker,subject_id,excess_EEG,excess_EEG_time,excess_marker,excess_marker_time,state=state,look_for_trigger=look_for_trigger,tmax=tmax)
        '''
        håndter når der er mere end 2 markers tilgængelige i stabile blokke
        '''
        if len(epoch):
            epoch=preproc1epoch(epoch,info_fs500,SSP=False,reject=None,mne_reject=1,reject_ch=True,flat=None)
            stable_blocks[marker[0]-n_run*400,:,:]=epoch
            marker_all.append(marker)
            
    elif state=='train':
        
        

        # test if epochs are missing:
        ss=np.sum(np.sum(np.abs(stable_blocks),axis=2),axis=1)
        epochs0_idx=np.where(ss==0)[0]
        if len(epochs0_idx):
            rep=0
            while rep<30:
                print('WARNING missing '+str(len(epochs0_idx))+' stable epochs\n')
                rep+=1
            epochs_non0_avg=np.mean(np.delete(stable_blocks,epochs0_idx,axis=0),axis=0)
            stable_blocks[epochs0_idx]=epochs_non0_avg
        
        # run SSP
        projs,stable_blocksSSP=applySSP(stable_blocks,info_fs100,threshold=threshold)
        stable_blocksSSP=average_stable(stable_blocksSSP)
        #stable_blocksSSP_rem,reject,flat,bad_channels=removeEpochs(stable_blocksSSP,info_fs100)
        print('Training classifier') # on stable_blocksSSP   
        if n_run>0:
            print('Run more than 0 loop')
            ss_fb=np.sum(np.sum(np.abs(epochs_fb),axis=2),axis=1)
            epochs0_idx=np.where(ss_fb==0)[0]
            if len(epochs0_idx):
                rep=0
                while rep<30:
                    print('WARNING missing '+str(len(epochs0_idx))+' feedback epochs\n')
                    rep+=1
                    epochs_non0_avg=np.mean(np.delete(epochs_fb,epochs0_idx,axis=0),axis=0)
                    epochs_fb[epochs0_idx]=epochs_non0_avg
            
            stable_blocksSSP_append=np.append(stable_blocksSSP,epochs_fb,axis=0)
            fb_start=(n_run-1)*200
            y_run_append=np.append(y_run,y_feedback[fb_start:fb_start+200])
            print('About to train classifier cross2')
            clf,offset=trainLogReg_cross2(stable_blocksSSP_append,y_run_append)
            print('classifier cross2 trained')
            print('mean ' + str(np.mean(clf.coef_)))

        else:
            print('About to train classifier cross, nrun=0')
            clf,offset=trainLogReg_cross(stable_blocksSSP,y_run)
            print('mean ' + str(np.mean(clf.coef_)))
        offset=(np.max([np.min([offset,0.25]),-0.25]))/2
        print('Offset:' +str(offset))
        # prepare stable_blocks for next round (remove first 200 trials and pre-initialize to 600 trials)
        stable_blocks=stable_blocks[200:,:,:]
        stable_blocks=np.concatenate((stable_blocks,stable_blocks0),axis=0)
        y_run=np.concatenate((y_run[200:],y[800+400*n_run:1000+400*n_run]))
        
        
        print('Done training, moving to feedback')
        state='feedback'
        n_run+=1
        epochs_fb=np.zeros((200,n_channels,n_time))


                
    elif state=='feedback':
        #cnt_stable=800
        
#        if marker==613:
#            time.sleep(1.2)
#            
#        if marker==620:
#            time.sleep(3)
        
        epoch,state,marker,excess_EEG,excess_EEG_time,excess_marker,excess_marker_time,look_for_trigger=get_epoch(inlet_EEG,inlet_marker,store_EEG,store_marker,subject_id,excess_EEG,excess_EEG_time,excess_marker,excess_marker_time,state=state,look_for_trigger=look_for_trigger,tmax=tmax)
    
        t_test=marker-600-400*(n_run-1)
        if len(epoch):
            epoch=preproc1epoch(epoch,info_fs500,projs=projs,SSP=True,reject=reject,mne_reject=1,reject_ch=True,flat=flat)
            if t_test>0:    
                epoch=(epoch+epoch_prev)/2
            
            epochs_fb[t_test,:,:]=epoch
            print(t_test)
            #apply classifier
            pred_prob=testEpoch(clf,epoch)
            pred_prob[0][0,0]=np.min([ np.max([pred_prob[0][0,0]+offset,0]),1]) 
            pred_prob[0][0,1]=np.min([ np.max([pred_prob[0][0,1]-offset,0]),1])
            clf_output=pred_prob[0][0,int(y[marker[0]])]-pred_prob[0][0,int(y[marker[0]]-1)]
            alpha=sigmoid(clf_output)
            marker_alpha=[marker[0][0],alpha]
            print(marker_alpha)
            outlet_alpha.push_sample([alpha])
            epoch_prev=epoch
            
    
#%% Terminate logging of print statements
Transcript.stop()