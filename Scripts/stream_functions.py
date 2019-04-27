# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:42:04 2019

@author: sofha
"""

# Imports
from pylsl import StreamInlet, resolve_stream, resolve_byprop
import csv
import time
import numpy as np
import sys
from paths import subject_path_init
subject_path = subject_path_init()


class data_init:
    
    def __init__(self,fs,data_type,filename=None):
        '''
        Input:
        - fs: Sampling frequency
        - data_type: 'EEG' or 'marker' (trigger from experimental stimuli script)
        '''
        self.fs,self.filename,self.data_type = fs,filename,data_type
        

def save_data(data_info,sample,timestamp,user_id):
    #Saves sample and timestamp to csv file
    # Arguments:
    #   data_info: class from data_init
    #   sample: data chunk or sample to save
    #   timestamp: timestamp of chunk/sample
    #   user_id: subject ID
    
    if data_info.filename == None: # if file does not exist create it
        data_info.filename = subject_path+'\\'+user_id+'\\subject_'+user_id+'_'+data_info.data_type+'_'+time.strftime('%m-%d-%y_%H-%M')+'.csv'
        with open(data_info.filename,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_info.header)
    with open(data_info.filename,'a',newline='') as csvfile:
        writer = csv.writer(csvfile)
        if len(sample) <= 1:
            writer.writerow(np.append(np.array([sample]),np.array([timestamp])))
        else:
            writer.writerows(np.append(sample,np.array([timestamp]).T,axis=1))
    return data_info

def clear_stream(inlet):
    # Empty inlet for samples (pull all samples from the inlet)
    sample0, timestamp0 = inlet.pull_chunk(max_samples=1500)
    
    
def read_EEG_stream(fs=500,max_buf=2):
# Initialize EEG stream
# Arguments:
#   fs: sampling frequency
    # timeout: max time to look for whether EEG stream is available (seconds)
#   max_buf: maximum data to have in buffer (seconds)
    
    streamsEEG = resolve_byprop('type','EEG',timeout=1)
    inlet_EEG = StreamInlet(streamsEEG[0],max_buflen=max_buf)
    store_EEG = data_init(fs,'EEG') # Initialize object
    store_EEG.header = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4','Timestamp']
    return inlet_EEG,store_EEG

def read_marker_stream(stream_name ='MyMarkerStream3'):
    index_lsl = []
    lsl_created = []
    streams = resolve_byprop('type', 'Markers',timeout=10) # look for stream(s)
    for i in range(len(streams)):
        if (streams[i].name() == stream_name):
            index_lsl.append(i)
            lsl_created.append(streams[i].created_at()) # store when stream was created
    if index_lsl:
        if len(index_lsl) > 1: # Multiple stream available
            print("Not unique marker stream name, using most recent one")
            index_lsl = index_lsl[np.argmax(lsl_created)]
        else: # One stream available
            index_lsl = index_lsl[np.argmax(lsl_created)]
        print ("lsl stream available")
        inlet_marker = StreamInlet(streams[index_lsl])
        store_marker = data_init(500,'marker') # Initialize marker object
        store_marker.header = ['Marker','Timestamp']
    else:
        inlet_marker = []
        print('Warning: No marker inlet available')
    return inlet_marker,store_marker

def read_save_from_stream(inlet,store,user_id):
    # read and save data from inlet
    sample, timestamp = inlet.pull_chunk()
    sample = np.asarray(sample)
    timestamp = np.asarray(timestamp) 
    store = save_data(store,sample,timestamp,user_id)
    return sample,timestamp,store

def get_epoch(inlet_EEG,inlet_marker,store_EEG,store_marker,user_id,excess_EEG=[],excess_EEG_time=[],excess_marker=[],excess_marker_time=[],state='stable',look_for_trigger=1,tmax=1,fs=500):
    # Extracts EEG epoch from 0.1 second before marker/trigger (from psychopy) to tmax seconds after
    # Arguments:
    #   inlet_EEG, inlet_marker: EEG and stimuli trigger streams
    #   store_EEG, store_marker: class object of EEG and marker
    #   user_id: subject ID
    #   excess_EEG/marker(_time): EEG/marker buffered from previous epoch processing
    #   state: 'stable'/'feedback'
    #   look_for_trigger: look for stimuli marker
    #   tmax: epoch end (seconds)
    #   fs: sampling frequency
    # Outputs:
    #   epoch: extracted EEG epoch
    #   sample_marker: marker/trigger for stimuli onset
    #   and as above..
    
    t_latency = 0 # latency of EEG in relation to trigger
    tmin = -0.1 # seconds before stimulus onset
    t_epoch = tmax-tmin # length of epoch (seconds)
    s_epoch = int(t_epoch*fs) # samples in epoch
    s = 10 # extra samples to store for next epoch
    look_for_epoch = 1
    use_excess_triggers = 1
    
    
    while look_for_epoch:

        # read from marker stream 
        if look_for_trigger:
            sample_marker,timestamp_marker,store_marker = read_save_from_stream(inlet_marker,store_marker,user_id)
        if use_excess_triggers == 1: # Only go to excess trigger if current trigger has been processed
            if excess_marker: # if delay in system has caused markers to be buffered
                print('Using excess marker ' + str(excess_marker))
                sample_marker = excess_marker
                timestamp_marker = excess_marker_time
                excess_marker = []
                excess_marker_time = []    
            
        # read from EEG stream
        sample_EEG,timestamp_EEG,store_EEG = read_save_from_stream(inlet_EEG,store_EEG,user_id)
        if len(excess_EEG): 
            if len(sample_EEG):
                print(sample_EEG.shape)
                print(excess_EEG.shape)
                sample_EEG = np.concatenate((excess_EEG,sample_EEG),axis=0)
                timestamp_EEG = np.concatenate((excess_EEG_time,timestamp_EEG),axis=0)
            else: # if no new EEG data was read
                sample_EEG = excess_EEG
                timestamp_EEG = excess_EEG_time

        epoch = [] # initialize
        # Find stimuli marker onset in EEG
        if len(sample_marker): # if marker is present
            if len(sample_marker) > 1: # if more than one marker is present
                print("Warning. More than one trigger point recovered, using second recent one")
                excess_marker = np.asarray([sample_marker[-1]])
                excess_marker_time = np.asarray([timestamp_marker[-1]])
                sample_marker = np.asarray([sample_marker[-2]])
                timestamp_marker = np.asarray([timestamp_marker[-2]])            
                look_for_trigger = 0
                use_excess_triggers = 0
            else: 
                look_for_trigger = 1
                
            # Find timesample of EEG nearest stimuli onset plus tmin
            i_start = np.argmin(np.abs(timestamp_marker+t_latency+tmin-timestamp_EEG)) # find closest sample in the EEG corresponding to marker plus latency and baseline
            t_diff = timestamp_marker+t_latency+tmin-timestamp_EEG[i_start] # distance between EEG time sample and marker
            if np.abs(t_diff) > (1/fs): # Sample missing 
                print("Warning. TOO LONG BETWEEN EEG AND MARKER: ",t_diff)
            else:
                print("Time between EEG and marker: ",t_diff)
            
            avail_samples = (len(timestamp_EEG)-i_start) # No. samples from stimuli onset + tmin. i_start is timestamp of the EEG closely matched to timestamp of the marker, minus 0.1 sec
            if avail_samples >= s_epoch: # if one EEG epoch has been recovered
                epoch = sample_EEG[i_start:i_start+s_epoch,:] #  samples x channels. Add an epoch of size 900 ms, 450 samples
               
                look_for_epoch = 0 # done looking for epoch
                
                # Save EEG samples for next epoch
                if t_diff < (-2/fs): # Make sure that mismatches between EEG and marker do not accumlate over time
                    s_diff = int(np.abs(t_diff*fs)) # No. samples
                    print('Increasing excess_EEG by: ' + str(s_diff)) # Relevant if epochs are overlapping
                    excess_EEG = sample_EEG[i_start+fs-s-s_diff:,:]
                    excess_EEG_time = timestamp_EEG[i_start+fs-s-s_diff:]
                else:
                    excess_EEG = sample_EEG[i_start+fs-s:,:]
                    print('Saving' + str(excess_EEG.shape))
                    excess_EEG_time = timestamp_EEG[i_start+fs-s:]
                print("Ready to preprocess, marker: ",sample_marker)

            else:
                print("Warning. Not enough EEG samples available")
                print("Wait time ",np.max([0,(s_epoch-avail_samples)/fs])) # In seconds
                time.sleep(np.max([0,(s_epoch-avail_samples)/fs])+0.03)
                look_for_trigger = 0
                excess_EEG = sample_EEG
                excess_EEG_time = timestamp_EEG
               
        else:
            print("Warning. No trigger points recovered")
            time.sleep(0.1)
            look_for_trigger = 1
            excess_EEG = sample_EEG
            excess_EEG_time = timestamp_EEG
    
    # Get ready for next epoch, update state
    state,reset = get_state(state,sample_marker)
    if reset == 0: # If state has been changed
        excess_EEG = []
        excess_EEG_time = []
        excess_marker = []
        excess_marker_time = []
        look_for_trigger = 1
        
    return epoch,state,sample_marker,excess_EEG,excess_EEG_time,excess_marker,excess_marker_time,look_for_trigger

def get_state(state,sample_marker,offset=400,n_marker_train=200,n_marker_feedback=200):
    # Determines whether to change state between stable/train/feedback.
    
    # Arguments:
        # state: current state (stable/train/feedback)
        # sample_marker: marker number from psychopy
        # offset: how many samples to skip before alternating between types of epochs
        # n_marker_train: no. of stable epochs to record in each run (except in first run) 
        # n_marker_feedback: no. of feedback epochs to record in each run (except in first run)
    # Outputs:
        # state: current state (stable/train/feedback)
        # reset: whether to reset excess data arrays (done if state has been altered)
    
    interval = n_marker_train+n_marker_feedback # No. epochs in a run (default 400)
    
    if sample_marker >= offset+n_marker_train-1: 
        if not (sample_marker+1)%interval: # if 800, 1200...
            print('Feedback done, ready to collect stable blocks')
            state = 'stable'
            reset = 1
        elif (sample_marker+1-offset)%interval == n_marker_train: # if 600, 1000, 1400..
            state = 'train'
            print('Done collecting stable epochs')
            reset = 1
        else:
            reset = 0
    else:
        reset = 0
            
    return state, reset            

class Transcript(object):
# Save print statements to file
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def start(filename):
        """Start transcript, appending print output to given filename"""
        sys.stdout = Transcript(filename)

    def stop():
        """Stop transcript and return print functionality to normal"""
        sys.stdout.logfile.close()
        sys.stdout = sys.stdout.terminal