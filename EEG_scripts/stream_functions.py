# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:42:04 2019

@author: sofha
"""
from pylsl import StreamInlet, resolve_stream, resolve_byprop
import csv
import time
import numpy as np
import sys



import csv
class data_init:
    def __init__(self, fs,data_type,filename=None):
        self.fs,self.filename,self.data_type = fs,filename,data_type
        
glo=data_init(500,'test')
#glo.options.filename=time.strftime("%H%M%S_%d%m%Y")
def save_data(data,sample,timestamp,user_id):
    gamer_dir='C:\\Users\\nicped\\Documents\\GitLab\\project\\SUBJECTS\\'+user_id+'\\'
    #sofha_dir='C:\\Users\\sofha\\Documents\\GitLab\\project\\logging\\EEGdata\\'
    #if exist(glo.options.filename)
    if data.filename==None:
        data.filename=gamer_dir+'subject_'+user_id+'_'+data.data_type+'_'+time.strftime('%m-%d-%y_%H-%M')+'.csv'
        with open(data.filename,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data.header)
    with open(data.filename,'a',newline='') as csvfile:
        #fieldnames=['name1','name2']
        writer = csv.writer(csvfile)
        #time_s=np.array([timestamps]).T
        if len(sample)<=1:
            writer.writerow(np.append(np.array([sample]),np.array([timestamp])))
        else:
            writer.writerows(np.append(sample,np.array([timestamp]).T,axis=1))
    return data

def clear_stream(inlet):
    sample0, timestamp0 = inlet.pull_chunk(max_samples=1500)
    
    
    
def read_EEG_stream(fs=500,max_buf=2):
#import PySimpleGUI as sg    
    streamsEEG = resolve_byprop('type', 'EEG',timeout=10)
    inlet_EEG=StreamInlet(streamsEEG[0],max_buflen=max_buf)
    store_EEG=data_init(fs,'EEG')
    store_EEG.header=['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4','Timestamp']
    return inlet_EEG,store_EEG
#%%
def read_marker_stream(stream_name ='MyMarkerStream3'):#
    index_lsl=[]
    lsl_created=[]
    streams = resolve_byprop('type', 'Markers',timeout=10)
    for i in range (len(streams)):
        if (streams[i].name() == stream_name):
            index_lsl.append(i)
            lsl_created.append(streams[i].created_at())
    if index_lsl:
        if len(index_lsl)>1:
            print("Not unique marker stream name, using most recent one")
            index_lsl=index_lsl[np.argmax(lsl_created)]
        else:
            index_lsl=index_lsl[np.argmax(lsl_created)]
        print ("lsl stream available")
        inlet_marker = StreamInlet(streams[index_lsl])
        lsl_avail=1
        store_marker=data_init(500,'marker')
        store_marker.header=['Marker','Timestamp']
    else:
        inlet_marker=[]
        print('Warning: No marker inlet available')
    return inlet_marker,store_marker

def read_save_from_stream(inlet,store,user_id):
    sample, timestamp = inlet.pull_chunk()
    sample=np.asarray(sample)
    timestamp =np.asarray(timestamp) 
    store=save_data(store,sample,timestamp,user_id)
    return sample,timestamp,store

def get_epoch(inlet_EEG,inlet_marker,store_EEG,store_marker,user_id,excess_EEG=[],excess_EEG_time=[],excess_marker=[],excess_marker_time=[],state='stable',look_for_trigger=1,tmax=1):
    fs=500
    t_latency=0
    tmin=-0.1 # seconds before stimulus onset
    t_epoch=tmax-tmin # seconds
    s_epoch=int(t_epoch*fs) # samples
    s=10 #safety samples
    rej=0
    look_for_epoch=1
    multi_triggers=0
    # read from marker stream (sent from psychopy script)
    
    while look_for_epoch:
        #t1=time.clock()

        if look_for_trigger:
            sample_marker, timestamp_marker,store_marker=read_save_from_stream(inlet_marker,store_marker,user_id)
        if multi_triggers==0:
            if excess_marker:
                print('Using excess marker '+str(excess_marker))
                sample_marker=excess_marker#np.concatenate((excess_marker, sample_marker))#np.concatenate((excess_marker, sample_marker),axis=0)
                timestamp_marker=excess_marker_time#np.append(excess_marker_time, timestamp_marker.T)#np.concatenate((excess_marker_time, timestamp_marker),axis=0)
                excess_marker=[]
                excess_marker_time=[]    
            

        #else:
           
            #sample_marker,timestamp_marker=excess_marker,excess_marker_time
        
        # read from EEG stream
        sample_EEG,timestamp_EEG,store_EEG=read_save_from_stream(inlet_EEG,store_EEG,user_id)
        if len(excess_EEG):
            if len(sample_EEG):
                print(sample_EEG.shape)
                print(excess_EEG.shape)
                sample_EEG=np.concatenate((excess_EEG, sample_EEG),axis=0)
                timestamp_EEG=np.concatenate((excess_EEG_time, timestamp_EEG),axis=0)
            else:
                sample_EEG=excess_EEG
                timestamp_EEG=excess_EEG_time
        #print(sample_EEG.shape)
        epoch=[] # initialize
        if len(sample_marker): # find stimuli onset in EEG
            if len(sample_marker)>1: 
                print("Warning. More than one trigger point recovered, using second recent one")
                excess_marker=np.asarray([sample_marker[-1]])
                excess_marker_time=np.asarray([timestamp_marker[-1]])
                sample_marker=np.asarray([sample_marker[-2]])
                timestamp_marker=np.asarray([timestamp_marker[-2]])            
                look_for_trigger=0
                multi_triggers=1
            else: 
                look_for_trigger=1

            #min_time=min(np.abs(timestamp_marker-timestamp_EEG+t_latency)) # shortest distance between marker and EEG
            i_start=np.argmin(np.abs(timestamp_marker+ t_latency+tmin-timestamp_EEG )) # find closest sample in the EEG corresponding to marker plus latency and baseline
            t_diff=timestamp_marker+t_latency+tmin-timestamp_EEG[i_start]
            #min_time_all.append(min_time)
            if np.abs(t_diff)>1/fs:
                print("Warning. TOO LONG BETWEEN EEG AND MARKER: ",t_diff)
            else:
                print("Time between EEG and marker: ",t_diff)
            avail_samples=(len(timestamp_EEG)-i_start) 
            print(i_start)
            print(timestamp_EEG[i_start])
            print(timestamp_marker)
            if avail_samples>=s_epoch:
                rej=0
                #C+=1
                epoch=sample_EEG[i_start:i_start+s_epoch,:] #  550x32 
                #epoch=preproc1epoch(epoch.T,info,projs,reject=20) # in MNE format
               
                look_for_epoch=0 
                print(i_start+fs-s)
                if t_diff<-2/fs: # Make sure that mismatches between EEG and marker do not accumlate over time
                    s_diff=int(np.abs(t_diff*fs)) # no. samples
                    print('Increasing excess_EEG by: '+str(s_diff))
                    excess_EEG=sample_EEG[i_start+fs-s-s_diff:,:]
                    excess_EEG_time=timestamp_EEG[i_start+fs-s-s_diff:]
                else:
                    excess_EEG=sample_EEG[i_start+fs-s:,:]
                    print('saving' + str(excess_EEG.shape))
                    excess_EEG_time=timestamp_EEG[i_start+fs-s:]
                print("Ready to preprocess, marker",sample_marker)

            else:
                rej+=1
                print("Warning. Not enough EEG samples available")
                print("Wait time",np.max([0,(s_epoch-avail_samples)/fs]))
                time.sleep(np.max([0,(s_epoch-avail_samples)/fs])+0.03)
                look_for_trigger=0
                excess_EEG=sample_EEG
                excess_EEG_time=timestamp_EEG
                #print('No of markers',len(sample_marker))
                #if len(excess_marker):    
                #    sample_marker=np.concatenate((sample_marker,excess_marker),axis=0)
                #   timestamp_marker=np.concatenate((timestamp_marker,excess_marker_time),axis=0)
                    
                #print(len(excess_marker))
                #excess_marker=sample_marker
                #excess_marker_time=timestamp_marker
    
        else:# not sample_lsl.any():
            rej+=1
            print("Warning. No trigger points recovered")
            t2=time.clock()
            time.sleep(0.1)#max(pull_interval-(t2-t1),0))
            look_for_trigger=1
            excess_EEG=sample_EEG
            excess_EEG_time=timestamp_EEG
    
    if sample_marker>598: 
        if not (sample_marker+1)%400: # if 400, 800, 1200...
            print('Feedback done, ready to collect stable blocks')
            state='stable'
            excess_EEG=[]
            excess_EEG_time=[]
            excess_marker=[]
            excess_marker_time=[]
            look_for_trigger=1
        elif (sample_marker+1)%400==200: # if 600, 1000, 1400..
            state='train'
            print('Training')
            excess_EEG=[]
            excess_EEG_time=[]
            excess_marker=[]
            excess_marker_time=[]
            look_for_trigger=1
    #else:    
     #   
        
    return epoch,state,sample_marker,excess_EEG,excess_EEG_time,excess_marker,excess_marker_time,look_for_trigger#excess_marker,excess_marker_time,

class Transcript(object):

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