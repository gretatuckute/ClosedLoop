# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:01:35 2019

Functions for classifying epoched EEG data in a binary classification task:


@author: Greta
"""

# Imports
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from scipy.stats import zscore
import pandas as pd
import os 
import mne
from numpy import unravel_index
from scipy.signal import detrend

def scale2DArray(eeg_array,axis=1):
    '''
    Scales a 3D array to a 2D array, with scaling on the specified axis.
    
    Input:
    - eeg_array in the following format: (trials, channels, time samples).
    - axis: normalization axis (mean 0, std 1)
    '''
    
    no_trials = eeg_array.shape[0]
    no_chs = eeg_array.shape[1]
    no_samples = eeg_array.shape[2]

    X_res = np.reshape(eeg_array,(no_trials,no_chs*no_samples))
    X_z = zscore(X_res, axis=axis)
    #X_mean=X_res.mean(axis=axis,keepdims=True)
    #X_std=X_res.std(axis=axis)
#    if axis==0:
#        X_z=(X_res - X_mean) / X_std[:,None].T
#    else:
#        X_z=(X_res - X_mean) / X_std[:,None]

    return X_z#,X_mean,X_std

def scale1DArray(eeg_array,axis=1):
    '''
    Scales a 2D array, with scaling on the specified axis.
    
    Input:
    - eeg_array in the following format: (channels, time samples).
    - axis: normalization axis (mean 0, std 1)

    '''
    no_chs = eeg_array.shape[0]
    no_samples = eeg_array.shape[1]

    X_res = np.reshape(eeg_array,(1,no_chs*no_samples))
    X_z = zscore(X_res, axis=axis)
    return X_z


def trainLogReg_cross2(X,y):
    '''
    Train classifier based on epochs based on stable blocks and recent NF blocks: (800,n_channels,n_samples)
    The offset is based on the most recent 200 trials.
    
    Input:
    - X: Epochs (after preprocessing, SSP and reshape/standardize). Length 800 trials.
    
    Output:
    - clf: SAGA LogReg with l1 penalty, fitted on X and y
    - offset: Offset value for correction of classification bias
    '''
    
    X=scale2DArray(X,axis=1)
    
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1',max_iter=100)
    
    #score=[]
    #conf=[]
    X_train=X[0:600,:]
    y_train=y[0:600]
    
    #classifier=train2Log(X_train,y_train)

    clf = classifier.fit(X_train,y_train)
    X_val=X[600:,:]
    #y_val=y[600:]
    pred_prob_val =  clf.predict_proba(X_val)
    pred_sort=np.sort(pred_prob_val[:,0])
    offset=0.5-pred_sort[100]
    #y_pred=clf.predict(X_val)
    #score.append(metrics.accuracy_score(y_val,y_pred) )
    #conf.append(metrics.confusion_matrix(y_val,y_pred))

    clf = classifier.fit(X,y)

#    pred_prob =  clf.predict_proba(X) 
#    score_train = clf.score(X,y) 
    
    return clf,offset#,score,conf

def trainLogReg_cross(X,y):
    '''
    Train classifier based on epochs based on stable blocks: (600,n_channels,n_samples). For the first NF run.

    Input:
    - X: Epochs (after preprocessing, SSP and reshape/standardize). Length 600 trials.
    
    Output:
    - clf: SAGA LogReg with l1 penalty, fitted on X and y
    - offset: Offset value for correction of classification bias
    '''
    
    X=scale2DArray(X,axis=1)
    
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1',max_iter=100)
    
    #score=[]
    #conf=[]
    
    #classifier=train2Log(X,y)

    X_train=X[0:400,:]
    y_train=y[0:400]
    
    clf = classifier.fit(X_train,y_train)
    X_val=X[400:,:]
    y_val=y[400:]
    pred_prob_val =  clf.predict_proba(X_val)
    pred_sort=np.sort(pred_prob_val[:,0])
    offset1=0.5-pred_sort[100]
    y_pred=clf.predict(X_val)
    #score.append(metrics.accuracy_score(y_val,y_pred) )
    #conf.append(metrics.confusion_matrix(y_val,y_pred))

    X_train=np.concatenate((X[0:200,:],X[400:600,:]),axis=0)
    y_train=np.concatenate((y[0:200],y[400:600]),axis=0)
    clf = classifier.fit(X_train,y_train)
    X_val=X[200:400,:]
    y_val=y[200:400]
    pred_prob_val =  clf.predict_proba(X_val)
    pred_sort=np.sort(pred_prob_val[:,0])
    offset2=0.5-pred_sort[100]
    y_pred=clf.predict(X_val)
    #score.append(metrics.accuracy_score(y_val,y_pred))
    #conf.append(metrics.confusion_matrix(y_val,y_pred))

    X_train=X[200:,:]
    y_train=y[200:]
    clf = classifier.fit(X_train,y_train)
    X_val=X[:200,:]
    y_val=y[:200]
    pred_prob_val =  clf.predict_proba(X_val)
    y_pred=clf.predict(X_val)
    #score.append(metrics.accuracy_score(y_val,y_pred))
    pred_sort=np.sort(pred_prob_val[:,0])
    offset3=0.5-pred_sort[100]
    #conf.append(metrics.confusion_matrix(y_val,y_pred))

    offset=(offset1+offset2+offset3)/3
    
#    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1',max_iter=100)
    clf = classifier.fit(X,y)

#    pred_prob =  clf.predict_proba(X) 
#    score_train = clf.score(X,y) 
    
    return clf,offset#,score,conf


def trainLogReg(X,y):
    '''
    Train classifier based on epochs based on stable blocks: (600,n_channels,n_samples)

    Input:
    - X: Epochs (after preprocessing, SSP and reshape/standardize)
    '''
    
    X=scale2DArray(X,axis=1)
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1')
    
    clf = classifier.fit(X,y)

#    pred_prob =  clf.predict_proba(X) 
#    score_train = clf.score(X,y) 
    
    return clf

def testEpoch(clf,epoch,y=None):
    '''
    Test epoch based on trained classifier

    Input:
    - X: Epoch (after preprocessing, SSP and reshape/standardize)
    '''
    
    epoch=scale1DArray(epoch)
    #y_pred = clf.predict(epoch)
    
    pred_prob =  clf.predict_proba(epoch)

    return pred_prob,clf.predict(epoch) 


def sigmoid_base(x, A, B, C, D):
    """4PL lgoistic equation."""
    return ((A-D)/(1+10**((C-x)*B))) + D

def sigmoid(x):
    offset1=0.0507
    offset2=0.0348
    A1=0.98+offset1
    B=1.3 # steepness
    C=0 # inflection point
    D1=1-A1#0.17
    D2=0.17-offset2
    A2=1-D2
    #A,B,C,D = 0.5,2.5,8,7.3
    if x>=0:
        alpha=sigmoid_base(x,A1,B,C,D1)

    else:
        alpha=sigmoid_base(x,A2,B,C,D2)
        
    return alpha



    