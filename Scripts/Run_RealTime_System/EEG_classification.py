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
from sympy import symbols, solve

def scale2DArray(eeg_array,axis=1):
    '''
    Scales a 3D array to a 2D array, with scaling on the specified axis.
    
    Input:
    - eeg_array: Array of EEG data in the following format: (trials, channels, time samples).
    - axis: Normalization axis (mean 0, std 1)
    '''
    
    no_trials = eeg_array.shape[0]
    no_chs = eeg_array.shape[1]
    no_samples = eeg_array.shape[2]

    X_res = np.reshape(eeg_array,(no_trials,no_chs*no_samples))
    X_z = zscore(X_res, axis=axis)

    return X_z

def scale1DArray(eeg_array,axis=1):
    '''
    Scales a 2D array, with scaling on the specified axis.
    
    Input:
    - eeg_array: Array of EEG data in the following format: (channels, time samples).
    - axis: Normalization axis (mean 0, std 1)

    '''
    no_chs = eeg_array.shape[0]
    no_samples = eeg_array.shape[1]

    X_res = np.reshape(eeg_array,(1,no_chs*no_samples))
    X_z = zscore(X_res, axis=axis)
    return X_z


def trainLogReg(X,y):
    '''
    Trains a classifier on EEG epochs from stable blocks: (n_trials,n_channels,n_samples)
    No offset for correction of classifier bias.

    Input:
    - X: Epochs. 
    - y: Binary category array.
    
    Output:
    - clf: SAGA LogReg with L1 penalty, fitted on X and y.
    '''
    
    X = scale2DArray(X,axis=1)
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1')
    
    clf = classifier.fit(X,y)

#    pred_prob =  clf.predict_proba(X) 
#    score_train = clf.score(X,y) 
    
    return clf


def trainLogReg_cross(X,y):
    '''
    Trains a classifier on EEG epochs from stable blocks: (600,n_channels,n_samples). 
    For the first NF run (n_run=0).
    Offset for correction of classifier bias computed in 3-fold cross-validation (mean of 3 offset values)

    Input:
    - X: Epochs (after preprocessing, SSP correction and two-trial averaging). Length 600 trials.
    - y: Binary category array.
    
    Output:
    - clf: SAGA LogReg with L1 penalty, fitted on X and y.
    - offset: Offset value for correction of classification bias.
    '''
    
    X = scale2DArray(X,axis=1)
    
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1',max_iter=100)
    
    # score = []
    # conf = []
    
    # Estimation of offset for correction of classifier bias. 3-fold cross-validation. 
    # Cross-validation fold number 1.
    X_train = X[0:400,:]
    y_train = y[0:400]
    
    clf = classifier.fit(X_train,y_train)
    X_val = X[400:,:]
    # y_val = y[400:]
    pred_prob_val = clf.predict_proba(X_val)
    pred_sort = np.sort(pred_prob_val[:,0])
    offset1 = 0.5 - pred_sort[100]
    
    # Following piece of code can be uncommented for validation accuracy and confusion matrices.
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    # Cross-validation fold number 2.
    X_train = np.concatenate((X[0:200,:],X[400:600,:]),axis=0)
    y_train = np.concatenate((y[0:200],y[400:600]),axis=0)
    clf = classifier.fit(X_train,y_train)
    X_val = X[200:400,:]
    # y_val = y[200:400]
    pred_prob_val = clf.predict_proba(X_val)
    pred_sort = np.sort(pred_prob_val[:,0])
    offset2 = 0.5-pred_sort[100]
    
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    # Cross-validation fold number 3.
    X_train = X[200:,:]
    y_train = y[200:]
    clf = classifier.fit(X_train,y_train)
    X_val = X[:200,:]
    # y_val = y[:200]
    pred_prob_val = clf.predict_proba(X_val)
    
    pred_sort = np.sort(pred_prob_val[:,0])
    offset3 = 0.5-pred_sort[100]
    
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    offset = (offset1+offset2+offset3)/3
    
    clf = classifier.fit(X,y)

    # pred_prob = clf.predict_proba(X) 
    # score_train = clf.score(X,y) 
    
    return clf, offset #,score,conf

def trainLogReg_cross2(X,y):
    '''
    Trains a classifier on EEG epochs from stable blocks and recent NF blocks: (800,n_channels,n_samples)
    The offset is computed based on the most recent 200 trials.
    
    Input:
    - X: Epochs (after preprocessing, SSP correction and two-trial averaging). Length 800 trials.
    - y: Binary category array.
    
    Output:
    - clf: SAGA LogReg with L1 penalty, fitted on X and y.
    - offset: Offset value for correction of classification bias.
    '''
    
    X = scale2DArray(X,axis=1)
    
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1',max_iter=100)
    
    # score = []
    # conf = []
    X_train = X[0:600,:]
    y_train = y[0:600]
    
    clf = classifier.fit(X_train,y_train)
    X_val = X[600:,:]
    # y_val = y[600:]
    pred_prob_val = clf.predict_proba(X_val)
    pred_sort = np.sort(pred_prob_val[:,0])
    offset = 0.5-pred_sort[100]
    
    # y_pred = clf.predict(X_val)
    # score.append(metrics.accuracy_score(y_val,y_pred))
    # conf.append(metrics.confusion_matrix(y_val,y_pred))

    clf = classifier.fit(X,y)

#    pred_prob =  clf.predict_proba(X) 
#    score_train = clf.score(X,y) 
    
    return clf, offset #,score,conf

def trainLogReg_cross_offline(X,y):
    '''
    For offline estimation of training accuracy.
    '''
    
    X=scale2DArray(X,axis=1)
    
    classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1',max_iter=100)
    
    if X.shape[0] == 1750:
        val_split = 583
    else:
        val_split = int((X.shape[0])/3) #Split in 3 equal sized validation folds
    
    offset_lst = []
        
    for entry in range(3): 

        indVal = np.asarray(range(val_split*(entry),(entry+1)*val_split))

        X_val = X[indVal,:]
#        y_val = y[indVal]
        
        indTrain = np.asarray(range(X.shape[0]))
        indTrain = np.delete(indTrain,indVal,axis=0)
        
        X_train = X[indTrain,:]
        y_train = y[indTrain]
            
        clf_val = classifier.fit(X_train, y_train)
        pred_prob_val = clf_val.predict_proba(X_val)
        
        pred_sort = np.sort(pred_prob_val[:,0])

        offset = 0.5 - pred_sort[int(val_split/2)]
        offset_lst.append(offset)

    offset_mean = np.mean(offset_lst)
    
    clf = classifier.fit(X,y)

    return clf,offset_mean


def testEpoch(clf,epoch,y=None):
    '''
    Tests an epoch based on a trained classifier. 

    Input:
    - clf: Trained classifier.
    - X: Epoch (after preprocessing, SSP correction and averaging).
    
    Output:
    - pred_prob: Prediction probability of classifier.
    - clf.predict(epoch): Predicted, binary value of the classifier.
    '''
    
    epoch = scale1DArray(epoch)
    # y_pred = clf.predict(epoch)
    
    pred_prob =  clf.predict_proba(epoch)

    return pred_prob, clf.predict(epoch) 


def sigmoid_base(x, A, B, C, D):
    '''
    Sigmoid 4 parameter base.
    '''
    
    return ((A-D)/(1+10**((C-x)*B))) + D

def sigmoid(x):
    '''
    Transfer function for mapping prediction probability to alpha value between 0.17 and 0.98.
    Transfer function not continuous.
    
    Input:
    - x: Float between -1 and 1, prediction probability value.
    
    Output:
    - alpha: Float between 0.17 and 0.98, mapped using the sigmoid function with specified parameters.
    
    '''
    
    offset1 = 0.0507
    offset2 = 0.0348
    A1 = 0.98 + offset1
    B = 1.3 # Steepness
    C = 0 # Inflection point
    D1 = 1 - A1 # 0.17
    D2 = 0.17 - offset2
    A2 = 1 - D2
    
    if x >= 0:
        alpha = sigmoid_base(x,A1,B,C,D1)

    else:
        alpha = sigmoid_base(x,A2,B,C,D2)
        
    return alpha


def solveSigmoid(alpha):
    '''
    alpha = ((A-D)/(1+10**((C-x)*B))) + D
    '''
    x = symbols('x',real=True)

    
    offset1 = 0.0507
    offset2 = 0.0348
    A1 = 0.98 + offset1
    B = 1.3 # Steepness
    C = 0 # Inflection point
    D1 = 1 - A1 # 0.17
    D2 = 0.17 - offset2
    A2 = 1 - D2
    
    if alpha >= 0.5: # I.e. when the clf output is >= 0 
        expr = (((A1-D1)/(1+10**((C-x)*B))) + D1) - alpha # Last val here is alpha, i.e. the output from the sigmoid function
    if alpha < 0.5:
        expr = (((A2-D2)/(1+10**((C-x)*B))) + D2) - alpha
        
    return solve(expr)

#%% Test if sigmoid conversion works
# clfout = d_all2['13']
# clfout_13 = clfout['CLFO_test']
# alpha_13 = clfout['ALPHA_test']

# clfout_match = []
# for alpha in alpha_13[:20]:
#     clfout_val = (solveSigmoid(alpha)[0]).evalf(3)
#     clfout_match.append(clfout_val)









