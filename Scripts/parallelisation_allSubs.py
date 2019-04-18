#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:47:28 2018

@author: jonf
"""
import sys
import numpy as np
import pandas as pd
import os
import pickle
import multiprocessing
import time
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection, linear_model
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore

multiprocessing.cpu_count()


class p:
    # =============================================================================
    # Default setting
    # =============================================================================
    time_left_for_this_task = 600
    ensemble_size		    = 50
    per_run_time_limit      = 60
    deci                    = 5
    kFolds                  = 5
    R                       = 100
    startR                  = 50
    testSize                = 0.25
    shouldScaleX            = True
    runinpar                = True
    printTrainTestIdxs      = False
    shouldSplit             = True
    xPath                   = ""
    clsreg                  = 'classification'
    nCors                   = 4
    shouldCheckFirst        = True


def fit(X,y,clf_type):
    #start  = time.time()
    
    if clf_type == 'sagalogl1':
        classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l1')
        
    if clf_type == 'sagalogl2':
        classifier = LogisticRegression(solver='saga',C=1,random_state=1,penalty='l2')
    
    if clf_type == 'lbfgslogl2':
        classifier = LogisticRegression(solver='lbfgs',C=1,random_state=1,penalty='l2')
        
    if clf_type == 'lda':
        classifier = LinearDiscriminantAnalysis()
        
    if clf_type == 'svm':
        classifier = svm.SVC(random_state=1,C=1,probability=True)
    
    if clf_type == 'sagalogl1c10':
        classifier = LogisticRegression(solver='saga',C=10,random_state=1,penalty='l1')
        
    if clf_type == 'sagalogl1c100':
        classifier = LogisticRegression(solver='saga',C=100,random_state=1,penalty='l1')
        
    if clf_type == 'sagalogl1c1000':
        classifier = LogisticRegression(solver='saga',C=1000,random_state=1,penalty='l1')
        
    if clf_type == 'sagalogl2c01':
        classifier = LogisticRegression(solver='saga',C=0.1,random_state=1,penalty='l2')
    
    if clf_type == 'lbfgslogl2c01':
        classifier = LogisticRegression(solver='lbfgs',C=0.1,random_state=1,penalty='l2')
        
    if clf_type == 'lbfgslogl2c001':
        classifier = LogisticRegression(solver='lbfgs',C=0.01,random_state=1,penalty='l2')
        
    if clf_type == 'lbfgslogl2c001':
        classifier = LogisticRegression(solver='lbfgs',C=0.001,random_state=1,penalty='l2')
        
    if clf_type == 'lbfgslogl2c10':
        classifier = LogisticRegression(solver='lbfgs',C=10,random_state=1,penalty='l2')
        
    if clf_type == 'lbfgslogl2c100':
        classifier = LogisticRegression(solver='lbfgs',C=100,random_state=1,penalty='l2')
    
    if clf_type == 'ada':
        classifier =  AdaBoostClassifier(random_state=1)
        
        
       

    
    
    clf = classifier.fit(X,y)
    
    print('fitting done')
    
    # Save train predictions?
    #np.savetxt('results/'+p.clsreg+'/predictionTrain---'+subModName+'---r~'+str(i)+'.csv',train_predictions, fmt='%.5e', delimiter=',')
    #end = time.time()

    pickle.dump(clf, open('C:\\Users\\Greta\\Documents\\GitLab\\project\\Python_Scripts\\pckl_files\\'+str(clf_type)+'.pkl','wb'))


def train(X, y,X_test,y_test):
    
#    clf_types = ['sagalogl1','sagalogl2','lbfgslogl2','lda','svm','sagalogl1c10','sagalogl2c01','lbfgslogl2c01','lbfgslogl2c001','sagalogl1c100','sagalogl1c1000']
    clf_types = ['sagalogl1','sagalogl2','lbfgslogl2','lda']

    
    processes   = [] # for parallelism
    
    for c in clf_types:
        proc = multiprocessing.Process(target=fit, args=(X,y,c,))
        proc.start()
        processes.append(proc)
        print('len of processes: ' + str(len(processes)))
        
        # Hold until processes are done if exceeding no. cores
        if len(processes) >= p.nCors:
            for pr in processes:
                pr.join()
            processes   = []
    
    # End all processes
#    for pr in processes:
#        pr.join()
#        
    pckls = []
    
    # open clfs
    for c in clf_types:
        
        with open('C:\\Users\\Greta\\Documents\\GitLab\\project\\Python_Scripts\\pckl_files\\clf'+str(c)+'.pkl', "rb") as fin:
            pcklclf = pickle.load(fin)
            pckls.append(pcklclf)
    
    len_clf = len(clf_types)-1
    
    str_lst=[str(x) for x in range(len_clf)]
    
    est_lst = []
    for count,string in enumerate(str_lst):
        
        est = (string, pckls[count])
        est_lst.append(est)
        
    
    
    
    eclassifier = VotingClassifier(estimators=est_lst, voting='soft')
    #eclassifier = VotingClassifier(estimators=[('1', pckls[0]), ('2', pckls[1]),('3', pckls[2]),('4', pckls[3])], voting='hard')
    #eclassifier = VotingClassifier(estimators=[('1', pckls[0])], voting='hard')
    #eclassifier=pckls[0]

    eclf = eclassifier.fit(X, y)
    
    score_test = eclf.score(X_test,y_test)
    print(score_test)
    print(est_lst)
    
    return eclf

        
#%%

if __name__ == '__main__':
#    X=np.random.rand(200,400)
#    y=np.zeros((200))
#    y[0:20]=1
    
    Xz=scale2DArray(stable_blocksSSP1)

    Xz_test=nf_arr
    
#    X=Xz # defined from part2 GT classification, b=0
    y1=y_run
    
#    X_test=nf_arr
    y_test1=y_test
    
    eclf1=train(Xz,y1,Xz_test,y_test1) #outputs the classifier
    #fit(X,y,'ll1saga')
