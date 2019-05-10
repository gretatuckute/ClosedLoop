#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:47:28 2018

@author: jonf
"""
import numpy as np
import multiprocessing

from offline_analysis_FUNC_v1 import *

multiprocessing.cpu_count()


class p:
    # =============================================================================
    # Default setting
    # =============================================================================
    nCors                   = 8
    shouldCheckFirst        = True


def fit(subjID):    
    print('Starting analyzeOffline for subjID '+str(subjID))
    analyzeOffline(subjID)
    print('analyzeOffline done for subjID '+str(subjID))
    

def train():
    # sub_types = ['07','08','11','13','14','15','16','17','18','19',\
    #       '21','22','23','24','25','26','27','30','31','32','33','34']
    
    sub_types=['30']
    
    processes   = [] # for parallelism
    
    for subjID in sub_types:
        proc = multiprocessing.Process(target=fit, args=(subjID,))
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


        
#%%

if __name__ == '__main__':
    train()
