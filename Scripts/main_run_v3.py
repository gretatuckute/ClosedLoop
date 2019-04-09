# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:12:52 2018

@author: Greta
"""

# Imports
import os
import numpy as np
import random
import time
from paths import script_path_init
from psychopy import gui, core
from pylsl import StreamInlet, resolve_stream, resolve_byprop

script_path = script_path_init()
os.chdir(script_path)

############### Global variables ###############
from experiment_closed_loop_vol3 import * 

global stableSaveCount
global imgIdx

### TEST RUNS ###

#runTest(day='1')
    
# runTest(day='2')


####  Behavioral run ### 
# runBehDay(day=4) # dag1: 1 og 4, dag 3: 3 og 5


#### NF run ### 

runNFday(subjID='90',numRuns=5,numBlocks=8,blockLen=50)


#runNFday(subjID='19',numRuns=5,numBlocks=8,blockLen=50)

