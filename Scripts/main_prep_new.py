# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:12:52 2018


@author: Greta
"""

# Imports
import os
from paths import script_path_init

script_path = script_path_init()
os.chdir(script_path)

from experiment_closed_loop_vol3 import createIndicesNew, closeWin
import numpy as np
import random

closeWin()

############### Global variables ###############
global stableSaveCount
global imgIdx


subjID_prep = 'ab'

numRuns = 2 # E.g. 3 creates 3*8*50 = 1200 indices


catComb1 = [['dog','cat','airplane','bus'],
            ['airplane','bus','dog','cat']]

catComb2 = [['dog','cat','bus','airplane'],
            ['bus','airplane','dog','cat']]

catComb3 = [['cat','dog','airplane','bus'],
            ['airplane','bus','cat','dog']]

catComb4 = [['cat','dog','bus','airplane'],
            ['bus','airplane','cat','dog']]

############### PREP ONLY: INDICES FOR FUSING IMGS ###############
# Create fused images to display for 1 run (8 blocks)

randLst1 = [0,0,1,1]
randLst2 = [0,0,1,1]

# Generates the chosen combination for a subj

# Randomly assign 4 subjs to catComb1, 4 subjs to catComb2 etc.

for run in list(range(0,numRuns)):
    randCat  = (random.sample(randLst1, 4)) + (random.sample(randLst2, 4))
    for index in randCat:
        aDom = catComb4[index][0]
        aLure = catComb4[index][1]
        nDom = catComb4[index][2]
        nLure = catComb4[index][3]

        createIndicesNew(aDom, aLure, nDom, nLure, subjID_forfile=subjID_prep,exp_day=5)
        
