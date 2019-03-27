# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:12:52 2018


@author: Greta
"""

# Imports
import os
from paths import project_path_init

project_path = project_path_init()
os.chdir(project_path)

from experiment_closed_loop_vol29 import createIndices, closeWin
import numpy as np
import random

closeWin()

############### Global variables ###############
global stableSaveCount
global imgIdx


subjID_prep = 'pp'
# subjID_prep = str(11).zfill(2)
# exp_day_val = str(1).zfill(2) Not using leading zeros here

numRuns = 2 # E.g. 3 creates 3*8*50 = 1200 indices


catComb1 = [['male','female','indoor','outdoor'],
            ['indoor','outdoor','male','female']]

catComb2 = [['male','female','outdoor','indoor'],
            ['outdoor','indoor','male','female']]

catComb3 = [['female','male','indoor','outdoor'],
            ['indoor','outdoor','female','male']]

catComb4 = [['female','male','outdoor','indoor'],
            ['outdoor','indoor','female','male']]

############### PREP ONLY: INDICES FOR FUSING IMGS ###############
# Create fused images to display for 1 run (8 blocks)

randLst1 = [0,0,1,1]
randLst2 = [0,0,1,1]

# Generates the chosen combination for a subj

# Randomly assign 4 subjs to catComb1, 4 subjs to catComb2 etc.

for run in list(range(0,numRuns)):
    randCat  = (random.sample(randLst1, 4)) + (random.sample(randLst2, 4))
    for index in randCat:
        aDom = catComb3[index][0]
        aLure = catComb3[index][1]
        nDom = catComb3[index][2]
        nLure = catComb3[index][3]

        createIndices(aDom, aLure, nDom, nLure, subjID_forfile=subjID_prep,exp_day=3)
        
