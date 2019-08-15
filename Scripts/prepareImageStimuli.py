# -*- coding: utf-8 -*-
'''
Generates .csv files for experimental stimuli (composite images). The .csv file contains 4 columns: attentive category name, binary category number, 
image 1 used in the composite image and image 2 used in the composite image (strings of file directories to the images). 
'''

# Imports
import os
import random
from paths import script_path_init

script_path = script_path_init()
os.chdir(script_path)

from experimentFunctions import createIndices, closeWin

closeWin()

#### Global variables ####
global blockIdx
global imgIdx

subjID_prep = '01'

numRuns = 6 # E.g. 6 runs creates 6*8*50 = 2400 composite images indices (string of path to directories of images)

# Different combinations of attentive versus non-attentive categories. Four in total.
catComb1 = [['male','female','indoor','outdoor'],
            ['indoor','outdoor','male','female']]

catComb2 = [['male','female','outdoor','indoor'],
            ['outdoor','indoor','male','female']]

catComb3 = [['female','male','indoor','outdoor'],
            ['indoor','outdoor','female','male']]

catComb4 = [['female','male','outdoor','indoor'],
            ['outdoor','indoor','female','male']]

# Create fused images to display for the specified number of runs (8 blocks per run as default)

randLst1 = [0,0,1,1]
randLst2 = [0,0,1,1]

# Generates a .csv file with the specified number of composite images for the specified subject (subjID_prep)

for run in list(range(0, numRuns)):
    randCat  = (random.sample(randLst1, 4)) + (random.sample(randLst2, 4))
    for index in randCat:
        aDom = catComb1[index][0]
        aLure = catComb1[index][1]
        nDom = catComb1[index][2]
        nLure = catComb1[index][3]

        createIndices(aDom, aLure, nDom, nLure, subjID_forfile=subjID_prep, exp_day=2) # Experimental day (exp_day) is for naming the .csv file.
        
