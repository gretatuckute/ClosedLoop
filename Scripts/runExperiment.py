# -*- coding: utf-8 -*-
'''
The function, runNFday (from experimentFunctions) runs the neurofeedback system (experimental script synchronized with EEG recordings').
'''

# Imports
import os
from paths import script_path_init

script_path = script_path_init()
os.chdir(script_path)

# Load experimental functions
from experimentFunctions import * 

# Global variables
global blockIdx
global imgIdx

###### TEST RUNS ######
# runTest(day='1') # Show example of the behavioral paradigm
# runTest(day='2') # Show example of the neurofeedback paradigm

###### Behavioral run ######
# runBehDay(numRuns=2, numBlocks=8, blockLen=50) 

###### Neurofeedback run ######
runNFday(subjID='01', numRuns=5, numBlocks=8, blockLen=50)

