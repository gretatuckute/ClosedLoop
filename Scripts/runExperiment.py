# -*- coding: utf-8 -*-
'''
The function, runNFday (from experimentFunctions) runs the neurofeedback system (experimental script synchronized with EEG recordings').
'''

# Imports
import os
import settings

script_path = settings.script_path_init()
os.chdir(script_path)

# Load experimental functions
import experimentFunctions

# Global variables
global blockIdx
global imgIdx

###### TEST RUNS ######
# experimentFunctions.runTest(day='1') # Show example of the behavioral paradigm
# experimentFunctions.runTest(day='2') # Show example of the neurofeedback paradigm

###### Behavioral run ######
# experimentFunctions.runBehDay(numRuns=settings.numRuns, numBlocks=settings.numBlocks, blockLen=settings.blockLen) 

###### Neurofeedback run ######
experimentFunctions.runNFday(subjID=settings.subjID, numRuns=settings.numRuns, numBlocks=settings.numBlocks, blockLen=settings.blockLen)

