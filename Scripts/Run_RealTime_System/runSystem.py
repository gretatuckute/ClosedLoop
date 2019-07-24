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

# Load experimental functions
from experimentFunctions import * 

# Global variables
global blockIdx
global imgIdx

### TEST RUNS ###
# runTest(day='1') # Show example of 
# runTest(day='2')

###  Behavioral run ### 
# runBehDay(day=1) 

### Neurofeedback run ### 
runNFday(subjID='90',numRuns=5,numBlocks=8,blockLen=50)

