# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:08:54 2018

@author: gretatuckute

vol2.1: Possible to display fused images before saving them first (only based on indices)
vol2.2: Refines vol2.1 and adds breaks, probe
vol2.3: outlet
vol2.4: remove logging, fix indicesfile to start with python numbering (0)
vol2.5: add subjID info to .csv files, creating a list (timeLst) and .csv log for when a stimuli image is presented. 
Functions for displaying and drawing single images (not fused) from multiple categories.
vol2.6: add global subjID from info file. Add the possibility to change the subjID arg in createIndices. Added binary category column in
createIndices file, push imgIdx in runImage function instead of [1], added expDay info to createIndices function, runBehDay and runNFday functions added,
save alpha to lst and csv, save imgIdx to lst and csv, add read_alpha_stream function, add alpha inlet search in runNFday function,
added autoLog again to imagestim (EXP mode), 

vol2.9. Manual subject and expDay info 
vol3.0: remove not used code, new paths

"""
# Imports
import os
import glob
import sys  
from PIL import Image
import random
from random import sample
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, resolve_byprop
from random_1subject import assign_subject,add_avail_feedback_sub,gen_group_fname

import numpy as np
from psychopy import gui, visual, core, data, event, monitors, logging
import time 
import csv
import pandas as pd
from paths import data_path_init, subject_path_init, script_path_init

# Paths
data_path = data_path_init()
subject_path = subject_path_init()
script_path = script_path_init()

os.chdir(subject_path)

######### Input subject ID and experiment day #######

subjID = '90'
expDay = '1' # '2': feedback day


############### Global variables ###############

global stableSaveCount 
global imgIdx

stableSaveCount = 1 
imgIdx = 0

############### Data frames for logging ###############
df_stableSave = pd.DataFrame(columns=['attentive_cat','binary_cat','img1', 'img2']) # For createIndices function
df_timeLst = pd.DataFrame()
df_alphaLst = pd.DataFrame()
df_meanAlphaLst = pd.DataFrame()

timeLst = []
imgIdxLst = []
alphaLst = []
alphaLstSave = []
meanAlphaLst = []


############### Outlet ###############
info = StreamInfo('PsychopyExperiment20', 'Markers', 1, 0, 'int32', 'myuidw43536')
outlet = StreamOutlet(info)

############### Experimental initialization ###############

# Initializing window
#win = visual.Window(size=[1910, 1070], fullscr=True,winType='pyglet',allowGUI=True,screen=1,units='pix') #GPU

win = visual.Window(size=[900, 400], fullscr=False,winType='pyglet',allowGUI=True,screen=1,units='pix') #GPU


# Initializing fixation text and probe word text 
textFix = visual.TextStim(win=win, name='textFix', text='+', font='Arial',units='pix',height=42,
                                pos=(0, 0), wrapWidth=None, ori=0,
                                color='white', colorSpace='rgb', opacity=1, depth=-1.0)


# Initializing stimuli presentation times (in Hz)
frameRate = 60
probeTime = 360
fixTime = 120
stimTime = 60 # stimuli time for presenting each image

# Initialization of button press
globalClock = core.Clock() 
keys = event.getKeys(keyList=None, timeStamped=globalClock)

# Prepare PsychoPy log
log_base = time.strftime('%m-%d-%y_%H-%M')
logWritePath = subject_path + '\\' + subjID + '\\PsychoPyLog_subjID_' + str(subjID) + '_day_' + str(expDay) + '_' + str(log_base) + '.csv'

logWritePathKey = subject_path + '\\' + subjID + '\\keypress_subjID_' + str(subjID) + '_day_' + str(expDay) + '_' + str(log_base) + '.csv'

logging.LogFile(logWritePath, level=logging.EXP, filemode='w')
logging.console = True

logging.LogFile(logWritePathKey, level=logging.DATA, filemode='w') # Log file for button press only
logging.setDefaultClock(globalClock)
logging.console = True

def log(msg):
    """Prints messages in the promt and adds msg to PsychoPy log file. """ 
    logging.log(level=logging.EXP, msg=msg) 

    
def closeWin():
    """Closes the pre-defined experimental window (win). """
    win.close()
    
if event.getKeys(keyList=["escape"]):
    closeWin()


############### Experiment functions ###############

def findCategories(directory):
    """Returns the overall category folders in /data/.
    
    # Arguments
        directory: Path to the data (use data_path)
        
    # Returns
        found_categories: List of overall categories
    """
    
    found_categories = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            found_categories.append(subdir)
    return found_categories

def recursiveList(directory):
    """Returns list of tuples. Each tuple contains the path to the folder 
    along with the names of the images in that folder.
    
    # Arguments
        directory: Path to the data.
        
    # Returns
        list to use in findImages to iterate through folders
    """
    
    follow_links = False
    return sorted(os.walk(directory, followlinks=follow_links), key=lambda tpl: tpl[0])

def findImages(directory):
    """Returns the number of images and categories (subcategories) in a given folder.
    
    # Arguments
        directory: Path to the data.
        
    # Returns
        noImages: Total number of images in each category folder
        imagesInEachCategory = dictionary of size 2 (if 2 categories), with key = category name, and value = list containing image paths
    """

    imageFormats = {'jpg', 'jpeg', 'pgm'}

    categories = findCategories(directory)
    noCategories = len(categories) # Total number of folders/categories in the given folder - currently not dierctly returned

    noImages = 0
    imagesInEachCategory = {}
    for subdir in categories:
        subpath = os.path.join(directory, subdir)
        for root, _, files in recursiveList(subpath):
            for fname in files:
                is_valid = False
                for extension in imageFormats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    noImages += 1
                imagesInEachCategory[subdir] = [subpath + '\\' + ii for ii in files]
    # print('Found %d images belonging to %d categories.\n' % (noImages, noCategories-1))
    return noImages, imagesInEachCategory


def createRandomImages(dom='female',lure='male'):
    """Takes two input categories, and draws 45 images from the dominant category, and 5 from the lure category
    
    # Returns
        fusedList: One list consisting of 50 images from dominant and lure categories in random order
        fusedCats: One list of the category names from fusedList (used in directory indexing)
    """
    
    categories = findCategories(data_path) 
    noImages, imagesInEachCategory = findImages(data_path)
    
    for key, value in imagesInEachCategory.items():        
        if key == dom:
            randomDom = sample(value, 45) # Randomly takes X no of samples from the value list corresponding to that category
        if key == lure:
            randomLure = sample(value, 5)
            
    fusedList = randomDom + randomLure
    random.shuffle(fusedList)
    
    fusedCats = []
    for item in fusedList:
        str_item = str(item)
        str_item = str_item.split('\\')
        fusedCats.append(str_item[-2])
    
    return fusedList, fusedCats


def createIndices(aDom, aLure, nDom, nLure, subjID_forfile=subjID, exp_day=1): 
    """ Returns 50 fused images (only the indices) based on a chosen attentive category, and a chosen unattentive category.
        It is possible to save the fused images in a folder named after the attentive category:
            Uncomment code in this function
    
    # Input:
    - aDom, aLure, nDom, nLure: names of attentive dominant category, attentive lure category, nonattentive dominant category and nonattentive lure category.
    - subjID_forfile: subject ID used for naming the .csv file with indices.
    - exp_day: experimental day for naming the .csv file with indices.
    
    # Output:
    No direct output, but writes to a .csv file with 4 columns: attentive category name, binary category number, image 1 used in the composite
    image and image 2 used in the composite image. Writes to a .csv file in log_file_path with the title "createIndices + subjID_forfile + exp_day.csv".
    
    """

    aCatImages, aCats = createRandomImages(dom=aDom,lure=aLure) # 50 attend category images
    nCatImages, nCats = createRandomImages(dom=nDom,lure=nLure) 
    
    # FOR SAVING IMAGES IN FOLDERS
    # Save information about the attentive category for later probe word text display
#    if aDom == 'male' or aDom == 'female':
#        aFolder = 'faces' # Attentive folder name
#
#    if aDom == 'indoor' or aDom == 'outdoor':
#        aFolder = 'scenes'

    # Save binary category. 0 for scenes, 1 for faces

    if aDom == 'indoor' or aDom == 'outdoor':
        binary_cat = 0
    if aDom == 'female' or aDom == 'male':
        binary_cat = 1

    imageCount = 0
    global stableSaveCount
    
    for i in range(len(aCatImages)):
        
        background = Image.open(os.path.join(aCatImages[i]), mode='r')
        foreground = Image.open(os.path.join(nCatImages[i]))
        
        # FOR SAVING
        # fusedImage = Image.blend(background, foreground, .5)
        # fusedImage.save(stable_path + '\\' + aFolder + str(stableSaveCount) + '\\' 'no_' + str(imageCount) + '.jpg')
    
        background.close()
        foreground.close()
        # fusedImage.close()
    
        imageCount += 1
        
        # Logging the image paths/IDs to a df
        df_stableSave.loc[i + (stableSaveCount * len(aCatImages))-50] = [aDom, binary_cat, aCatImages[i], nCatImages[i]] 
        
    print('Created {0} fused image indices, with a total no. of blocks currently created: {1}.\n'.format(len(aCatImages), str(stableSaveCount)))

    df_stableSave.to_csv(subject_path + '\\createIndices_' + str(subjID_forfile) + '_day_' + str(exp_day) + '.csv') 

    stableSaveCount += 1
    
    del aDom, aLure, nDom, nLure
    
def createIndicesNew(aDom, aLure, nDom, nLure, subjID_forfile=subjID, exp_day=1): 
    """ Returns 50 fused images (only the indices) based on a chosen attentive category, and a chosen unattentive category.
        It is possible to save the fused images in a folder named after the attentive category:
            Uncomment code in this function
    
    # Input:
    - aDom, aLure, nDom, nLure: names of attentive dominant category, attentive lure category, nonattentive dominant category and nonattentive lure category.
    - subjID_forfile: subject ID used for naming the .csv file with indices.
    - exp_day: experimental day for naming the .csv file with indices.
    
    # Output:
    No direct output, but writes to a .csv file with 4 columns: attentive category name, binary category number, image 1 used in the composite
    image and image 2 used in the composite image. Writes to a .csv file in log_file_path with the title "createIndices + subjID_forfile + exp_day.csv".
    
    """

    aCatImages, aCats = createRandomImages(dom=aDom,lure=aLure) # 50 attend category images
    nCatImages, nCats = createRandomImages(dom=nDom,lure=nLure) 

    if aDom == 'bus' or aDom == 'airplane':
        binary_cat = 0
    if aDom == 'cat' or aDom == 'dog':
        binary_cat = 1

    imageCount = 0
    global stableSaveCount
    
    for i in range(len(aCatImages)):
        
        background = Image.open(os.path.join(aCatImages[i]), mode='r')
        foreground = Image.open(os.path.join(nCatImages[i]))
 
        background.close()
        foreground.close()
        # fusedImage.close()
    
        imageCount += 1
        
        # Logging the image paths/IDs to a df
        df_stableSave.loc[i + (stableSaveCount * len(aCatImages))-50] = [aDom, binary_cat, aCatImages[i], nCatImages[i]] 
        
    print('Created {0} fused image indices, with a total no. of blocks currently created: {1}.\n'.format(len(aCatImages), str(stableSaveCount)))

    df_stableSave.to_csv(subject_path + '\\createIndices_' + str(subjID_forfile) + '_day_' + str(exp_day) + '.csv') 

    stableSaveCount += 1
    
    del aDom, aLure, nDom, nLure


def createNonFusedIndices(mixture=[12,13,12,13],no_blocks=8,no_runs=3):
    """ Returns indices of non-fused images (random order) in a .csv file.
    
    # Arguments:
        mixture: Assuming a block size of 50 and 4 categories, the mixture is a list consisting of ratios for the different categories.
        no_blocks: number of blocks to run. Block size depends on the mixture list. Currently set to 50.
        no_runs: number of runs to complete of the blocks.
        
    # Returns
        Creates a .csv file with image indices as well as overall image category (faces/scenes), saved to the log path. 
    """

    allImages = [[],[],[],[]]
        
    dir_male = data_path + '\\male'
    dir_female = data_path + '\\female'
    dir_indoor = data_path + '\\indoor'
    dir_outdoor = data_path + '\\outdoor'
    
    all_dirs = [dir_male,dir_female,dir_indoor,dir_outdoor]
    
    for count, directory in enumerate(all_dirs):
        for file in os.listdir(directory):
            if file.lower().endswith(".jpg"):
                allImages[count].append(directory + '\\' + file)
    
    randomImages = []

    noIterations = no_blocks * no_runs

    for run in range(no_runs):
        for blocks in range(no_blocks):
            
            for count, number in enumerate(mixture):
                randomImg = random.sample(allImages[count],number)
                randomImages.append(randomImg)
            
    randomImages = np.concatenate(randomImages).ravel()
    np.random.shuffle(randomImages)
    
    randomImages.tolist()
    
    catLst = []
    catLst2 = []
    
    for entry in randomImages:
        imgName = np.char.split(entry,sep='.')
        imgName = imgName.tolist()
        imgName = imgName[0]
        imgName = imgName.split('\\')
        catName = imgName[-1]
        print(catName)
        
        wordLst = []
        
        for char in catName:
            if char.isalpha() is True:
                wordLst.append(char)
        catLst.append(wordLst)
            
    for entry in catLst:
        new = ''.join(entry)
        catLst2.append(new)

    df_nonFused = pd.DataFrame(np.array(randomImages),catLst2)
    df_nonFused.to_csv(subject_path + '\\createIndicesNonFused_' + str(subjID) + '.csv') 
    

def fuseImage(csvfile,alpha=0.5):
    """Returns a single fused image based on a pre-generated csv file with indices of images.
    
    # Arguments:
        csvfile: containing indices of which images to fuse in column 2 and 3, and name of the attentive category in column 1
        alpha: blending proportion. Default 0.5 for stable blocks.
        
    # Output
        Calls runImage, displaying the fused image    
    """
    
    # Read from csv file here and use the img IDs as input
    global imgIdx
    
    with open(csvfile) as csv_file:
    #with open(log_path + '\\createIndices.csv') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
    
    # Implement some kind of global count that knows which row to take
        wantedRow = imgIdx + 1 # counter...
        
        rowInfo = csv_reader[wantedRow]
        # print(rowInfo)
        
        foregroundID = rowInfo[3]
        backgroundID = rowInfo[4]
            
    foreground = Image.open(foregroundID, mode='r')
    background = Image.open(backgroundID, mode='r')
    
    fusedImg = Image.blend(background, foreground, alpha)
    
    # fusedImg.show()
    runImage(fusedImg)
    
    imgIdxLst.append(imgIdx)
    
    imgIdx += 1
    
    background.close()
    foreground.close()
    
def fuseImageSlow(csvfile,alpha=0.5):
    """Returns a single fused image based on a pre-generated csv file with indices of images.
    
    # Arguments:
        csvfile: containing indices of which images to fuse in column 2 and 3, and name of the attentive category in column 1
        alpha: blending proportion. Default 0.5 for stable blocks.
        
    # Output
        Calls runImage, displaying the fused image    
    """
    
    # Read from csv file here and use the img IDs as input
    global imgIdx
    
    with open(csvfile) as csv_file:
    #with open(log_path + '\\createIndices.csv') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
    
    # Implement some kind of global count that knows which row to take
        wantedRow = imgIdx + 1 # counter...
        
        rowInfo = csv_reader[wantedRow]
        # print(rowInfo)
        
        foregroundID = rowInfo[3]
        backgroundID = rowInfo[4]
            
    foreground = Image.open(foregroundID, mode='r')
    background = Image.open(backgroundID, mode='r')
    
    fusedImg = Image.blend(background, foreground, alpha)
    
    # fusedImg.show()
    runImageSlow(fusedImg)
    
    imgIdxLst.append(imgIdx)
    
    imgIdx += 1
    
    background.close()
    foreground.close()
    
def prepNonFusedImage(csvfile):
    """Returns a single non-fused image based on a pre-generated csv file with indices of images.
    
    # Arguments:
        csvfile: containing indices of which image to display. Based on global imgIdx counter.
        
    # Output
        Calls runImage, displaying the image in the pre-selected window (win).
    """
    
    # Read from csv file here and use the img IDs as input
    global imgIdx
    
    with open(csvfile) as csv_file:
    #with open(log_path + '\\createIndices.csv') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
    
    # Implement some kind of global count that knows which row to take
        wantedRow = imgIdx # counter...
        
        rowInfo = csv_reader[wantedRow]
        img = rowInfo[1]
        
    imgIdx += 1
        
    runImage(img)
    

def runImage(fusedImg):
    """Runs a single fused image based on output from fuseImage in the defined experimental window (win).
    
    # Arguments:
        The fused image (from fuseImage function)
    
    # Output:
        Shows the image in a predefined no. of Hz
    """
    
    image = visual.ImageStim(win, image = fusedImg, autoLog=True,units='pix',size=(175,175))
    
    outlet.push_sample([imgIdx])
    t = globalClock.getTime()
    timeLst.append(t)
    
    for frameNew in range(0,stimTime):
        if frameNew >= 0:
            image.draw()
        win.flip()
    
def runImageSlow(fusedImg):
    """Runs a single fused image based on output from fuseImage in the defined experimental window (win).
    
    # Arguments:
        The fused image (from fuseImage function)
    
    # Output:
        Shows the image in a predefined no. of Hz
    """
    
    image = visual.ImageStim(win, image = fusedImg, autoLog=True,units='pix',size=(175,175))
    
    outlet.push_sample([imgIdx])
    t = globalClock.getTime()
    timeLst.append(t)
    
    for frameNew in range(0,120):
        if frameNew >= 0:
            image.draw()
        win.flip()
        
def runBreak(breakLen,message):
    """Runs a break the defined experimental window (win).
    
    # Arguments:
        breakLen: in Hz
    
    # Output:
        Displays the break in a predefined no. of Hz
    """
    
    message = str(message)
    
    textBreak = visual.TextStim(win=win, name='textBreak', text=message, font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1, depth=0.0)

    for frameNew in range(0,breakLen): 
        textBreak.draw()
        win.flip()



def runFixProbe(csvfile):
    """Displays fixation cross and probe word text in the defined experimental window (win).
    
    # Arguments:
        csvfile used for the image indices (containing name of attentive category in column 1)
    
    # Output:
        Displays fixation and probe text in a predefined no. of Hz
    """
    
    with open(csvfile) as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        
        wantedRow = imgIdx + 1
        
        rowInfo = csv_reader[wantedRow]
        
        attentiveText = str(rowInfo[1])
        
    textGeneral = visual.TextStim(win=win, name='textGeneral', text=attentiveText, font='Arial',
                             units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                             color='white', colorSpace='rgb', opacity=1, depth=0.0)    
    
    #outlet.push_sample([3])
    for frameN in range(0,probeTime):
        textGeneral.draw()
        win.flip() 
        
    for frameN in range(0,fixTime):
        textFix.draw()
        win.flip() 

    
def saveDataFrame(subjID):
    """
    
    """
    df_timeLst = pd.DataFrame(timeLst,imgIdxLst)
    df_timeLst.to_csv(subject_path + '\\' + subjID + '\\imageTime_subjID_' + str(subjID) + '_day_' + str(expDay) + '_' + str(log_base) + '.csv') 
    
    
def saveAlphaDataFrame(subjID):
    """
    
    """
    df_alphaLst = pd.DataFrame(alphaLst)
    df_alphaLst.to_csv(subject_path + '\\' + subjID + '\\alpha_subjID_' + str(subjID) + '.csv') 
    
    
def saveMeanAlphaDataFrame(subjID):
    """
    
    """
    df_meanAlphaLst = pd.DataFrame(meanAlphaLst)
    df_meanAlphaLst.to_csv(subject_path + '\\' + subjID + '\\MEANalpha_subjID_' + str(subjID) + '_'  + '_day_' + str(expDay) + str(log_base) + '.csv') 
    
def read_marker_stream(stream_name ='alphaStream'):
    '''
    Reads stream from an outlet (alphaStream from run_CL script)
    
    '''
    index_alpha = []
    alpha_created = []
    streams = resolve_byprop('type', 'Markers',timeout=10)
    for i in range (len(streams)):
        if (streams[i].name() == stream_name):
            index_alpha.append(i)
            alpha_created.append(streams[i].created_at())
    if index_alpha:
        if len(index_alpha)>1:
            #print("Not unique marker stream name, using most recent one")
            index_alpha=index_alpha[np.argmax(alpha_created)]
        #print ("alpha stream available")
        #inlet = StreamInlet(streams[index_alpha[0]]) #REAL ONE
        inlet = StreamInlet(streams[index_alpha[0]])

        alpha_avail=1
#        store_marker=data_init(500,'marker')
#        store_marker.header=['Marker','Timestamp']
    else:
        inlet_marker=[]
        print('Warning: No marker inlet available')
    return inlet

def runTest(day='1'):
    '''
    Runs a demo version of the experimental script (either day 1 and or 3) without EEG recordings.
    GENERATE FILES FROM GPU comp
    Change break time
    Add an alpha file 
    
    # Input
    - day: either 1 or 2. Day 2 illustrates neurofeedback blocks.
    
    '''    
    
    time.sleep(5)

    numBlocks = 2
    blockLen = 10

    runLen = numBlocks * blockLen 
    
    subjID = '99'
    expDay = day
    
    if expDay == '1':
    
        for run in list(range(1)):
            for ii in list(range(0,runLen)):
                if ii % blockLen == 0: 
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
                    
                fuseImageSlow(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
                
                if ii == runLen-1:
                    runBreak(600,'10 second break')
                                               
                    closeWin()
                
    if expDay == '2':
        
        alphafile = subject_path + '\\' + subjID + '\\alphamock.csv'
        with open(alphafile) as csv_file:
            alphaLst = list(csv.reader(csv_file, delimiter=','))
            del alphaLst[0]
        
        alphaIdx = 0
        runLenHalf = 20
        
        for ii in list(range(0,runLenHalf)): # The first 4 blocks
            
            if ii == 0: 
               runBreak(180,'Recording stable blocks')
            
            if ii % blockLen == 0: # Shows the category text probe
                runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
                
            fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv',alpha=0.5)
            
            if ii == runLenHalf-1: # Break between the stable blocks of NF runs
                runBreak(600,'10 second break') # Remove this break? Or, this is when the clf is trained
                            
        for jj in list(range(0,runLenHalf)): # The last 4 blocks
 
            if jj == 0: 
               runBreak(180,'Neurofeedback is starting')
           
            if jj % blockLen == 0: # Shows the category text probe
                runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
       
            if jj in [0,1,10]: # First 3 trials are stable. Currently hardcoded.
                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv',alpha=0.5)
                alphaIdx += 1
                
            else:
               
                alphaVal = alphaLst[alphaIdx][0]
                alphaVal = float(alphaVal)
        
                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv',alpha=alphaVal)
                
                alphaIdx += 1
            
            if jj == runLenHalf-1: # Break after a finished run
                                    
                runBreak(300,'Experiment finished')
                closeWin()


def runBehDay(numRuns=2,numBlocks=8,blockLen=50,day=1):
    '''
    Runs the experimental script for behavioral days (1 and 3) without EEG recordings.
    
    numRuns = 2
    
    '''
    if day==1:
        assign_subject(subjID)
    
    
    time.sleep(25)

    runLen = numBlocks * blockLen # Should be 8 * 50
    runLenHalf = runLen/2
        
    for run in list(range(0,numRuns)):
        for ii in list(range(0,runLen)):
            if ii == runLenHalf:
                runBreak(1800,'30 second break')
                
            if ii % blockLen == 0: 
                runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
            
                
            fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
            
            if ii == runLen-1 and run == 0:
                runBreak(1800,'30 second break')
                
        if run == numRuns-1:
            runBreak(300,'This part of the experiment finished')
            saveDataFrame(subjID) # Saves the timing of image stimuli 
            
            closeWin()
            


def runNFday(subjID,numRuns,numBlocks,blockLen):
    '''
    Runs the experimental script for neurofeedback real-time EEG recordings.
    
    # Input
    - numRuns: number of NF runs (besides the first, stable run)
    
    '''
    
    import copy
    
    time.sleep(30)
    
    runLen = numBlocks * blockLen # Should be 8 * 50
    runLenHalf = int(runLen/2)
    
    control_file = open(subject_path + '\\' + subjID + '\\feedback_subjID' + str(subjID) + '.txt','r')
    control = [x.rstrip("\n") for x in control_file.readlines()]

     ### CONTROL SUBJECTS ###
    if control[0] == '0':
        subj_orig=copy.copy(subjID) #True copy
        
        subjID=control[1]
        alphafile = subject_path + '\\' + subjID + '\\alpha_subjID_' + str(subjID) + '.csv'

        with open(alphafile) as csv_file:
            alphaLst2 = list(csv.reader(csv_file, delimiter=','))
            del alphaLst2[0]
            
            alphaLstMock = [item[1] for item in alphaLst2]
        
        alphaIdx = 0
        
        # Stable run
        for run in range(1):
            for ii in list(range(0,runLen)):
                
                if ii == 0: 
                   runBreak(180,'Recording stable blocks')
                   
                if ii == runLenHalf:
                    runBreak(1800,'30 second break')
                
                if ii % blockLen == 0: # Correct: ii % 50?
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')

                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')#,alpha=randomAlpha)    
                
                if ii == runLen-1:
                    runBreak(1800,'30 second break')
                                    
        # NF runs (consisting of stable and NF)
        for run in list(range(0,numRuns)):
            for ii in list(range(0,runLenHalf)): # The first 4 blocks
                
                if ii == 0: 
                   runBreak(180,'Recording stable blocks')
                
                if ii % blockLen == 0: # Shows the category text probe
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')

                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')#,alpha=randomAlpha)
                
                if ii == runLenHalf-1: # Break between the stable blocks of NF runs
                    runBreak(1800,'30 second break') # This is when the clf is trained
        
            # Train classifier and find alpha stream - MOCK
            
            for jj in list(range(0,runLenHalf)):
                
                if jj == 0:
                    runBreak(180,'Neurofeedback is starting')
                    
                if jj % blockLen == 0:
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
                    
                if jj in [0,1,2,50,51,52,100,101,102,150,151,152]: # First 3 trials are stable. Currently hardcoded.
                    fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv',alpha=0.5)
                    alphaVal = alphaLstMock[alphaIdx]
                    alphaVal = float(alphaVal) 
                    alphaLst.append(alphaVal)
                    alphaIdx += 1
                    
                else:
                    alphaVal = alphaLstMock[alphaIdx]
                    alphaVal = float(alphaVal) 
                                
                    alphaIdx += 1

                    alphaLst.append(alphaVal)
        
                    print('alphaVal: ' + str(alphaVal))
                    print('current imgIdx: ' + str(imgIdx))
                    
                    # Find the mean alpha value based on where in the list the alphaIdx is 
                    mean_alpha = np.mean(alphaLst[-3:])
                    meanAlphaLst.append(mean_alpha)
                    
                    fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv',alpha=mean_alpha)
                    
                if jj == runLenHalf-1:
                    
                    if run != numRuns-1:
                        
                        runBreak(1800,'30 second break')  
                        
            if run == 1:
                runBreak(360,'Experiment halfway')
                    
            if run == numRuns-1:  
                runBreak(300,'Experiment finished')
                saveDataFrame(subj_orig) # Saves the timing of image stimuli 
                saveAlphaDataFrame(subj_orig) # Saves alpha values used in the NF experiment
                saveMeanAlphaDataFrame(subj_orig) #TYPE ERROR, har rettet
                
                closeWin()
        
    
    ### NF SUBJECTS ###
    if control[0] == '1':
        # proceed onto the rest
        subjID = control[1]
        
        # Stable run
        for run in range(1):
            for ii in list(range(0,runLen)):
                
                if ii == 0: 
                   runBreak(180,'Recording stable blocks')
                   
                if ii == runLenHalf:
                    runBreak(1800,'30 second break')
                
                if ii % blockLen == 0: # Correct: ii % 50?
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')

                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')#,alpha=randomAlpha)    
                if ii == runLen-1:
                    runBreak(1800,'30 second break')
                                    
        # NF runs (consisting of stable and NF)
        for run in list(range(0,numRuns)):
            for ii in list(range(0,runLenHalf)): # The first 4 blocks
                
                if ii == 0: 
                   runBreak(180,'Recording stable blocks')
                
                if ii % blockLen == 0: # Shows the category text probe
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
                    
                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')#,alpha=randomAlpha)
                
                if ii == runLenHalf-1: # Break between the stable blocks of NF runs
                    runBreak(1800,'30 second break') # This is when the clf is trained
        
                    
            # Train classifier and find alpha stream
            inlet_alpha = read_marker_stream(stream_name ='alphaStream')
            
            for jj in list(range(0,runLenHalf)): # The last 4 blocks
    
                if jj == 0: 
                    runBreak(180,'Neurofeedback is starting')
               
                if jj % blockLen == 0: # Shows the category text probe
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
           
                if jj in [0,1,2,50,51,52,100,101,102,150,151,152]: # First 3 trials are stable. Currently hardcoded.
                    alphamarker,timestamp = inlet_alpha.pull_chunk(timeout=0.005)
                   
                    if len(alphamarker):
                       alphaVal = alphamarker[-1][0]
                    else:
                       alphaVal = 0.5
                       
                    alphaLst.append(alphaVal)
                    fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv',alpha=0.5)
                    
                else:
                    alphamarker,timestamp  = inlet_alpha.pull_chunk(timeout=0.005)
                    
                    if len(alphamarker):
                       alphaVal = alphamarker[-1][0]
                    else:
                       alphaVal = 0.5
                        
                    alphaLst.append(alphaVal)
                              
                    print('alphaVal: ' + str(alphaVal))
                    print('current imgIdx: ' + str(imgIdx))
                       
                    mean_alpha = np.mean(alphaLst[-3:])
                    meanAlphaLst.append(mean_alpha)
                    
                    fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv',alpha=mean_alpha) 
                
                if jj == runLenHalf-1: # Break after a finished run
                    
                    if run != numRuns-1: # Avoid a break after the very last run 
                    
                        runBreak(1800,'30 second break')
                    
            if run == 1:
                runBreak(300,'Experiment halfway')

                
            if run == numRuns-1:  
                runBreak(300,'Experiment finished')
                saveDataFrame(subjID) # Saves the timing of image stimuli 
                saveAlphaDataFrame(subjID) # Saves alpha values used in the NF experiment
                saveMeanAlphaDataFrame(subjID)
                group_fname=gen_group_fname(subjID)
                add_avail_feedback_sub(subjID,group_fname)

                
                closeWin()


