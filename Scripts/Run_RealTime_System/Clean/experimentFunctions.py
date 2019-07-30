# -*- coding: utf-8 -*-
'''
Functions for running the PsychoPy experimental script and generating stimuli (composite images) for the experiment.
The script creates and outlet for sending markers (trigger points) for each stimuli onset.
'''

# Imports
import os
import numpy as np
import time 
import csv
import pandas as pd

from PIL import Image
import random
from random import sample
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, resolve_byprop
from randomizeParticipants import assign_subject, add_avail_feedback_sub, gen_group_fname

from psychopy import visual, core, data, event, monitors, logging
from paths import data_path_init, subject_path_init, script_path_init

# Initialize paths
data_path = data_path_init()
subject_path = subject_path_init()
script_path = script_path_init()

os.chdir(subject_path)

###### Input subject ID and experimental day ######

subjID = '01'
expDay = '2' # '2': feedback day

###### Global variables ######

global blockIdx 
global imgIdx

blockIdx = 1 
imgIdx = 0

###### Data frames for logging ######
df_imgIdxSave = pd.DataFrame(columns=['attentive_cat','binary_cat','img1', 'img2']) 
# For createIndices function. Columns: 
# 1) Name of attentive category
# 2) Binary category number
# 3) Path to image 1
# 4) Path to image 2 

df_timeLst = pd.DataFrame() # For saving the time points where experimental stimuli were shown in the experiment
df_alphaLst = pd.DataFrame() # For saving alpha values used in the experiment
df_meanAlphaLst = pd.DataFrame() # For saving meaned alpha values used in the experiment

timeLst = []
imgIdxLst = []
alphaLst = []
alphaLstSave = []
meanAlphaLst = []

###### Outlet ######
info = StreamInfo('PsychopyExperiment', 'Markers', 1, 0, 'int32', 'myuidw43536') # Has to match the experiment name in runClosedLoop.py script
outlet = StreamOutlet(info)

###### Experiment initialization ######

# Initializing window
win = visual.Window(size=[1910, 1070], fullscr=True, winType='pyglet', allowGUI=True, screen=1, units='pix') 

# Initializing fixation text and probe word text 
textFix = visual.TextStim(win=win, name='textFix', text='+', font='Arial',units='pix',height=42,
                                pos=(0, 0), wrapWidth=None, ori=0,
                                color='white', colorSpace='rgb', opacity=1, depth=-1.0)


# Initializing stimuli presentation times (in Hz)
frameRate = 60
probeTime = 360 # frames to display probe word
fixTime = 120 # frames to display fixation cross
stimTime = 60 # frames for presenting each image

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
    '''Prints messages in the promt and adds msg to PsychoPy log file. '''
    logging.log(level=logging.EXP, msg=msg) 
    
def closeWin():
    '''Closes the pre-defined experimental window (win). '''
    win.close()
    
###### Experiment functions ######

def findCategories(directory):
    '''
    Returns the category folders in the given input directory.
    
    # Arguments
        directory: string
            Path to the data (use data_path).
        
    # Returns
        found_categories: list
            List of overall categories.
    '''
    
    found_categories = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            found_categories.append(subdir)
    return found_categories

def recursiveList(directory):
    '''
    Returns list of tuples. Each tuple contains the path to the folder along with the names of the images in that folder.
    
    # Arguments
        directory: string
            Path to the data.
        
    # Returns
        os.walk: function
        Function for iterating through folders (to use in findImages).
    '''
    
    follow_links = False
    return sorted(os.walk(directory, followlinks=follow_links), key=lambda tpl: tpl[0])

def findImages(directory):
    '''
    Returns the number of images and categories (subcategories) in a given folder.
    
    # Arguments
        directory: string
            Path to the data.
        
    # Returns
        noImages: int
            Total number of images in each category folder.
        imagesInEachCategory: dict
            Dictionary of size 2 (if 2 categories), with key = category name, and value = list containing image paths.
    '''

    imageFormats = {'jpg', 'jpeg', 'pgm'}

    categories = findCategories(directory)
    # noCategories = len(categories) # Total number of folders/categories in the given folder - currently not returned

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


def createRandomImages(dom='female', lure='male'):
    '''
    Takes two input categories, and draws 45 images from the dominant category, and 5 from the lure category.
    
    # Arguments
        dom: string
            Category name of dominant image category.
        lure: string
            Category name of non-dominant (lure) image category.
    
    # Returns
        fusedList: list
            List consisting of 50 images from dominant and lure categories in random order.
        fusedCats: list
            List of the category names from fusedList (used in directory indexing).
    '''
    
    # categories = findCategories(data_path) 
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
    '''
    Creates 50 fused images (only the indices/string of paths) based on a chosen attentive category, and a chosen unattentive category.
    It is possible to save the fused images in a folder named after the attentive category: Uncomment code in this function.
    
    # Arguments
        aDom, aLure, nDom, nLure: string
            Names of attentive dominant category, attentive lure category, non-attentive dominant category and non-attentive lure category.
        subjID_forfile: string
            Subject ID used for naming the .csv file with indices.
        exp_day: string
            Experimental day for naming the .csv file with indices.
    
    # Returns
        No direct output, but writes to a .csv file with 4 columns: attentive category name, binary category number, image 1 used in the composite
        image and image 2 used in the composite image. 
    
    Writes to a .csv file in log_file_path with the title "createIndices + subjID_forfile + exp_day.csv".
    '''

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
    global blockIdx
    
    for i in range(len(aCatImages)):
        
        background = Image.open(os.path.join(aCatImages[i]), mode='r')
        foreground = Image.open(os.path.join(nCatImages[i]))
        
        # FOR SAVING
        # fusedImage = Image.blend(background, foreground, .5)
        # fusedImage.save(stable_path + '\\' + aFolder + str(blockIdx) + '\\' 'no_' + str(imageCount) + '.jpg')
    
        background.close()
        foreground.close()
        # fusedImage.close()
    
        imageCount += 1
        
        # Logging the image paths/IDs to a df
        df_imgIdxSave.loc[i + (blockIdx * len(aCatImages))-50] = [aDom, binary_cat, aCatImages[i], nCatImages[i]] 
        
    print('Created {0} fused image indices, with a total no. of blocks currently created: {1}.\n'.format(len(aCatImages), str(blockIdx)))

    df_imgIdxSave.to_csv(subject_path + '\\createIndices_' + str(subjID_forfile) + '_day_' + str(exp_day) + '.csv') 

    blockIdx += 1
    
    del aDom, aLure, nDom, nLure


def createNonFusedIndices(mixture=[12,13,12,13], no_blocks=8, no_runs=3):
    ''' 
    Creates indices of non-fused images (random order) in a .csv file.
    
    # Arguments
        mixture: list
            Assuming a block size of 50 and 4 categories, the mixture is a list consisting of ratios for the different categories.
        no_blocks: int
            Number of blocks to run. Block size depends on the mixture list. Currently set to 50.
        no_runs: int
            Number of runs.
        
    # Returns
        No direct output, but writes to a .csv file with image indices as well as overall image category (faces/scenes), saved to the log path. 
    '''

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
    

def fuseImage(csvfile, alpha=0.5):
    '''
    Returns a single fused image based on a pre-generated .csv file with indices of images.
    
    # Arguments
        csvfile: string
            String path containing indices of which images to fuse in column 2 and 3, and name of the attentive category in column 1.
        alpha: float
            Blending proportion. Default 0.5 for stable blocks.
        
    # Returns
        Calls runImage, displaying the fused image    
    '''
    
    # Read from csv file and use the img IDs as input.
    global imgIdx
    
    with open(csvfile) as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
    
        wantedRow = imgIdx + 1 
        
        rowInfo = csv_reader[wantedRow]
        
        foregroundID = rowInfo[3]
        backgroundID = rowInfo[4]
            
    foreground = Image.open(foregroundID, mode='r')
    background = Image.open(backgroundID, mode='r')
    
    fusedImg = Image.blend(background, foreground, alpha)
    
    # fusedImg.show() # Show composite image
    runImage(fusedImg)
    
    imgIdxLst.append(imgIdx)
    
    imgIdx += 1
    
    background.close()
    foreground.close()
    
def fuseImageSlow(csvfile,alpha=0.5):
    '''
    Same function as fuseImage, but shows the image in a longer time interval (for demos).    
    '''
    
    global imgIdx
    
    with open(csvfile) as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
    
        wantedRow = imgIdx + 1 
        
        rowInfo = csv_reader[wantedRow]
        
        foregroundID = rowInfo[3]
        backgroundID = rowInfo[4]
            
    foreground = Image.open(foregroundID, mode='r')
    background = Image.open(backgroundID, mode='r')
    
    fusedImg = Image.blend(background, foreground, alpha)
    
    runImageSlow(fusedImg)
    
    imgIdxLst.append(imgIdx)
    
    imgIdx += 1
    
    background.close()
    foreground.close()
    
def prepNonFusedImage(csvfile):
    '''
    Returns a single non-fused image based on a pre-generated csv file with indices of images.
    
    # Arguments
        csvfile: string
            String path containing indices of which image to display. Based on global imgIdx counter.
        
    # Returns
        Calls runImage, displaying the image in the pre-selected window (win).
    '''
    
    global imgIdx
    
    with open(csvfile) as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
    
        wantedRow = imgIdx
        
        rowInfo = csv_reader[wantedRow]
        img = rowInfo[1]
        
    imgIdx += 1
        
    runImage(img)
    

def runImage(fusedImg):
    '''
    Runs a single fused image based on output from fuseImage in the defined experimental window (win).
    
    # Arguments
        fusedImg: Image object
            The fused image (from fuseImage function).
    
    # Returns
        No direct output. Shows the image in a predefined number of frames.
    '''
    
    image = visual.ImageStim(win, image = fusedImg, autoLog=True,units='pix',size=(175,175))
    
    outlet.push_sample([imgIdx])
    t = globalClock.getTime()
    timeLst.append(t)
    
    for frameNew in range(0,stimTime):
        if frameNew >= 0:
            image.draw()
        win.flip()
    
def runImageSlow(fusedImg):
    '''
    Same as runImage, but shows the image in 120 frames (instead of the default: 60 frames).
    '''
    
    image = visual.ImageStim(win, image = fusedImg, autoLog=True,units='pix',size=(175,175))
    
    outlet.push_sample([imgIdx])
    t = globalClock.getTime()
    timeLst.append(t)
    
    for frameNew in range(0,120):
        if frameNew >= 0:
            image.draw()
        win.flip()
        
def runBreak(breakLen,message):
    '''
    Runs a break the defined experimental window (win).
    
    # Arguments
        breakLen: int
            Length of break in frames.
    
    # Returns
        No direct output. Displays the break in a predefined number of frames.
    '''
    
    message = str(message)
    
    textBreak = visual.TextStim(win=win, name='textBreak', text=message, font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1, depth=0.0)

    for frameNew in range(0,breakLen): 
        textBreak.draw()
        win.flip()


def runFixProbe(csvfile):
    '''
    Displays fixation cross and probe word text in the defined experimental window (win).
    
    # Arguments
        csvfile: string
            String path containing indices of which images to fuse in column 2 and 3, and name of the attentive category in column 1.
    
    # Returns
        No direct output. Displays fixation and probe text in a predefined number of frames.
    '''
    
    with open(csvfile) as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        
        wantedRow = imgIdx + 1
        
        rowInfo = csv_reader[wantedRow]
        
        attentiveText = str(rowInfo[1])
        
    textGeneral = visual.TextStim(win=win, name='textGeneral', text=attentiveText, font='Arial',
                             units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                             color='white', colorSpace='rgb', opacity=1, depth=0.0)    
    
    for frameN in range(0,probeTime):
        textGeneral.draw()
        win.flip() 
        
    for frameN in range(0,fixTime):
        textFix.draw()
        win.flip() 

    
def saveDataFrame(subjID):
    '''
    Saves the data frame to the subject_path.
    '''
    df_timeLst = pd.DataFrame(timeLst,imgIdxLst)
    df_timeLst.to_csv(subject_path + '\\' + subjID + '\\imageTime_subjID_' + str(subjID) + '_day_' + str(expDay) + '_' + str(log_base) + '.csv') 
    
    
def saveAlphaDataFrame(subjID):
    '''
    Saves the data frame to the subject_path.
    '''
    df_alphaLst = pd.DataFrame(alphaLst)
    df_alphaLst.to_csv(subject_path + '\\' + subjID + '\\alpha_subjID_' + str(subjID) + '.csv') 
    
    
def saveMeanAlphaDataFrame(subjID):
    '''
    Saves the data frame to the subject_path.
    '''
    df_meanAlphaLst = pd.DataFrame(meanAlphaLst)
    df_meanAlphaLst.to_csv(subject_path + '\\' + subjID + '\\MEAN_alpha_subjID_' + str(subjID) + '_'  + 'day_' + str(expDay) + str(log_base) + '.csv') 
    
def readMarkerStream(stream_name ='alphaStream'):
    '''
    Reads stream from an outlet (alphaStream from runClosedLoop.py script).
    The outlet contains alpha values for updating the image stimuli based on recorded and decoded EEG.
    
    # Arguments
    stream_name: string
        Name of stream from runClosedLoop.py script.
    
    # Returns
        inlet: Pylsl object
            Inlet for connecting with the EEG runClosedLoop.py script.
            The inlet contains the alpha values (decoded EEG responses) for updating stimuli.
    
    '''
    index_alpha = []
    alpha_created = []
    streams = resolve_byprop('type', 'Markers', timeout=10)
    for i in range (len(streams)):
        if (streams[i].name() == stream_name):
            index_alpha.append(i)
            alpha_created.append(streams[i].created_at())
    if index_alpha:
        if len(index_alpha) > 1:
            index_alpha = index_alpha[np.argmax(alpha_created)]
        inlet = StreamInlet(streams[index_alpha[0]])
    else:
        print('Warning: No marker inlet available')
    return inlet

def runTest(day='1'):
    '''
    Runs a demo version of the experimental script (either day 1 and or 2) without EEG recordings.
    
    # Arguments
        day: string
            either '1' or '2'. Day 1 illustrate stable blocks and day 2 illustrates neurofeedback blocks.
            
    # Returns
        Runs the experimental demo script.
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
                runBreak(600,'10 second break')
                            
        for jj in list(range(0,runLenHalf)): # The last 4 blocks
 
            if jj == 0: 
               runBreak(180,'Neurofeedback is starting')
           
            if jj % blockLen == 0: # Shows the category text probe
                runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
       
            if jj in [0,1,10]: # First 3 trials are stable, alpha = 0.5
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


def runBehDay(numRuns=2, numBlocks=8, blockLen=50):
    '''
    Runs the experimental script for behavioral days (1 and 3) without EEG recordings.
    
    If expDay is day 1, the participant can be randonmly assigned to an experimental group using randomizeParticipants.py (uncomment code).
    
    # Arguments
        numRuns: int
            Number of runs (one run is a certain number of blocks).
        
        numBlocks: int
            Number of blocks (one run).
        
        blockLen: int
            Number of images to display in each block. 
    
    # Returns
        No direct output. Runs the experimental script for the behavioral paradigm.    
    '''
    
    # if expDay == '1':
    #     assign_subject(subjID) # For assigning participants into neurofeedback and control groups.
    
    time.sleep(25)

    runLen = numBlocks * blockLen 
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
    The script checks whether the subject (based on subjID) is a control or neurofeedback subject (based on a .txt file in the subject folder).
    If control: sham feedback. If neurofeedback: real-time, correct neurofeedback.
        
    # Arguments
        subjID: string
            Subject ID.
            
        numRuns: int
            Number of runs (one run is a certain number of blocks).
        
        numBlocks: int
            Number of blocks (one run).
        
        blockLen: int
            Number of images to display in each block.
        
        day: int
            Experimental day.  
    
    # Returns
        No direct output. Runs the experimental script for the EEG neurofeedback paradigm.
    '''
    
    import copy
    
    time.sleep(30)
    
    runLen = numBlocks * blockLen 
    runLenHalf = int(runLen/2)
    
    # Read whether participant is control or feedback subject from file generated on day 1. 
    control_file = open(subject_path + '\\' + subjID + '\\feedback_subjID' + str(subjID) + '.txt','r')
    control = [x.rstrip("\n") for x in control_file.readlines()]

     ### CONTROL SUBJECTS ###
    if control[0] == '0':
        subj_orig=copy.copy(subjID) 
        
        subjID=control[1] # Use alpha file and createIndices file from matched feedback subject
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

                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')    
                
                if ii == runLen-1:
                    runBreak(1800,'30 second break')
                                    
        # NF runs (consisting of stable and NF)
        for run in list(range(0,numRuns)):
            for ii in list(range(0,runLenHalf)): # The first 4 blocks
                
                if ii == 0: 
                   runBreak(180,'Recording stable blocks')
                
                if ii % blockLen == 0: # Shows the category text probe
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')

                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
                
                if ii == runLenHalf-1: # Break between the stable blocks of NF runs
                    runBreak(1800,'30 second break') # This is when the clf is trained
        
            # Train classifier and find alpha stream - SHAM feedback
            
            for jj in list(range(0,runLenHalf)):
                
                if jj == 0:
                    runBreak(180,'Neurofeedback is starting')
                    
                if jj % blockLen == 0:
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
                    
                if jj in [0,1,2,50,51,52,100,101,102,150,151,152]: # First 3 trials are stable, alpha = 0.5
                    fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv', alpha=0.5)
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
                    
                    fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv', alpha=mean_alpha)
                    
                if jj == runLenHalf-1:
                    
                    if run != numRuns-1:
                        
                        runBreak(1800,'30 second break')  
                        
            if run == 1:
                runBreak(360,'Experiment halfway')
                    
            if run == numRuns-1:  
                runBreak(300,'Experiment finished')
                saveDataFrame(subj_orig) # Saves the timing of image stimuli 
                saveAlphaDataFrame(subj_orig) # Saves alpha values used in the experiment
                saveMeanAlphaDataFrame(subj_orig) 
                
                closeWin()
        
    
    ### NEUROFEEDBACK SUBJECTS ###
    if control[0] == '1':
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

                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')    
                if ii == runLen-1:
                    runBreak(1800,'30 second break')
                                    
        # NF runs (consisting of stable and NF)
        for run in list(range(0,numRuns)):
            for ii in list(range(0,runLenHalf)): # The first 4 blocks
                
                if ii == 0: 
                   runBreak(180,'Recording stable blocks')
                
                if ii % blockLen == 0: # Shows the category text probe
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
                    
                fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
                
                if ii == runLenHalf-1: # Break between the stable blocks of NF runs
                    runBreak(1800,'30 second break') # This is when the clf is trained
        
                    
            # Train classifier and find alpha stream
            inlet_alpha = readMarkerStream(stream_name ='alphaStream')
            
            for jj in list(range(0,runLenHalf)): # The last 4 blocks
    
                if jj == 0: 
                    runBreak(180,'Neurofeedback is starting')
               
                if jj % blockLen == 0: # Shows the category text probe
                    runFixProbe(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv')
           
                if jj in [0,1,2,50,51,52,100,101,102,150,151,152]: # First 3 trials are stable
                    alphamarker,timestamp = inlet_alpha.pull_chunk(timeout=0.005)
                   
                    if len(alphamarker):
                       alphaVal = alphamarker[-1][0]
                    else:
                       alphaVal = 0.5
                       
                    alphaLst.append(alphaVal)
                    fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv', alpha=0.5)
                    
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
                    
                    fuseImage(subject_path + '\\' + subjID + '\\createIndices_' + str(subjID) + '_day_' + str(expDay) + '.csv', alpha=mean_alpha) 
                
                if jj == runLenHalf-1: # Break after a finished run
                    if run != numRuns-1: # Avoid a break after the very last run 
                        runBreak(1800,'30 second break')
                    
            if run == 1:
                runBreak(300,'Experiment halfway')

                
            if run == numRuns-1:  
                runBreak(300,'Experiment finished')
                saveDataFrame(subjID) # Saves the timing of image stimuli 
                saveAlphaDataFrame(subjID) # Saves alpha values used in the experiment
                saveMeanAlphaDataFrame(subjID)
                
                # Uncomment code below if using randomizeParticipants.py
                # group_fname = gen_group_fname(subjID)
                # add_avail_feedback_sub(subjID,group_fname)
                
                closeWin()


