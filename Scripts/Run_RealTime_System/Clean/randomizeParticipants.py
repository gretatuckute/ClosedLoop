# -*- coding: utf-8 -*-
'''
Functions for randomly assigning participants as feedback participants or matched controls.
'''

# Imports
import numpy as np
import pandas as pd
import os
import shutil
from paths import script_path_init, subject_path_init

subject_path = subject_path_init()

def copy_files(from_file, to_file):
    '''
    Copies .csv files containing directories to images to the subject folder and renames the files.
    '''
    shutil.copy(from_file+'_day_1.csv',to_file+'_day_1.csv')
    shutil.copy(from_file+'_day_2.csv',to_file+'_day_2.csv')
    shutil.copy(from_file+'_day_3.csv',to_file+'_day_3.csv')
    shutil.copy(from_file+'_day_4.csv',to_file+'_day_4.csv')
    shutil.copy(from_file+'_day_5.csv',to_file+'_day_5.csv')

def copy_alpha(from_file,to_file):
    shutil.copy(from_file,to_file)
    
def read_re_move_indices(subjects_dir, sub, feedback_from=None):
    '''
    Copies .csv files containing directories to images to the subject folder.
    
    # Arguments
        subjects_dir: string
            Path to main subject directory.
            
        sub: string
            Subject ID
            
        feedback_from: None or string
            None if feedback subject, else subject ID of matched feedback subject.
    '''
    
    indices_dir = subjects_dir+'createIndices\\'
    if feedback_from == None: # If feedback participant.
        
        # Look for available/unused createIndices files (.csv file with image stimuli)
        avail_indices_fname = indices_dir+'avail_indices.txt'
        f_indices = open(avail_indices_fname,'r')
        avail_indices = f_indices.readlines()
        f_indices.close()
        if len(avail_indices) == 0:
            print('No new available indices files!')
        i = np.random.permutation(len(avail_indices))[0]
        indices = avail_indices[i][0:2]
        indices_fname = indices_dir+'createIndices_'+indices
        f_indices = open(avail_indices_fname,'w')
        [np.savetxt(f_indices,[avail[0:2]],fmt='%s') for avail in avail_indices if indices not in avail]
        f_indices.close()
        
    else: # If control participant: get indices file from feedback participant.
        indices_fname=subjects_dir+feedback_from+'\\createIndices_'+feedback_from
    
    indices_to_fname = subjects_dir+sub+'\\'+'createIndices_'+sub#+'_day_1'.csv
    copy_files(indices_fname, indices_to_fname)   
 

def feedback_txt(subjects_dir, sub, feedback, feedback_from=None):
    '''
    Generates a .txt file containing whether subject is feedback (1) or control (0). 
    If control subject, the matched feedback subject is written in row 2.
    
    # Arguments
        subjects_dir: string
            Path to main subject directory
        
        sub: string
            Subject ID
            
        feedback: 0 or 1 
            Whether feedback (1) or control (0) subject.
            
        feedback_from: string
            If control subject, which feedback subject to match with.
    '''
    if feedback_from == None:
        feedback_from = sub
    fname = subjects_dir+sub+'\\feedback_subjID'+sub+'.txt'
    f = open(fname,'w')
    np.savetxt(f, [feedback],fmt='%d')
    np.savetxt(f, [feedback_from],fmt='%s')
    f.close()

    
def assign_remove_avail_feedback_sub(group_fname):
    '''
    Finds a matched feedback subject for a control subject in the relevant experimental group.
    ''' 
    f = open(group_fname, 'r')
    available_subjectIDs = f.readlines()
    f.close()
    n_avail = len(available_subjectIDs)
    i = np.random.permutation(n_avail)[0]
    feedback_from = available_subjectIDs[i][0:2]
    f = open(group_fname,'w')
    [np.savetxt(f,[avail[0:2]], fmt='%s') for avail in available_subjectIDs if feedback_from not in avail]
    f.close()
  
    return feedback_from
    
def add_avail_feedback_sub(add_subjectID, group_fname):
    ''' After the neurofeedback run of a feedback subject, adds the subject ID to list of available 
    feedback subjects for matching of control subjects.
    '''
    file_exist = os.path.isfile(group_fname)
    if file_exist:
        f_group = open(group_fname, 'a')
        np.savetxt(f_group, [add_subjectID], fmt='%s')
        f_group.close()
    else:
        f_group = open(group_fname,'w')
        np.savetxt(f_group,[add_subjectID], fmt='%s')
        f_group.close()
    
def assign_feedback(subjects_dir,sub,group_fname):
    ''' 
    Moves and creates files if a subject is assigned to be a feedback subject.
    '''
    feedback_txt(subjects_dir, sub, 1)
    read_re_move_indices(subjects_dir, sub)
    
def assign_control(subjects_dir,sub,group_fname):
    ''' 
    Moves and creates files if a subject is assigned to be a control subject.
    '''
    feedback_from=assign_remove_avail_feedback_sub(group_fname)
    feedback_txt(subjects_dir,sub,0,feedback_from)
    read_re_move_indices(subjects_dir,sub,feedback_from)

def gen_group_fname(sub):
    '''
    Generates the group name based on an .xlsx file with subject and experimental time slot information.
    '''
    
    subjects_dir = subject_path+'\\'
    data_file = 'C:\\Users\\nicped\\Documents\\google_drive\\booking.xlsx'
    colnames = ['name', 'subID', 'age', 'gender', 'hand', 'day1_date', 'day1_time', 'day2_date', 'day2_time',
                'day3_date', 'day3_time']
    data = pd.read_excel(data_file, names=colnames, converters={'subID': str})
    subjID = data.subID.tolist()
    age = data.age.tolist()
    gender = data.gender.tolist()
    hand = data.hand.tolist()

    ID = subjID.index(sub)
    age_range1 = range(18, 27)
    age_range2 = range(27, 36)
    if age[ID] in age_range1:
        age_range = 0
    elif age[ID] in age_range2:
        age_range = 1
    else:
        print('Age range does not exist')

    subjID[ID]
    filename = 'ageRange' + str(age_range) + '_gender' + gender[ID] + '_hand' + hand[ID] + '.txt'

    data_dir = subjects_dir + 'randomization_files\\'
    group_fname = data_dir + filename
    return group_fname

def assign_subject(sub):
    '''
    Assigns a subject to either control or feedback group.
    '''
    
    subjects_dir = subject_path+'\\'
    data_file = 'C:\\Users\\nicped\\Documents\\google_drive\\booking.xlsx' # File containing information about subjects and experimental time slots.
    colnames = ['name', 'subID', 'age', 'gender', 'hand', 'day1_date', 'day1_time', 'day2_date', 'day2_time',
                'day3_date', 'day3_time']
    data = pd.read_excel(data_file, names=colnames, converters={'subID': str})
    subjID = data.subID.tolist()
    age = data.age.tolist()
    gender = data.gender.tolist()
    hand = data.hand.tolist()

    ID = subjID.index(sub)
    age_range1 = range(18, 27)
    age_range2 = range(27, 36)
    if age[ID] in age_range1:
        age_range = 0
    elif age[ID] in age_range2:
        age_range = 1
    else:
        print('Age range does not exist')

    subjID[ID]
    filename = 'ageRange' + str(age_range) + '_gender' + gender[ID] + '_hand' + hand[ID] + '.txt'

    data_dir = subjects_dir + 'randomization_files\\'
    group_fname = data_dir + filename
    file_exist = os.path.isfile(group_fname)

    if file_exist:
        f = open(group_fname, 'r')
        available_subjectIDs = f.readlines()
        f.close()
        n_avail = len(available_subjectIDs)
        if n_avail:
            #print('Available feedback matches, entering randomization')
            if n_avail > 2:  # CONTROL
                #print('More than two avail, assigning control')
                assign_control(subjects_dir, sub, group_fname)
            else:
                #print('Enter randomization')
                feedback = np.random.permutation(2)[0]
                if feedback == 1:  # FEEDBACK
                    #print('Random feedback')
                    assign_feedback(subjects_dir, sub, group_fname)
                else:  # CONTROL
                    #print('Random control')
                    assign_control(subjects_dir, sub, group_fname)

                    # remove avail_subID
        else:  # FEEDBACK
            #print('No available feedback matches, assigning feedback')
            assign_feedback(subjects_dir, sub, group_fname)


    else:  # FEEDBACK
        print('File does not exist, assigning subject as feedback person')
        assign_feedback(subjects_dir, sub, group_fname)
