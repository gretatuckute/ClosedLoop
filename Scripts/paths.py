# -*- coding: utf-8 -*-
'''
Initialization of paths for system scripts, subjects directory, and data directory.
'''

import os

def script_path_init(): # Base directory for ClosedLoop GitHub
    script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return script_path

# def script_path_init():
#     script_path = base_dir_init() + '\Scripts'
#     return script_path

def data_path_init(): # Data (images) storage directory
    data_path = base_dir_init() + '\imageStimuli'
    return data_path

def subject_path_init(): # Subjects directory, for storing EEG data 
    subject_path = base_dir_init() + '\subjectsData'
    return subject_path


if __name__ == '__main__':
    base_dir = script_path_init()
    print('====== Current directory ======')
    print(base_dir)
   