# Closed-Loop EEG Neurofeedback System

This GitHub contains code for running a closed-loop, real-time EEG neurofeedback system.

The system is initially designed for a visual attention paradigm ([deBettencourt et al., 2015](https://www.nature.com/articles/nn.3940)), where the subjective attentional state of the participant is decoded in real-time and used to update the subsequent visual stimuli to modulate the attentional state using feedback.

The system can easily be adapted for various visual paradigms (see **Stimuli**).

## Running the system
1)	Start recording using the EEG equipment (we used [Enobio 32](https://www.neuroelectrics.com/products/enobio/)).
2)	Start *runClosedLoop.py* (the script will wait for initialization of step 3). 
3)	In a different terminal/console, start *runSystem.py* (initializes the experimental script). 

Visualization of the major components of the system:

![](systemComponents.png)

The system requires images for the visual paradigm (see **Stimuli**). The recorded EEG data will be saved in the \ClosedLoop\subjectsData folder (see “Saving data”).

## Stimuli
The current paradigm uses composite images (an overlay of two images from different categories). The stimuli are provided by a .csv file, as illustrated below:

![](createIndices_example.PNG)

The number of rows have to correspond to the number of experimental trials.

The .csv file can be generated using the script *prepareImageStimuli.py*. This requires images in the folders in \ClosedLoop\imageStimuli\. For the face images we used the [FERET database](https://www.nist.gov/itl/iad/image-group/color-feret-database) and for the scene images we used the [SUN database](https://groups.csail.mit.edu/vision/SUN/).

A subject ID (example: '01') has to be entered in line 22 of *prepareImageStimuli.py*. The .csv file will thus be saved in  \ClosedLoop\subjectsData\subjectID\. 

The script assumes four different image categories: male, female, indoor, outdoor (can be changed by changing folder names and category combinations, catComb, in *prepareImageStimuli.py*, lines 28-38).

Non-composite images can also be generated using the function *createNonFusedIndices* in *experimentFunctions.py* and run using the function *prepNonFusedImage* instead of fuseImage in either runBehDay for the behavioral paradigm, and runNFday for the neurofeedback paradigm (has to be manually changed).

## Experimental paradigm
Participants had to respond to and, by extension, focus their attention towards subcategories of faces: female
and male, and scenes: indoor and outdoor. Stimuli were composite images of the two categories (equal image
mixture ratio, alpha, during training blocks). During feedback blocks, the decoded task-relevant EEG representation
was used to continuously update the image mixture of the stimuli in a closed-loop manner. If participants
attended well (high levels of task-relevant information in their brain) the task-relevant image became easier to
see, and vice versa (see [deBettencourt et al., 2015](https://www.nature.com/articles/nn.3940)).

The experiment contains behavioral days (visual stimuli presentation without EEG recording) and a neurofeedback day (EEG recording and neurofeedback during visual stimuli presentation). 

The experimental structure for the neurofeedback day is as follows:

- One block consists of 50 images (can be changed using the argument blockLen (block length, default 50) in the function runNFday or runBehDay). Each image is displayed for 1 second (60 frames, assuming a frame rate of 60 Hz). 
- One run consists of 8 blocks (can be changed using the argument numBlocks (number of blocks, default 8) in the function runNFday or runBehDay).
- The experiment consists of 6 runs (can be changed using the argument numRuns (number of runs, default 6) in the function runNFday or runBehDay). 
- For behavioral days, the number of runs is 2 instead of 6 runs.

For the neurofeedback paradigm, the first 600 trials (12 blocks) are used for recording EEG data. A classifier is trained based on these blocks, and used for providing feedback in the subsequent 200 trials (4 blocks). This is followed by runs consisting of 4 blocks of recording EEG (‘stable’ blocks) and 4 blocks of providing feedback (‘feedback’ blocks). 

## Saving data
The data will be stored and saved in the subject's folder \ClosedLoop\subjectsData\subjectID\. The .csv file containing image stimuli also has to be located in this folder (default in *prepareImageStimuli.py*, see **Stimuli**).

*runSystem.py* will save the following files after a completed neurofeedback session:
-	subject_01_EEG_TIMESTAMP.csv: All of the recorded, raw EEG data. 
-	subject_01_marker_TIMESTAMP.csv: All the markers (time points of stimuli/EEG epoch onset) for experimental trials. 
-	stream_logfile_subject_01_TIMETAMP.log: A log of the EEG streaming (created by runClosedLoop.py and streamFunctions.py).
-	imageTime_subjID_01_day_2_TIMESTAMP.csv: All time points of stimuli onsets. Same clock as in the two files below.
-	PsychoPyLog_subjID_01_day_2_TIMESTAMP.csv: Log of all changes in the experimental window (using [PsychoPy logging](http://www.psychopy.org/coder/codeLogging.html)).
-	Keypress_subjID_01_day_2_TIMESTAMP.csv: Time points for all recorded keypresses during the experiment.
-	alpha_subjID_01.csv: The computed alpha values (image mixture interpolation factor, see **Experimental paradigm**) used to update image stimuli during ‘feedback’ blocks.
-	MEAN_alpha_subjID_01_day_2_TIMESTAMP.csv: The mean alpha values used to update image stimuli during ‘feedback’ blocks.


# Description of scripts

### runClosedLoop.py
*runClosedLoop.py* must be started before *runSystem.py*, since it waits for a marker (trigger point for stimuli onset) from the experimental script which is called in *runSystem.py*.
*runClosedLoop.py* will stream the EEG data. The EEG sampling rate, number of channels and epoch length can be changed in this script. Based on the experimental structure, the script will change the system states among:

1) ‘stable’ (recording of EEG data for training of the decoding classifier)
2) ‘train’ (training of the decoding classifier)
3) ‘feedback’ (preprocessing and classification of EEG data for neurofeedback)

### realtimeFunctions.py
Functions for working with real-time EEG data in Python:
Standardizing, scaling, artefact correction (SSP projection), preprocessing, classification.
The functions are called in *runClosedLoop.py*.

### streamFunctions.py
Functions for finding the EEG stream and the experimental stream containing markers (trigger points for stimuli onset, from experimental script: *experimentFunctions.py*), saving data, writing log files and changing system states for the neurofeedback system.
The functions are called in *runClosedLoop.py*.

### runSystem.py
*runSystem.py* starts the experimental script. As explained in **Experimental paradigm**, the paradigm consists of behavioral days (day 1 and 3), and a neurofeedback day (day 2). 

For the behavioral experiment, the function *runBehDay* from *experimentFunctions.py* is used.  
For the neurofeedback experiment, the function *runNFday* from *experimentFunctions.py* is used.

Manually enter the experimental day and subject ID in *experimentFunctions.py* (line 32-33) and in *runClosedLoop.py* (line 23). The information in these scripts must match.

A simple .txt file named “feedback_subjID_01.txt” has to be located in the subject’s folder containing 1 in the first row and their own subject ID in the second line (example provided in \ClosedLoop\subjectsData\01\). 
This feedback .txt file provides an opportunity to make participants function as controls, and hence receive yoked, sham neurofeedback (feedback based on another participant’s brain response). In this case, the .txt file has to contain 0 in the first row, and the subject ID of the matched neurofeedback participant in the second row.

### experimentFunctions.py
Functions for running the PsychoPy experimental script and generating stimuli (composite images) for the experiment.
The functions are called in *runSystem.py*.

### paths.py
Initialization of paths for scripts, subjects directory, and image stimuli directory.

## Additional scripts (not necessary for running the system)

1)	*randomizeParticipants.py*: Functions for automated randomization of participants into 'feedback' participants or matched controls. Generates the “feedback_subjID_X.txt file in the subject's folder \ClosedLoop\subjectData\subjectID\, denoting whether a participant
is feedback (first row containing 1) or control (first row containing 0).

2) Offline analysis: 

## Dependencies/acknowledgements:
- [PsychoPy](https://www.psychopy.org/)
- [MNE](https://mne-tools.github.io/stable/index.html) 
- [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer)
- [NumPy](https://www.numpy.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)

*Cognitive Systems, Department of Applied Mathematics and Computer Science, Technical University of Denmark, 2018-19* 

In collaboration with [Sofie Therese Hansen](https://github.com/STherese) and Professor Lars Kai Hansen.
