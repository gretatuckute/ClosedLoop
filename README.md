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

The system requires images for the visual paradigm (see **Stimuli**). The recorded EEG data will be saved in the \ClosedLoop\subjectsData folder (see “”)

## Stimuli
The current paradigm uses composite images (an overlay of two images from different categories). The stimuli are provided by a .csv file, as illustrated below:

![](createIndices_example.png)

## Experimental paradigm
Participants had to respond to and, by extension, focus their attention towards subcategories of faces: female
and male, and scenes: indoor and outdoor. Stimuli were composite images of the two categories (equal image
mixture ratio during training blocks). During feedback blocks, the decoded task-relevant EEG representation
was used to continuously update the image mixture of the stimuli in a closed-loop manner. If participants
attended well (high levels of task-relevant information in their brain) the task-relevant image became easier to
see, and vice versa. Thus, the feedback worked as an amplifier of participants attentional state, with the goal
to make participants’ aware of attention fluctuations and hence improve sustained attention abilities


## Dependencies/acknowledgements:
- Psychopy (https://www.psychopy.org/)
- MNE (https://mne-tools.github.io/stable/index.html) 
- Lab streaming layer (https://github.com/sccn/labstreaminglayer)

Additionally a number of images must be available. For the face images we used the FERET database (https://www.nist.gov/itl/iad/image-group/color-feret-database) and for the scene images we used the SUN database (https://groups.csail.mit.edu/vision/SUN/).
