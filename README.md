# Closed-Loop EEG Neurofeedback System

This GitHub contains code for running a closed-loop, real-time EEG neurofeedback system.

The system is initially designed for a visual attention paradigm (deBettencourt), where the subjective attentional state of the participant is decoded in real-time and used to update the subsequent visual stimuli to modulate the attentional state using feedback.

The system can easily be adapted for various visual paradigms (see **Stimuli**).

## Running the system
1)	Start recording using the EEG equipment (we used [Enobio 32](https://www.neuroelectrics.com/products/enobio/)).
2)	Start *runClosedLoop.py* (the script will wait for initialization of step 3). 
3)	In a different terminal/console, start *runSystem.py* (initializes the experimental script). 

Visualization of the major components of the system:

![](systemComponents.pdf)



## Dependencies/acknowledgements:
- Psychopy (https://www.psychopy.org/)
- MNE (https://mne-tools.github.io/stable/index.html) 
- Lab streaming layer (https://github.com/sccn/labstreaminglayer)

Additionally a number of images must be available. For the face images we used the FERET database (https://www.nist.gov/itl/iad/image-group/color-feret-database) and for the scene images we used the SUN database (https://groups.csail.mit.edu/vision/SUN/).
