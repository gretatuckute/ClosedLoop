# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:29:34 2019

@author: Greta
"""
from PIL import Image

docDir = 'C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Documents\\'


background = Image.open(docDir+'face3627.jpg', mode='r')
foreground = Image.open(docDir+'scene3268.jpg')
    
# FOR SAVING
fusedImage = Image.blend(foreground,background, .5)

fusedImage.show()

fusedImage.save(docDir+'e7.jpg')

background.close()
foreground.close()
fusedImage.close()