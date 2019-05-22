# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:29:34 2019

@author: Greta
"""
from PIL import Image

background = Image.open('face2181.jpg', mode='r')
foreground = Image.open('scene624.jpg')
    
# FOR SAVING
fusedImage = Image.blend(background,foreground, .20)

fusedImage.show()

fusedImage.save('aswop20.jpg')

background.close()
foreground.close()
fusedImage.close()