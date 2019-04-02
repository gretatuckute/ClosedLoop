# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:05:27 2019

@author: Greta
"""

test=np.array([2,3,4,1,2,2,4])

test2=np.array([2,3,0])

np.nanmean(test2)

test[test == 2] = np.nan
