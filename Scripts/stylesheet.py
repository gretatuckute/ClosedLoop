# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:12:16 2019

@author: Greta
"""

#%% Plot styles
plt.style.use('seaborn-notebook')

matplotlib.rc('font', **font)

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['grid.alpha'] = 0.5
matplotlib.rcParams['figure.titlesize'] = 20 # Does not work? Change in the specific title
matplotlib.rcParams['axes.labelsize'] = 'xx-small'
matplotlib.rcParams['figure.autolayout'] = True
# matplotlib.rcParams['xtick.labelsize'] = 'xx-small'
# matplotlib.rcParams['ytick.labelsize'] = 'xx-small'
# matplotlib.rcParams['font.size'] = 22