# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:42:34 2019

@author: Greta
"""
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['lines.linewidth'] = 3     # line width in points
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rc('text',usetex=True)
# matplotlib.rc('text',usetex=False)
matplotlib.rc('font',family='serif')
plt.rcParams.update({'font.size':12})
matplotlib.rcParams['grid.alpha'] = 1
matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'
matplotlib.rcParams['axes.labelsize'] = 'x-large'


lst=[0,0,0]

plt.style.use('dark_background') 
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['axes.labelweight'] = 5
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
matplotlib.rcParams['mathtext.fontset'] = 'cm'


hfont = {'fontname':'Helvetica'}


plt.subplots()
plt.plot(np.arange(40),np.hstack((lst,clf_output_test[153:190])),linewidth=5,color='white')
plt.ylim(-1,1)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,40)
plt.xticks([0,9,19,29,39],['1','10','20','30','40'])
plt.hlines(0,xmin=-0.5,xmax=41.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

plt.savefig('eig2.png',dpi=280)


for ii in range(0,20):
    print(ii)
    plt.subplots()
    plt.plot(np.arange(ii+3),np.hstack((lst,clf_output_test[103:103+ii])),linewidth=5,color='white')
    plt.ylim(-1,1)
    plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
    plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
    plt.xlim(0,20)
    plt.xticks([0,4,9,14,19],['1','5','10','15','20'])
    plt.hlines(0,xmin=-0.5,xmax=21.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
    plt.xlabel('Trial Number')
    
    plt.savefig('fig_b2'+str(ii)+'.png',dpi=280)
    

for ii in range(0,40):
    plt.subplots()
    plt.plot(np.arange(ii+3),np.hstack((lst,clf_output_test[3:3+ii])),linewidth=5,color='white')
    plt.ylim(-1,1)
    plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
    plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
    plt.xlim(0,40)
    plt.xticks([0,9,19,29,39],['1','10','20','30','40'])
    plt.hlines(0,xmin=-0.5,xmax=41.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
    plt.xlabel('Trial Number')
    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
    
    plt.savefig('fig_b1_'+str(ii)+'.png',dpi=280)
    
    
#%% save first ones
plt.subplots()
plt.plot([0,1,2],[0,0,0],color='white',linewidth=5)
plt.ylim(-1,1)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,40)
plt.xticks([0,9,19,29,39],['1','10','20','30','40'])
plt.hlines(0,xmin=-0.5,xmax=41.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

plt.savefig('fig_2.png',dpi=280)    
    


