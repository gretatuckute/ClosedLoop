# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:42:34 2019

@author: Greta
"""
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os


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


#%%
lst=[0,0,0]

plt.style.use('dark_background') 
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['axes.labelweight'] = 10
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
# matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.fontset'] = 'stix'



hfont = {'fontname':'Helvetica'}

#%% Load clf output file
clf_output_test = np.load('P:\\Research2018_2019\\DTU_closed_loop\\closed_loop_data\\90_demo\\clf_out.npy')

os.chdir('P:\\Research2018_2019\\DTU_closed_loop\\closed_loop\\Demo_Plots\\Attempt_3')

#%% 40 trials

plt.subplots()
plt.plot(np.arange(40),np.hstack((lst,clf_output_test[3:40])),linewidth=5,color='white')
plt.ylim(-1,1)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,40)
plt.xticks([0,9,19,29,39],['1','10','20','30','40'])
plt.hlines(0,xmin=-0.5,xmax=41.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number', weight='bold')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

# plt.savefig('eig2.png',dpi=280)

#%% less (15), indoor
plt.subplots()
plt.plot(np.arange(18),np.hstack((lst,clf_output_test[303:318])),linewidth=5,color='white')
plt.ylim(-1.05,1.05)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,17)
plt.xticks([0,4,9,14],['1','5','10','15'])
plt.hlines(0,xmin=-0.5,xmax=14.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number', weight='bold')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

#%% male, first one 
plt.subplots()
plt.plot(np.arange(18),np.hstack((lst,clf_output_test[3:18])),linewidth=5,color='white')
plt.ylim(-1.05,1.05)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,17)
plt.xticks([0,4,9,14],['1','5','10','15'])
plt.hlines(0,xmin=-0.5,xmax=14.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number', weight='bold')
plt.title('Category: male',size='x-large', weight='bold',loc='left')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

#%% save male
for ii in range(0,16):
    plt.subplots()
    plt.plot(np.arange(ii+3),np.hstack((lst,clf_output_test[3:3+ii])),linewidth=5,color='white')
    plt.ylim(-1.05,1.05)
    plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
    plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
    plt.xlim(0,17)
    plt.xticks([0,4,9,14],['1','5','10','15'])
    plt.hlines(0,xmin=-0.5,xmax=17.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
    plt.xlabel('Trial Number', weight='bold')
    plt.title('Category: Faces',size='x-large', weight='bold',loc='left')
    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

    plt.savefig('fig_male_'+str(ii)+'.png',dpi=280)
    
    
#%% save indoor
for ii in range(0,16):
    plt.subplots()
    plt.plot(np.arange(ii+3),np.hstack((lst,clf_output_test[253:253+ii])),linewidth=5,color='white')
    plt.ylim(-1.05,1.05)
    plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
    plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
    plt.xlim(0,17)
    plt.xticks([0,4,9,14],['1','5','10','15'])
    plt.hlines(0,xmin=-0.5,xmax=17.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
    plt.xlabel('Trial Number', weight='bold')
    plt.title('Category: Scenes',size='x-large', weight='bold',loc='left')
    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

    plt.savefig('fig_scenes_'+str(ii)+'.png',dpi=280)



#%%
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
    
    
#%% save first ones, male
plt.subplots()
plt.plot([0,1,2],[0,0,0],color='white',linewidth=5)
plt.ylim(-1.05,1.05)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,17)
plt.xticks([0,4,9,14],['1','5','10','15'])
plt.hlines(0,xmin=-0.5,xmax=17.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number', weight='bold')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

#%%
plt.savefig('fig_2.png',dpi=280)    
    
# MALE
#%% save first (blank) 
plt.subplots()
plt.plot(0,0,color='white',linewidth=5)
plt.ylim(-1.05,1.05)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,17)
plt.xticks([0,4,9,14],['1','5','10','15'])
plt.hlines(0,xmin=-0.5,xmax=17.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number', weight='bold')
plt.title('Category: Faces',size='x-large', weight='bold',loc='left')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

plt.savefig('fig_blank_cat.png',dpi=280)    


#%% save first ones
plt.subplots()
plt.plot([0,1,2],[0,0,0],color='white',linewidth=5) # First two: 0,1.25, three: 0, 1, 2. First one: 0.5
plt.ylim(-1.05,1.05)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,17)
plt.xticks([0,4,9,14],['1','5','10','15'])
plt.hlines(0,xmin=-0.5,xmax=17.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number', weight='bold')
plt.title('Category: Faces',size='x-large', weight='bold',loc='left')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)


plt.savefig('fig_pre_1_2_3.png',dpi=280)    

# INDOOR
#%% save first (blank) 
plt.subplots()
plt.plot(0,0,color='white',linewidth=5)
plt.ylim(-1.05,1.05)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,17)
plt.xticks([0,4,9,14],['1','5','10','15'])
plt.hlines(0,xmin=-0.5,xmax=17.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number', weight='bold')
plt.title('Category: Scenes',size='x-large', weight='bold',loc='left')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

plt.savefig('fig_blank_scenes.png',dpi=280)    


#%% save first ones
plt.subplots()
plt.plot([0,0.5],[0,0],color='white',linewidth=5) # First two: 0,1.25, three: 0, 1, 2. First one: 0.5
plt.ylim(-1.05,1.05)
plt.yticks([-1,-0.5,0,0.5,1],['-1','-0.5','0','0.5','1'])
plt.ylabel('Real-Time Category Decoding') # Task Relevant Category decoded
plt.xlim(0,17)
plt.xticks([0,4,9,14],['1','5','10','15'])
plt.hlines(0,xmin=-0.5,xmax=17.5,linestyles='dashed',label='Chance',zorder=4,linewidth=2,color='white')
plt.xlabel('Trial Number', weight='bold')
plt.title('Category: Scenes',size='x-large', weight='bold',loc='left')
plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)


plt.savefig('fig_pre_scenes_1.png',dpi=280)    

