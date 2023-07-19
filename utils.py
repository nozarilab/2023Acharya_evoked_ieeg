# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:46:10 2022

@author: gagan
"""

import numpy as np
from scipy.stats import wilcoxon as wx
from scipy.stats import ttest_rel as tr

import matplotlib.pyplot as plt

def get_pval(output,keys=None,verbose=1):

    nexps = len(output)
    if 'var' in output:
        nexps = nexps-1
        
    if keys is None:
        keys = list(output.keys())
        
    if 'var' in keys:
        keys.remove('var')
        
    p_vals = np.zeros((nexps,nexps))+0.5
           
    for i in range(nexps):
        ik = keys[i]
        oik = output[ik].reshape((-1,))
        for j in range(nexps):
            jk = keys[j]
            ojk = output[jk].reshape((-1,))
            if i==j:
                continue
            if np.mean((oik-ojk)**2)!=0.0:
                [_,p]=wx(oik,ojk,alternative='less')
                p_vals[i,j] = p           
    return p_vals, keys

def get_pval_tr(output,keys=None):

    nexps = len(output)
    if 'var' in output:
        nexps = nexps-1
        
    if keys is None:
        keys = list(output.keys())
        
    if 'var' in keys:
        keys.remove('var')
        
    p_vals = np.zeros((nexps,nexps))+0.5
           
    for i in range(nexps):
        ik = keys[i]
        oik = output[ik].reshape((-1,))
        for j in range(nexps):
            jk = keys[j]
            ojk = output[jk].reshape((-1,))
            if i==j:
                continue
            if np.mean((oik-ojk)**2)!=0.0:
                [_,p]=tr(oik,ojk,alternative='less')
                p_vals[i,j] = p           
    return p_vals, keys


def pval_plot(output,freq=None,rot=90):
    p_vals, keys = get_pval(output,freq)
    plt.figure(dpi=500)
    plt.imshow(p_vals,vmax=1.0,vmin=0.0,cmap='gray')
    plt.xticks(range(len(keys)),keys,rotation = rot)
    plt.yticks(range(len(keys)),keys,rotation = 90-rot)
    plt.show()