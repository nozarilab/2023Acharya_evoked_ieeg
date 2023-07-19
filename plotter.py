# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:56:57 2023

@author: gagan
"""
from identification import regret
import os
from itertools import product
import pickle as pkl
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_PATH = './results_new'


keys = [(0,0),(1, 0),(1, 100),(1, 200),(1, 300),(50, 0),(50, 100),(50, 200),(50, 300),
        (100, 0),(100, 100),(100, 200),(100, 300),(150, 0),(150, 100),(150, 200),(150, 300),
        (200, 0),(200, 100),(200, 200),(200, 300),(250, 0),(250, 100),(250, 200),(250, 300),
        (300, 0),(300, 100),(300, 200),(300, 300),(400, 0),(400, 100),(400, 200),(400, 300),
        (500, 0),(500, 100),(500, 200),(500, 300)]


def plot_fig2():
    files = glob(os.path.join(OUTPUT_PATH,'arx','*.pkl'))
    models_reg = np.zeros((len(files),len(keys)))
    for i,filename in enumerate(files):
        with open(filename,'rb') as f:
            m = pkl.load(f)
        for j,k in enumerate(keys):
            try:
                models_reg[i,j] = int(regret(m,k,method='mse')==0)
            except:
                models_reg[i,j] = int(regret(m,'zero',method='mse')==0)
            
    models_mse = np.zeros((len(files),len(keys)))
    for j,filename in enumerate(files):
        with open(filename,'rb') as f:
            m = pkl.load(f)
        for i,k in enumerate(keys):
            try:
                idx = m[1][0].index(k)
            except:
                idx = m[1][0].index('zero')
            models_mse[j,i] = m[1][1][idx]/m[1][-1]
            
    fig, ax1 = plt.subplots(figsize=(20, 8),dpi=500)
    ax2 = ax1.twinx()
    
    use_keys = [(0,0)]
    idxs = [0]
    lags = [0]
    for j,k in enumerate(keys):
        if len(k)==2:
            if k[1]==100:
                use_keys.append(k)
                idxs.append(j)
                lags.append(k[0])
    ax2.bar([str(k) for k in use_keys],models_reg.sum(axis=0)[idxs]*100/len(files),
            color=(0,0.4,0.8,0.7))
    
    
    ax1.set_xlabel('Autoregressive lags',fontsize=30)
    ax2.plot(range(-1,len(lags)+1), [90]*(len(lags)+2),'--',color='red',lw=4);
    ax1.set_xticklabels(lags, fontsize=20)
    ax2.set_ylabel("Model win %", color=(0,0.4,0.8), fontsize=30)
    ax1.set_ylabel("Normalized MSE", color=(0,0.2,0.0), fontsize=30)
    ax2.set_yticks([0,50,90,100],[0,50,90,100], fontsize=20)
    ax1.plot([str(k) for k in use_keys],models_mse.mean(axis=0)[idxs],
             'o-',color=(0,0.2,0),lw=4)
    ax1.set_yticks([0.5,1.0],[0.5,1.0], fontsize=20)
    ax1.set_xlim([-0.5,9.5])
    
    plt.show()
    
    
    bmodel_mse_count = np.array([0]*len(keys))
    bmodel_wilcox_count = np.array([0]*len(keys))
    idxs = {keys[i]:i for i in range(len(keys))}
    idxs['zero'] = 0
    for i,filename in enumerate(files):
        with open(filename,'rb') as f:
            m = pkl.load(f)
        key = m[1][0][0]
        bmodel_mse_count[idxs[key]]+=1
        key = m[2][0][0]
        bmodel_wilcox_count[idxs[key]]+=1
    bmodel_mse_count = 100*bmodel_mse_count/bmodel_mse_count.sum()
    bmodel_wilcox_count = 100*bmodel_wilcox_count/bmodel_wilcox_count.sum()
    
    r = np.arange(len(keys))
    width = 0.25
    fig = plt.figure(dpi=500,figsize=(10,3))
    plt.bar(r, bmodel_wilcox_count, color = (0,0.4,0.8),
            width = width, edgecolor = 'black',
            label='Based on median\n(signed-rank test)')
    plt.bar(r + width, bmodel_mse_count, color = 'r',
            width = width, edgecolor = 'black',
            label='Based on MSE')
    
    plt.xlabel("Autoregressive and input lags (L,M)",fontsize=20)
    plt.ylabel("Win rate (%)",fontsize=20)
    #plt.title("Number of people voted in each year")
    
    # plt.grid(linestyle='--')
    plt.xticks(r + width/2,keys,rotation=90,fontsize=15)
    plt.yticks([0,30],[0,30],fontsize=15)
    plt.legend(fontsize=15)
    
    plt.show()



def plot_fig5():
    files = glob(os.path.join(OUTPUT_PATH,'switch','*.pkl'))
    
    keys = [(i,j,k) for i,j,k in product([0,1],[0,1],[0,1])]
    
    idxs = {keys[i]:i for i in range(len(keys))}
    bmodel_mse_count = np.array([0]*len(keys))
    bmodel_wilcox_count = np.array([0]*len(keys))
    
    bmodel_pat_mse = {}
    bmodel_pat_wilcox = {}
    
    mse_list = []
    var_list = []
    for filename in files:
        sub = os.path.split(filename)[-1].split('_')[0]
        if sub not in bmodel_pat_wilcox:
            bmodel_pat_wilcox[sub] = np.array([0]*len(keys))
            bmodel_pat_mse[sub] = np.array([0]*len(keys))
            
        with open(filename,'rb') as f:
            m = pkl.load(f)
        
        kk = 0
        for k in m[1][0]:
            if len(k)==4:
                key = (int(k[1]>0),k[2],k[3])
                mse_list.append(m[1][1][kk])
                var_list.append(m[1][-1])
                kk += 1 
                bmodel_mse_count[idxs[key]]+=1
                bmodel_pat_mse[sub][idxs[key]]+=1
                break
                
        for k in m[2][0]:
            if len(k)==4:
                key = (int(k[1]>0),k[2],k[3])
                bmodel_wilcox_count[idxs[key]]+=1
                bmodel_pat_wilcox[sub][idxs[key]]+=1
                break
        
    bmodel_mse_count = 100*bmodel_mse_count/bmodel_mse_count.sum()
    bmodel_wilcox_count = 100*bmodel_wilcox_count/bmodel_wilcox_count.sum()
    
    for sub in bmodel_pat_wilcox:
        bmodel_pat_wilcox[sub] = 100*bmodel_pat_wilcox[sub]/bmodel_pat_wilcox[sub].sum()
        bmodel_pat_mse[sub] = 100*bmodel_pat_mse[sub]/bmodel_pat_mse[sub].sum()
    
    
    labels = {(0,0,0):'AR',(1,0,0):'ARX',(0,1,0):'Switched AR',(1,1,0):'Switched ARX',
              (0,0,1):'VAR',(1,0,1):'VARX',(0,1,1):'Switched VAR',(1,1,1):'Switched VARX'}
    
    r = np.arange(8)
    width = 0.25
    fig = plt.figure(dpi=500,figsize=(5,5))
    plt.bar(r, bmodel_wilcox_count, color = (0,0.4,0.8),
            width = width, edgecolor = 'black',
            label='Based on median\n(signed-rank test)')
    plt.bar(r + width, bmodel_mse_count, color = 'r',
            width = width, edgecolor = 'black',
            label='Based on MSE')
      
    plt.xlabel("Model",fontsize=20)
    plt.ylabel("Win rate (%)",fontsize=20)
    #plt.title("Number of people voted in each year")
      
    # plt.grid(linestyle='--')
    plt.xticks(r + width/2,[labels[k] for k in keys],rotation=90,fontsize=15)
    plt.yticks([0,50],[0,50],fontsize=15)
    plt.legend(fontsize=15)
      
    plt.show()
    
    subs = list(bmodel_pat_wilcox.keys())
    fig = plt.figure(dpi=500,figsize=(3,5))
    for i in range(5):
        for j in range(2):
            sub = subs[2*i+j]
            ax = plt.subplot(5,2,2*i+j+1)
            ax.bar(r, bmodel_pat_wilcox[sub], color = (0,0.4,0.8),
                    width = width,
                    label='Based on median\n(signed-rank test)')
            ax.bar(r+width, bmodel_pat_mse[sub], color = 'r',
                    width = width,
                    label='Based on MSE')
            ax.set_title(f'Sub: {sub}')
            ax.set_ylim([0,100])
            ax.set_xticks([])
            ax.set_yticks([])
            
            if i==4:
                ax.set_xticks(r + width/2,[labels[k] for k in keys],rotation=90,fontsize=10,ha='right')
    
    fig.tight_layout()
    plt.show()