# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:33:41 2023

@author: gagan
"""

import numpy as np
from data_loader import get_data, zscore, create_dataset_windows, get_tt
import cmlreaders as cml
from glob import glob
from config import ROOTDIR
import os
from tqdm import tqdm
from utils import get_pval, get_pval_tr
import pickle as pkl
from multiprocessing import Process


def sweep_ARX(c,datadict,l=0.10,seed=10):
    lags = 500
    mlags = 300
    output = {}
    tx,ty,U,U_amp,chunks= create_dataset_windows(datadict,c,lags=lags,
                                                 mlags=mlags,allow_u=True)
    
    tridxs = get_tt(U,U_amp,chunks, seed=seed)
    _,_,train, test = tridxs
    test_y = ty[test].reshape((-1,1))
    res_err =  test_y
    se = res_err**2
    output[(0,0)] = se
    output['var'] = np.std(ty[test])**2
    
    
    train_x = tx[train,:]  
    train_y = ty[train].reshape((-1,1))
    test_y = ty[test].reshape((-1,1))
    test_x = tx[test,:]
    
    n_sam = train_x.shape[0]
    G = train_x.T@train_x/n_sam+l*np.diag(np.ones(train_x.shape[-1],))
    g = train_x.T@train_y/n_sam
    
    for ll in [1,50,100,150,200,250,300,400,500]:
        for mm in [0,100,200,300]:
            label = (ll,mm)
            #print(label)
            idxs = np.concatenate([np.arange(0,ll),np.arange(lags,lags+mm)])
            
            G_i = G[idxs][:,idxs]
            g_i = g[idxs,:]
            Ginv = np.linalg.inv(G_i)
            
            theta = Ginv@g_i
            
            res_err =  test_x[:,idxs]@theta-test_y
            se = res_err**2
            output[label] = se
                
    return output


def regret(m,k,method='mse'):
    idx = 1
    if method=='wilcox':
        idx = 2
    mse_list = m[idx][1]
    model_list = m[idx][0]
    sem_list = m[idx][2]
    f1 = np.array(mse_list)
    j = model_list.index(k)
    ul = mse_list[j]+1.96*sem_list[j]
    ll = mse_list[j]-1.96*sem_list[j]
    n1 = np.sum((f1-ll)[:j]<0)
    n2 = np.sum((ul-f1)[:j]<0)
    return int(n1+n2)

def get_best_model(m,method='mse'):
    idx = 1
    if method=='wilcox':
        idx = 2
    min_model = m[idx][0][0]
    if min_model==(0,0):
        min_model = m[idx][0][1]
        
    for i in m[idx][0]:
        if i==(0,0):
            continue
        if regret(m,i,method)==0:
            if min_model[0]>i[0]:
                min_model = i
            elif min_model[0]==i[0] and min_model[1]>i[1]:
                min_model = i
    
    if regret(m, min_model, method)>0:
        return (0,0)
    
    return min_model

def evaluate_output(output, exp, sess, c, sub):
    p_vals, keys = get_pval_tr(output)
    
    pv = p_vals.sum(axis=0)
    r = (len(pv)-1-np.argsort(pv[::-1])[-len(pv):])[::-1]
    k = [keys[i] for i in r]
    sN = np.sqrt(len(output[k[0]]))
    
    label = (sub, sess, exp, c)
    vals = [label,[],[]]
    vals[1] = [k, [np.mean(output[ks]) for ks in k], 
               [np.std(output[ks])/sN for ks in k],output['var']]

        
    p_vals, keys = get_pval(output)
    pv = p_vals.sum(axis=0)
    r = (len(pv)-1-np.argsort(pv[::-1])[-len(pv):])[::-1]
    k = [keys[i] for i in r]
    vals[2] = [k, [np.mean(output[ks]) for ks in k], 
               [np.std(output[ks])/sN for ks in k], output['var']]
    
    return vals
    
def compare_ARX_params(c,sub,datadict,l=0.10,seed=10):    
    output = sweep_ARX(c,datadict,l,seed)
    exp,sess = list(datadict.keys())[0]
    vals = evaluate_output(output, exp, sess, c, sub)
    return vals

def sweep_switch_models(c,datadict, lags,mlags,l=0.10,seed=10):
    output = {}
    
    tx,ty,U,U_amp,chunks = create_dataset_windows(datadict,c,lags=lags,mlags=mlags,
                                                  allow_u=True,ilags=1,plags=1,
                                                  amp_inter_u=False, 
                                                  anode_only=False,amp_u=True, 
                                                  do_embed=False, pad_dur=500)
    
    dim = tx.shape[-1]
    tridxs = get_tt(U,U_amp,chunks, seed=seed)
    _,_,train, test = tridxs
    test_y = ty[test].reshape((-1,1))
    
    output['var'] = np.std(ty[test])**2
    
    iw = [0]*4+[1]*4
    mm = [0]*2 + [mlags]*2 + [0]*2 + [mlags]*2
    il = [0,1]*4
    
    train_x = tx[train,:]  
    train_y = ty[train].reshape((-1,1))

    test_x = tx[test,:]
    
    n_sam = train_x.shape[0]
    G = train_x.T@train_x/n_sam+l*np.diag(np.ones(train_x.shape[-1],))
    g = train_x.T@train_y/n_sam
    
    for i in range(8):
        
        label = (lags,mm[i],il[i],iw[i])
        #print(label)
        idxs = np.concatenate([np.arange(0,lags),np.arange(lags,(1+il[i])*lags)])
        idxs = np.concatenate([idxs, np.arange(2*lags,2*lags+mm[i])])
        
        if iw[i]==1:
            idxs = np.concatenate([idxs, np.arange(2*lags+mlags,dim)])

        G_i = G[idxs][:,idxs]
        g_i = g[idxs,:]
        Ginv = np.linalg.inv(G_i)
        
        theta = Ginv@g_i
        
        res_err =  test_x[:,idxs]@theta-test_y
        se = res_err**2
        output[label] = se
        
    return output

def compare_switch(c,sub,datadict, lags,mlags,l=0.10,seed=10):    
    output = sweep_switch_models(c,datadict, lags,mlags,l,seed)
    exp,sess = list(datadict.keys())[0]
    vals = evaluate_output(output, exp, sess, c, sub)
    return vals


def sweep_bilin_models(c,datadict, lags,mlags,l=0.10,seed=10):
    output = {}
    
    models = {'Switched':[False, True],'AWSL':[True,True],'Full':[True,False]}
    for i in models:
        aiu = models[i][0]
        ub = models[i][1]
        tx,ty,U,U_amp,chunks = create_dataset_windows(datadict,c,lags=lags,mlags=mlags,
                                                      allow_u=True,ilags=1,plags=1,
                                                      amp_inter_u=aiu, 
                                                      use_baseline=ub,
                                                      pad_dur=500)
    
    
        tridxs = get_tt(U,U_amp,chunks, seed=seed)
        _,_,train, test = tridxs
        test_y = ty[test].reshape((-1,1))
    
        train_x = tx[train,:]  
        train_y = ty[train].reshape((-1,1))
    
        test_x = tx[test,:]
    
        n_sam = train_x.shape[0]
        G = train_x.T@train_x/n_sam+l*np.diag(np.ones(train_x.shape[-1],))
        g = train_x.T@train_y/n_sam
        
        Ginv = np.linalg.inv(G)
        
        theta = Ginv@g
        
        res_err =  test_x@theta-test_y
        se = res_err**2
        output[i] = se
    output['var'] = np.std(ty[test])**2
            
    return output

def compare_bilin(c,sub,datadict, lags,mlags,l=0.10,seed=10):    
    output = sweep_bilin_models(c,datadict, lags,mlags,l,seed)
    exp,sess = list(datadict.keys())[0]
    vals = evaluate_output(output, exp, sess, c, sub)
    return vals

def get_exp_data(sub, sess):
    exp_interested = 'PS2'
    exp_sess = (exp_interested,str(sess))
    
    datadict = get_data(sub, exp_sess=[exp_sess], desiredrate=1000)
    datadict = zscore(datadict)
    anode = datadict[exp_sess][-1]
    
    n_channels = datadict[exp_sess][0].shape[0]
    return datadict, anode, n_channels


    