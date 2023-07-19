# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:27:40 2023

@author: gagan
"""

from identification import get_exp_data, compare_ARX_params, get_best_model
from identification import compare_switch, compare_bilin, regret
import os
from tqdm import tqdm
import pickle as pkl
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_PATH = './results_new'

def saver(vals, path, file_name):
    with open(os.path.join(path,file_name),'wb') as f:
        pkl.dump(vals,f,pkl.HIGHEST_PROTOCOL)
        
def runner(sub,sess,ldef=250,mdef=100, output_path=OUTPUT_PATH):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    paths = []
    for f in ['arx','switch','bilinear']:
        paths.append(os.path.join(output_path,f))
        if not os.path.exists(paths[-1]):
            os.mkdir(paths[-1])
    datadict, anode, n_channels = get_exp_data(sub,sess)
    
    for c in tqdm(range(n_channels)):
        ch = c if c!=anode else 'anode'
        file_name = f'{sub}_PS2_{sess}_{ch}.pkl'
        vals = compare_ARX_params(c,sub,datadict)
        saver(vals,paths[0],file_name)
        model = get_best_model(vals)        
        if model==(0,0):
            model = ldef,mdef
        lags, mlags = model
        if mlags==0:
            mlags = mdef
            
        vals = compare_switch(c,sub,datadict, lags,mlags,l=0.10,seed=10)
        saver(vals,paths[1],file_name)
        vals = compare_bilin(c,sub,datadict, lags,mlags,l=0.10,seed=10)
        saver(vals,paths[2],file_name)
        
        
subs_sessions = [('R1142N', '0'),('R1084T', '0'),('R1094T', '0'),('R1077T', '0'),
                 ('R1108J', '0'),('R1153T', '3'),('R1134T', '1'),('R1101T', '3'),
                 ('R1136N', '0'),('R1068J', '0')]


if __name__=="__main__":
    for ss in subs_sessions:
        runner(ss[0],ss[1])



    

        
            
        