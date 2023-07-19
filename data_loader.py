# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:11:52 2022

@author: gagan
"""

from config import ROOTDIR
import cmlreaders as cml
from glob import glob
import os
from scipy import signal
import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.model_selection import KFold

def get_exp_sess(sub):
    '''
    Function to output all the experiments and sessions corresponding to the 
    given subject

    Parameters
    ----------
    sub : string
        Subject name.

    Returns
    -------
    exp_sess : list
        List of all (experiment, session) pairs e.g. [['PS1','0'],['PS2','0'],...]. 
    '''
    exp_sess = []
    for exp in get_exps(sub):
        for sess in get_sessions(sub,exp):
            exp_sess.append([exp, sess])
    return exp_sess

def get_exps(sub):
    '''
    Function that outputs the list of experiments concerning the given subject
    Parameters
    ----------
    sub : string
        Subject name.

    Returns
    -------
    exps : list
        List of experiments e.g. ['PS1','PS2'].

    '''
    path = os.path.join(ROOTDIR,'protocols','r1','subjects',sub,'experiments','*')
    exps = [os.path.split(p)[-1] for p in glob(path)]
    return exps

def get_sessions(sub,exp):
    '''
    Function that outputs the list of sessions concerning the input experiment 
    on a given subject

    Parameters
    ----------
    sub : string
        Subject name.
    exp : string
        Experiment type 'PS1' or 'PS2' or 'PS3'.

    Returns
    -------
    sessions : list
        List of sessions for given subject and experiments e.g. ['0','1'].

    '''
    path = os.path.join(ROOTDIR,'protocols','r1','subjects',sub,'experiments',
                        exp,'sessions','*')
    sessions = [os.path.split(p)[-1] for p in glob(path)]
    return sessions
    
def get_data(sub="R1056M", exp_sess=[], desiredrate=500):
    '''
    Parameters
    ----------
    sub : string, optional
        Subject name. The default is "R1056M".
    exp_sess : list, optional
        List of (experiment, session) pairs e.g. [['PS2','0']]. The default is [].
        Will output data from all experiments and sessions if empty list is input.
    desiredrate : int, optional
        Desired sampling rate for the data. The default is 500.

    Returns
    -------
    data_chunks : dict
        Dictionary of data with the key being the each (experiment, session) 
        in exp_sess and value being the iEEG time series across all electrode
        chanenels

    '''
    #Assigning exp_sess all experiments and sessions if empty exp_sess is
    #input to the function
    if len(exp_sess)==0:
        exp_sess = get_exp_sess(sub)
    
    data_chunks = {}
    readcontact = False
    for es in exp_sess:
        print('Doing : ', es)
        reader = cml.CMLReader(subject=sub, experiment=es[0], session=es[1],
                               localization=0, montage=0, rootdir=ROOTDIR)
        if not readcontact:
            contact = reader.load('contacts')
            readcontact = True
            
        events = reader.load('task_events')
        eeg = reader.load_eeg(scheme=contact)
        
        #considering the sample range during which events occur
        #s: start index
        #f: end index
        s = int(events['eegoffset'][0])
        f = max(int(events['eegoffset'].to_numpy()[-1])+1,1000000)
        f = min(f,eeg.shape[-1])        
        eeg_data, q, samplerate = _process_data(eeg, desiredrate,s,f)
        
        eegoffset = events['eegoffset'].to_numpy()
        stim_idxs = np.where(events['is_stim'].to_numpy()==True)[0]

        stim_params = events['stim_params'].to_list()
        
        anode_lab = stim_params[stim_idxs[0]][0]['anode_label']
        anode_idx = contact['label'].to_list().index(anode_lab)
        
        cathode_lab = stim_params[stim_idxs[0]][0]['cathode_label']
        cathode_idx = contact['label'].to_list().index(cathode_lab)
        
        u_impulse = np.zeros((eeg_data.shape[-1]))
        u_ptrain = np.zeros((eeg_data.shape[-1]))
        u_ftrain = np.zeros((eeg_data.shape[-1]))
        print('Constructing input')
        if es[0] in ['PS1','PS2']:
            for i in stim_idxs:
                stim_t = eegoffset[i]
                amp = stim_params[i][0]['amplitude']
                freq = stim_params[i][0]['pulse_freq']
                n_pulses = stim_params[i][0]['n_pulses']
                dur = stim_params[i][0]['stim_duration']
                idx = int(q*(stim_t-s))   
                if n_pulses>1:
                    if (dur*freq)%1000!=0:
                        n_pulses = n_pulses+1
                    period = int(np.round((q*samplerate)/freq))
                    dur = int(dur*q)
                    u_ptrain[idx:idx+dur] = amp
                    u_ftrain[idx:idx+dur] = freq
                    for j in range(n_pulses):
                        u_impulse[idx] = amp
                        idx = idx + period                        
                else:
                    u_impulse[idx] = amp
                    u_ptrain[idx] = amp
                    u_ftrain[idx] = -1
                    
        data_chunks[(es[0],es[1])] = [eeg_data, u_impulse, u_ptrain, u_ftrain, 
                                      cathode_idx, anode_idx]
    return data_chunks
    
                    
def _process_data(eeg, desiredrate,s=0,fin=None):
    '''
    Function to resample the data and remove integer multiples of the power
    supply frequency (60Hz, 120Hz,...)

    Parameters
    ----------
    eeg : cmlreaders.eeg_container.EEGContainer 
        EEG data obtained using cml reader.
    desiredrate : int
        Desired sampling frequency (Hz).
    s : int, optional
        start sampling index of the eeg data . The default is 0.
    fin : int, optional
        end sampling index of the eeg data. The default is None.

    Returns
    -------
    eeg_data : np.array
        eeg data as a numpy array of shape n_channels x time.
    q : int
        desiredrate/samplerate.
    samplerate : float
        Original sampling frequency.
    '''
    if fin is None:
        fin = eeg.data.shape[-1]
    print('Resampling')
    samplerate = np.round(eeg.samplerate)       
    q = np.round(desiredrate/samplerate)
    if q>1:
        n = int(q*(fin-s))
        #eeg_data = signal.resample(eeg.data[0,:,s:], n, axis=-1)
        eeg_data = _resampler(eeg.data[0,:,s:fin], n)
    else:
        q = 1
        eeg_data = eeg.data[0,:,s:fin]
    w=2.0
    print('Filtering')
    for f in [60,120,180,240]:
        b, a = signal.butter(4, [f-w/2,f+w/2], btype='bandstop', fs=q*samplerate)
        eeg_data = signal.filtfilt(b,a,eeg_data)
    #detrending data
    ttime = eeg_data.shape[-1]
    x = np.concatenate([np.arange(0,ttime).reshape((ttime,1)), np.ones((ttime,1))],axis=-1)
    x[:,0] = x[:,0]/1000
    G = x.T@x
    g = (eeg_data@x).T
    eeg_data = eeg_data-(x@(np.linalg.inv(G)@g)).T
    return eeg_data, q, samplerate

def _resampler(data, n):
    '''
    Function to resample data to required sampling rate 
    (n is the number of samples required)
    '''
    output = np.zeros((data.shape[0],n))
    channels = data.shape[0]
    for i in range(channels):
        output[i] = signal.resample(data[i], n)
    return output

def _load(path):
    '''
    Function to load a pickle file.
    '''
    with open(path,'rb') as f:
        d = pkl.load(f)
    return d

def zscore(d):
    '''
    Function to zscore the iEEG data 

    '''
    m = []
    v = []
    n = []
    for k in d.keys():
        n.append(d[k][0].shape[-1])
        m.append(np.mean(d[k][0], axis=-1))
        v.append(np.std(d[k][0], axis=-1)**2)
        
    mu = np.zeros(m[0].shape)
    var = np.zeros(v[0].shape)
    N = sum(n)
    for i in range(len(n)):
        mu = mu+n[i]*m[i]/N
    for i in range(len(n)):
        var = var +  n[i]*(v[i]+(m[i]-mu)**2)/N
        
    sigma = var**0.5
    mu = mu.reshape((-1,1))
    sigma = sigma.reshape((-1,1))
    for k in d.keys():
        d[k][0] = (d[k][0]-mu)/sigma
    return d


def _get_X(data_y, data_u=None, inter_u=None, ch=0,lags=10,mlags=2,ilags=1,plags=0,
          anode_only=False,anode_idx=-1):
    '''
    Parameters
    ----------
    data_y : (np.array) iEEG data (shape: time X channels)
    data_u : (np.array, optional) Input time series (shape: time X 1)
    inter_u : (np.array, optional) Interaction input time series (shape: time X 1)
    ch : (int) channel to be predicted Dy_i = f(y_i,u, others)
    lags : (int) y_i lags to be included
    mlgas : (int) u lags to be included
    ilags : (int) interaction lags between y_i and u
    plags : (int) 0,1 or >1 lags of other channels included
    anode_only : (bool) Use only anode channel to model network effects
    anode_idx : (int) Index of the anode channel
    
    Returns
    -------
    (reg_data,y) : Regression features and target iEEG signal as a tuple
    '''
    data = data_y[:,ch].copy()    
    c = list(range(data_y.shape[-1]))

    N = len(data)
    if inter_u is None:
        inter_u = data_u
        if data_u is None:
            ilags = 0
        
    N_ = N-lags
    reg_data_x = np.zeros((N_,(ilags+1)*lags))
    reg_data_u = np.zeros((N_,mlags))
    
     
    #channel i lags
    for i in range(lags):
        reg_data_x[:,i] = data[lags-i-1:-1-i]
        for j in range(1,ilags+1):
            if lags-j+1>=0:
                reg_data_x[:,i+j*lags] = data[lags-i-1:-1-i]*inter_u[lags-j+1:N-j+1]
            else:
                u = np.concatenate([np.zeros((abs(lags-j+1),)),inter_u[0:N-j+1]])
                reg_data_x[:,i+j*lags] = data[lags-i-1:-1-i]*u
        
    #input lags
    reg_data = reg_data_x
    if data_u is not None and mlags>0:
        reg_data_u[:,0] = data_u[lags:]
        for i in range(1,mlags):
            if lags-i>=0:
                reg_data_u[:,i] = data_u[lags-i:-i]
            else:
                reg_data_u[i-lags:,i] = data_u[0:-i]               
        reg_data = np.concatenate([reg_data,reg_data_u],axis=-1)
    

    #data from other channels
    if plags>0:
        c.remove(ch)
        if anode_only:
            c = [anode_idx]
        others = []
        for i in range(plags):
            others.append(data_y[lags-(1+i):-(1+i),c].copy())                  
        others = np.concatenate(others,axis=-1)
        reg_data = np.concatenate([reg_data,others],axis=-1)
        
    y = data[lags:]-data[lags-1:-1]
    return reg_data, y

def get_embedding(cdata, inter_u, idx, tau, lags, ilags):
    N = len(inter_u)
    N_ = N-lags
    
    if inter_u is None:
        ilags = 0
        
    reg_data_x = np.zeros((N_,(ilags+1)*lags))
        
    for i in range(lags):
        start = idx+lags-tau*(i+1)
        if start<0:
            start = 0
        p = cdata[start:-tau*(i+1)].copy()
        reg_data_x[N_-len(p):,i] = p
        
        for j in range(1,ilags+1):
            if lags-j+1>=0:
                reg_data_x[:,i+j*lags] = reg_data_x[:,i]*inter_u[lags-j+1:N-j+1]
            else:
                u = np.concatenate([np.zeros((abs(lags-j+1),)),inter_u[0:N-j+1]])
                reg_data_x[:,i+j*lags] = reg_data_x[:,i]*u
                
    return reg_data_x
    

def create_dataset_windows(d,ch=0,lags=10,mlags=2,ilags=0,plags=0,
                           amp_inter_u=False, anode_only=False,
                           allow_u=False,amp_u=True, do_embed=False, pad_dur=500,
                           tau=10, embed_exp=False, use_baseline=True):
    Tx = []
    Ty = []
    data_u = None
    U = []
    U_amp = []
    anode_idx = d[list(d.keys())[0]][-1]
    key = list(d.keys())[0]
    
    if allow_u==True:
        data_u = d[key][1]/d[key][1].max()
        if amp_u==False:
            data_u[data_u>0] = 1.0
    if use_baseline:
        inter_u = d[key][2]/d[key][2].max()
    else:
        inter_u = d[key][1]/d[key][1].max()
        
    if amp_inter_u==False:
        inter_u[inter_u>0] = 1.0
        
    idxs_start = np.where(np.diff(d[key][2])>0)[0]+1
    idxs_end = np.where(np.diff(d[key][2])<0)[0]
    s_chunk = 0
    f_chunk = 0
    chunks = []
    for i in range(min(len(idxs_start), len(idxs_end))):
        idx = idxs_start[i]-pad_dur-lags-1
        idx_end = idxs_end[i]+pad_dur
        data = d[key][0][:,idx:idx_end].T
        
        u = data_u[idx:idx_end]
        iu = inter_u[idx:idx_end]
        
        tx, ty = _get_X(data,u,iu,ch,lags,mlags,ilags,plags,
                        anode_only,anode_idx)
        if embed_exp:
            ty = ty + tx[:,0]
            
        if do_embed:
            cdata = d[key][0][ch,:idx_end]
            tx[:,:(ilags+1)*lags] = get_embedding(cdata, iu, idx, tau, lags, ilags)
        
        f_chunk = s_chunk + ty.shape[0]
        chunks.append([s_chunk,f_chunk])
        s_chunk = f_chunk
        U.append(d[key][3][idx+lags:idx_end])
        U_amp.append(d[key][2][idx+lags:idx_end])
        Tx.append(tx.copy())
        Ty.append(ty.copy())
    Tx = np.concatenate(Tx,axis=0)
    Ty = np.concatenate(Ty,axis=0)
    U_freq = np.concatenate(U,axis=0).astype(int)
    U_amp = np.concatenate(U_amp,axis=0).astype(int)
    return Tx, Ty, U_freq, U_amp, chunks

def get_chunk_param(U_freq,U_amp,chunks):
    param_chunk = {}
    for j,c in enumerate(chunks):
        s,f = c
        amp = U_amp[s:f].max()
        freq = U_freq[s:f].max()
        if freq==0:
            freq=-1
        if (amp,freq) not in param_chunk:
            param_chunk[(amp,freq)]=[]

        param_chunk[(amp,freq)].append(j)
        
    return param_chunk
        
def get_tt(U_freq,U_amp,chunks,trate=0.25,seed=10):
    #get_tt: get_test_train
    param_chunk = get_chunk_param(U_freq,U_amp,chunks)
    freqs = np.unique(U_freq)
    train = {f:[] for f in freqs}
    test = {f:[] for f in freqs}
    rng = np.random.RandomState(seed)
    for p in param_chunk:
        n = len(param_chunk[p])
        n_test = int(n*trate)
        idxs = np.arange(n)
        rng.shuffle(idxs)
        for i in idxs:
            c = chunks[param_chunk[p][i]]
            if i<n_test:
                test[p[-1]].extend(list(range(c[0],c[1])))
            else:
                train[p[-1]].extend(list(range(c[0],c[1])))
    full_train = np.concatenate([train[f] for f in train],axis=0).astype(int)
    full_test = np.concatenate([test[f] for f in test],axis=0).astype(int)
        
    return train, test, full_train, full_test

def create_dataset_rs(d,ch=0,lags=10,plags=0):
    Tx = []
    Ty = []
   
    key = list(d.keys())[0]
    
    u = d[list(d.keys())[0]][3]
    freq_idxs = get_freq_idxs(u, rs_steps=0)
        
    s_chunk = 0
    f_chunk = 0
    chunks = []
    for i in range(len(freq_idxs[0])):
        
        idx = freq_idxs[0][i,0]+500
        idx_end = freq_idxs[0][i,1]-500
        data = d[key][0][:,idx:idx_end].T
        
        tx, ty = _get_X(data,lags=lags,plags=plags, ilags=0)
        
        
        f_chunk = s_chunk + ty.shape[0]
        chunks.append([s_chunk,f_chunk])
        s_chunk = f_chunk
        
        Tx.append(tx.copy())
        Ty.append(ty.copy())
    Tx = np.concatenate(Tx,axis=0)
    Ty = np.concatenate(Ty,axis=0)
    
    return Tx, Ty, chunks
        


def get_freq_idxs(U,srate=1000,rs_steps=0):
    '''
    Function that outputs time indices corresponding to different stim input 
    frequencies
    Parameters
    ----------
    U : 1D np.array 
        input frequency time series.
    srate : int, optional
        Sampling rate of signal. The default is 1000.
    rs_steps : int, optional
        Time steps after end of stim to be considered. The default is 0.
        
    Returns
    -------
    freq_idxs : dict
        Dictionary of keys (input frequencies) and 
        values ([[start, end],[],...]).

    '''
    freqs = np.unique(U).astype(int)
    freq_idxs = {}
    for f in freqs:
        freq_idxs[f]=[]
    s = -1
    prev_u = -1
    for i in range(len(U)):
        u = U[i]
    
        if prev_u!=u and s>0:
            freq_idxs[u].append([i])
            freq_idxs[prev_u][-1].append(i-1)
        if s==-1:
            freq_idxs[u].append([i])
            s = 1
        prev_u = u
    freq_idxs[prev_u][-1].append(i)
    
    #ignoring last resting state interval if it is not long enough
    if freq_idxs[0][-1][1]-freq_idxs[0][-1][0]<rs_steps+500:
        freq_idxs[0] = freq_idxs[0][:-1]
        
    for f in freq_idxs:
        freq_idxs[f] = np.array(freq_idxs[f])
        if f!=0:
            freq_idxs[f][:,1] = freq_idxs[f][:,1]+rs_steps
            freq_idxs[f][:,1][freq_idxs[f][:,1]>len(U)] = len(U)
                
        if f==0:
            for i in range(0,freq_idxs[0].shape[0]):
                z = freq_idxs[f][i,0]
                if z==0:
                    continue

                freq_idxs[f][i,0] = z+rs_steps
                
    return freq_idxs 
    
def kfolder(U,chunks,k=5,seed=10,gap=25,skip=[0,]):
    rng = np.random.default_rng(seed)
    full_test = []
    full_train = []
    train = {}
    test = {}
    for f in np.unique(U).astype(int):
        if f in skip:
            continue
        train[f] = []
        test[f] = []
        
    for c in chunks:
        f = int(U[c[0]:c[1]].max())
        if f==0:
            f=-1
        if f in skip:
            continue

        idxs = set(range(c[0],c[1]))
        c_size = int((c[1]-c[0])/k)

        k_idx = rng.integers(0,k)
        start = c[0]+c_size*k_idx
        end = start + c_size
        test[f].extend(list(range(start+gap,end-gap)))
        
        train[f].extend(list(idxs.difference(set(range(start,end)))))
        
    full_train = np.concatenate([train[f] for f in train],axis=0)
    full_test = np.concatenate([test[f] for f in test],axis=0)
    return train, test, full_train, full_test

def kfolder_rs(chunks,k=5,seed=10,gap=25):
    rng = np.random.default_rng(seed)
    test = []
    train = []

    for c in chunks:
        
        idxs = set(range(c[0],c[1]))
        c_size = int((c[1]-c[0])/k)

        k_idx = rng.integers(0,k)
        start = c[0]+c_size*k_idx
        end = start + c_size
        test.extend(list(range(start+gap,end-gap)))
        
        train.extend(list(idxs.difference(set(range(start,end)))))
        
    return train, test