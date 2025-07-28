# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportUnknownVariableType=false

from ntpath import samefile
import numpy as np
from optype.numpy import AnyStrArray
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import scipy.stats as stats
import neurokit2 as nk

import warnings
warnings.filterwarnings('ignore')

from numpy.typing import ArrayLike
type FeatureDict = dict[str, np.ndarray]


############################ STAT FEATURES ##############################

def eda_stat(array: ArrayLike, sampling_freq: float=1000) -> pd.DataFrame:
    x = np.array(array)
    eda = nk.eda_phasic(eda_signal=x, sampling_rate=sampling_freq)
    scr = np.array(eda['EDA_Phasic'])
    scl = np.array(eda['EDA_Tonic'])
    x_axis = np.linspace(0, scl.shape[0]/sampling_freq, scl.shape[0])
    slope = np.polyfit(x_axis,scl,1)[0]
    
    df = pd.DataFrame(data = [x.max(), x.min(), x.mean(), x.std(),
                              stats.kurtosis(x), stats.skew(x), np.quantile(x,0.5),
                              x.max()/x.min(), slope, scr.max(), scr.min(), scr.mean(), scr.std(),
                              scl.max(), scl.min(), scl.mean(), scl.std()]).T

    df.columns = ['max_eda', 'min_eda', 'mean_eda', 'sd_eda', 'ku_eda', 'sk_eda', 'median_eda',
                  'dynrange','scl_slope', 'max_scr', 'min_scr', 'mean_scr', 'sd_scr',
                  'max_scl', 'min_scl', 'mean_scl', 'sd_scl']
    
    return df

############################ TIME FEATURES ##############################

def eda_time(array: ArrayLike, sampling_freq: float=1000) -> pd.DataFrame:
    x = np.array(array)
    eda = nk.eda_phasic(x, sampling_freq)
    scr = np.array(eda['EDA_Phasic'])
    
    _, info = nk.eda_peaks(scr, sampling_freq)
    peaks = info['SCR_Peaks']
    amplitude = info['SCR_Amplitude']
    recovery = info['SCR_RecoveryTime']
    
    nSCR = len(info['SCR_Peaks']) / (x.shape[0]/sampling_freq/60) #to get the number of peaks per minute
    aucSCR = np.trapezoid(scr)
    meanAmpSCR = np.nanmean(amplitude)
    maxAmpSCR = np.nanmax(amplitude)
    meanRespSCR = np.nanmean(recovery)
    sumAmpSCR = np.nansum(amplitude) / (x.shape[0]/sampling_freq/60) # per minute
    sumRespSCR = np.nansum(recovery) / (x.shape[0]/sampling_freq/60) # per minute

        
    df = pd.DataFrame(data = [nSCR, aucSCR, meanAmpSCR, maxAmpSCR, meanRespSCR,
                             sumAmpSCR, sumRespSCR]).T
    
    df.columns = ['nSCR', 'aucSCR', 'meanAmpSCR', 'maxAmpSCR', 'meanRespSCR',
                  'sumAmpSCR', 'sumRespSCR']
    return df

############################ DATABASE ##############################

def eda_stat_features(dictionary: FeatureDict, sampling_freq: float=1000):
    data = dictionary.copy()
    df_stat = eda_stat(list(data.values())[0], sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_stat = pd.concat([df_stat, eda_stat(v, sampling_freq)], axis=0)
        names.append(k)
        
    return df_stat.set_axis(names)


def eda_time_features(dictionary: FeatureDict, sampling_freq: float=1000) -> pd.DataFrame:
    data = dictionary.copy()
    df_time = eda_time(list(data.values())[0], sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    first_item = next(items)
    
    for k,v in items:
        df_time = pd.concat([df_time, eda_time(v, sampling_freq)], axis=0)
        names.append(k)
        
    return df_time.set_axis(names)

def get_eda_features(dictionary: FeatureDict, sampling_freq: float=500) -> pd.DataFrame:
    df_stat = eda_stat_features(dictionary, sampling_freq)
    df_time = eda_time_features(dictionary, sampling_freq)
    
    return pd.concat([df_stat,df_time], axis=1)