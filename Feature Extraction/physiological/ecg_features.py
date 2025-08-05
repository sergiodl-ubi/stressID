import numpy as np
import scipy.io
import pandas as pd

import os
from pathlib import Path

import scipy.stats as stats
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import neurokit2 as nk
from neurokit2.hrv import hrv_utils
from neurokit2.signal import signal_psd

import warnings
warnings.filterwarnings('ignore')

from numpy.typing import ArrayLike
type FeatureDict = dict[str, np.ndarray]

############################ STAT FEATURES ##############################

def ecg_stat(array: ArrayLike) -> pd.DataFrame:
    x = np.array(array)
    df = pd.DataFrame(data = [x.max(), x.min(), x.mean(), x.std(),
                              stats.kurtosis(x), stats.skew(x), np.quantile(x,0.5),
                              np.quantile(x, 0.25), np.quantile(x,0.75),
                              np.quantile(x, 0.05), np.quantile(x, 0.95)]).T
    df.columns = ['max_ecg', 'min_ecg', 'mean_ecg', 'sd_ecg', 'ku_ecg', 'sk_ecg', 'median_ecg',
                  'q1_ecg', 'q3_ecg', 'q05_ecg', 'q95_ecg']
    
    return df
 
 
############################ HRV FEATURES ##############################    

def ecg_peaks(array: ArrayLike, sampling_rate: float=1000) -> np.ndarray:
    x = np.array(array)
    
    # distance : minimal horizontal distance (>= 1) in samples between neighbouring peaks. By default = None
    # height : required height of peaks. By default = None
    
    distance = sampling_rate / 3
    #height = x.max() / 2
    height = np.quantile(x, 0.99)/2 #use 99th quantile instead of max (to ignore outliers)
    r_peaks, _ = signal.find_peaks(x, distance= distance, height=height)
    
    return r_peaks
    
    
def ecg_time(array: ArrayLike, sampling_rate: float = 1000) -> pd.DataFrame:
    x = np.array(array)
    r_peaks = ecg_peaks(x, sampling_rate=sampling_rate)
    
    # RR intervals are expressed in number of samples and need to be converted into ms. By default, sampling_rate=1000
    # HR is given by: 60/RR intervals in seconds. 
    
    rri = np.diff(r_peaks) #RR intervals
    rri = 1000 * rri / sampling_rate #Convert to ms
    drri = np.diff(rri) #Difference between successive RR intervals
    hr = 1000*60 / rri #Heart rate
    
    meanHR = hr.mean()
    minHR = hr.min()
    maxHR = hr.max()
    sdHR = hr.std()
    modeHR =  maxHR - minHR
    nNN = rri.shape[0] / (x.shape[0]/sampling_rate/60) #to get the number of NN intervals per minute
    meanNN = rri.mean()
    SDSD = drri.std()
    CVNN = rri.std()
    SDNN = CVNN / meanNN
    pNN50 = np.sum(np.abs(drri)>50) / nNN * 100
    pNN20 = np.sum(np.abs(drri)>20) / nNN * 100
    RMSSD = np.sqrt(np.mean(drri**2))
    medianNN = np.quantile(rri, 0.5)
    q20NN = np.quantile(rri, 0.2)
    q80NN = np.quantile(rri, 0.8)
    minNN = rri.min()
    maxNN = rri.max()
    
    # HRV triagular index (HTI): The density distribution D of the RR intervals is estimated. The most frequent
    # RR interval lenght X is established. Y= D(X) is the maximum of the sample density distribution.
    # The HTI is then obtained as : (the number of total NN intervals)/Y

    bins = np.arange(meanNN - 3*CVNN , meanNN + 3*CVNN, 10)
    d,_ = np.histogram(rri, bins=bins)
    y = d.argmax()
    triHRV = nNN / y
    
    df = pd.DataFrame(data = [meanHR, minHR, maxHR, sdHR, modeHR, nNN, meanNN, 
                              SDSD, CVNN, SDNN, pNN50, pNN20, RMSSD, medianNN,
                              q20NN, q80NN, minNN, maxNN, triHRV]).T
    
    df.columns = ['meanHR', 'minHR', 'maxHR', 'sdHR', 'modeHR', 'nNN','meanNN', 'SDSD', 'CVNN', 
                  'SDNN', 'pNN50', 'pNN20', 'RMSSD', 'medianNN', 'q20NN','q80NN','minNN', 'maxNN', 'triHRV']
    return df
    
    

############################ FREQ FEATURES ##############################
    
def ecg_freq(
    array: ArrayLike,
    sampling_rate: float=1000,
    interpolation_rate: int=100,
    ) -> pd.DataFrame:

    columns = ['totalpower', 'LF', 'HF', 'ULF', 'VLF', 'VHF', 'LF/HF', 'rLF', 'rHF', 'peakLF', 'peakHF']
    x = np.array(array)
    r_peaks = ecg_peaks(x, sampling_rate=sampling_rate)

    if len(r_peaks) < 3:
        return pd.DataFrame([[np.nan]*len(columns)], columns=columns)

    # Sanitize input
    # If given peaks, compute R-R intervals (also referred to as NN) in milliseconds
    rri, rri_time, _ = hrv_utils._hrv_format_input(r_peaks, sampling_rate=sampling_rate)

    # Process R-R intervals (interpolated at 100 Hz by default)
    rri, rri_time, sampling_rate = nk.intervals_process(
        rri, intervals_time=rri_time, interpolate=True, interpolation_rate=interpolation_rate
    )

    psd = signal_psd(
        rri, sampling_rate=sampling_rate,
        min_frequency=0.001, max_frequency=0.5, normalize=False, order_criteria=None)



    hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate, normalize=False, interpolation_rate=interpolation_rate)

    hrv_lf = hrv_freq['HRV_LF'].iloc[0]
    hrv_hf = hrv_freq['HRV_HF'].iloc[0]
    if (hrv_lf + hrv_hf) > 0:
        hrv_rlf = hrv_lf / (hrv_lf + hrv_hf) * 100
        hrv_rhf = hrv_hf / (hrv_lf + hrv_hf) * 100
    else:
        hrv_rlf = np.nan
        hrv_rhf = np.nan

    df = pd.DataFrame(data = [
        hrv_freq['HRV_TP'].iloc[0],
        hrv_freq['HRV_LF'].iloc[0],
        hrv_freq['HRV_HF'].iloc[0],
        hrv_freq['HRV_ULF'].iloc[0],
        hrv_freq['HRV_VLF'].iloc[0],
        hrv_freq['HRV_VHF'].iloc[0],
        hrv_freq['HRV_LFHF'].iloc[0],
        hrv_rlf, hrv_rhf, peaklf, peakhf,
        ]
    ).T
    df.columns = columns

    return df
    
    
    
    
############################ NONLIN FEATURES ##############################  


def apEntropy(array: ArrayLike, m: int=2, r: float | None=None) -> float:
    # m : a positive integer representing the length of each compared run of data (a window).
    # By default m = 2
    # r : a positive real number specifying a filtering level. By default r = 0.2 * sd.
    
    x = np.array(array)
    N = len(x)
    r = 0.2 * x.std() if r == None else r == r
    
    # A sequence of vectors z(1),..., z(N-m+1) is formed from a time series of N equally 
    # spaced raw data values x(1),…,x(N), such that z(i) = x(1),...,x(i+m-1).
    
    # For each i in {1,..., N-m+1}, C = [number of z(j) such that d(x(i),x(j)) < r]/[N-m+1]
    # is computed, with d(z(i),z(j)) = max|x(i)-x(j)|
    
    # phi_m(r) = (N-m+1)^-1 x sum log(Ci) is computed, and the ap Entropy is given by: 
    # phi_m(r) - phi_m+1(r)

    def _maxdist(zi, zj):
        return max([abs(xi - xj) for xi, xj in zip(zi, zj)])
    
    def _phi(m):
        z = [[x[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for zj in z if _maxdist(zi, zj) <= r]) / (N - m + 1.0)
        for zi in z]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    
    apEn = abs(_phi(m + 1) - _phi(m))
    return apEn
    
    
def sampEntropy(array: ArrayLike, m: int=2, r: float | None=None) -> float:
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A Float
    
    # m: embedding dimension
    # r: tolerance distance to consider two data points as similar. By default r = 0.2 * sd.
    
    # SampEn is the negative logarithm of the probability that if two sets of simultaneous 
    # data points of length m have distance < r then two sets of simultaneous data points of 
    # length m + 1 also have distance < r. 
    
    x = np.array(array)
    N = len(x)
    r = 0.2 * x.std() if r == None else r == r
    
    # All templates vector of length m are defined. Distances d(xmi, xmj) are computed
    # and all matches such that d < r are saved. Same for the distances d(xm+1i, xm+1j).
    
    xmi = np.array([x[i : i + m] for i in range(N - m)])
    xmj = np.array([x[i : i + m] for i in range(N - m + 1)])
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    m += 1
    xm = np.array([x[i : i + m] for i in range(N - m + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    
    return -np.log(A / B)
    
    
def ecg_nonlinear(array: ArrayLike, sampling_rate: float=1000, m: int=2, r: float | None=None) -> pd.DataFrame:
    # Input: An array (numpy, dataframe, list) 
    # -> Output : A numpy array 
    
    # The Poincaré ellipse plot is diagram in which each RR intervals are plotted as a function 
    # of the previous RR interval value. SD1 is the standard deviation spread orthogonally 
    # to the identity line (y=x) and is the ellipse width. SD2 is the standard deviation spread
    # along the identity line and specifies the length of the ellipse.
    
    x = np.array(array)
    r_peaks = ecg_peaks(x, sampling_rate=sampling_rate)
    rri = np.diff(r_peaks)
    rr1 = rri[:-1]
    rr2 = rri[1:]
    SD1 =  np.std(rr2 - rr1) / np.sqrt(2)
    SD2 =  np.std(rr2 + rr1) / np.sqrt(2)
    SD1SD2 = SD1/SD2
    apEn = apEntropy(r_peaks, m=2)
    sampEn = sampEntropy(r_peaks, m=2)
    
    df = pd.DataFrame(data = [SD1, SD2, SD1SD2, apEn, sampEn]).T
    
    df.columns = ['SD1', 'SD2', 'SD1SD2', 'apEn', 'sampEn']
    return df
    



############################ DATABASE ##############################

def ecg_stat_features(dictionary: FeatureDict) -> pd.DataFrame:
    data = dictionary.copy()
    df_stat = ecg_stat( list(data.values())[0] )
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    _ = next(items)
    
    for k,v in items:
        df_stat = pd.concat([df_stat, ecg_stat(v)], axis=0) 
        names.append(k)
    
    return df_stat.set_axis(names)

def ecg_time_features(dictionary: FeatureDict, sampling_freq: float =500) -> pd.DataFrame:
    data = dictionary.copy()
    df_time = ecg_time(list(data.values())[0] , sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    _ = next(items)
    
    for k,v in items:
        df_time = pd.concat([df_time, ecg_time(v, sampling_freq)], axis=0)
        names.append(k)
        
    return df_time.set_axis(names)


def ecg_freq_features(dictionary: FeatureDict, sampling_freq: float =500) -> pd.DataFrame:
    data = dictionary.copy()
    df_freq = ecg_freq(list(data.values())[0], sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    _ = next(items)
    
    for k,v in items:
        df_freq = pd.concat([df_freq, ecg_freq(v, sampling_freq)], axis=0)
        names.append(k)
        
    return df_freq.set_axis(names)

def ecg_nonlinear_features(dictionary: FeatureDict, sampling_freq: float =500) -> pd.DataFrame:
    data = dictionary.copy()
    df_nonlinear = ecg_nonlinear( list(data.values())[0], sampling_freq)
    names = [list(data.keys())[0]]
    
    items = iter(data.items())
    _ = next(items)
    
    for k,v in items:
        df_nonlinear = pd.concat([df_nonlinear, ecg_nonlinear(v, sampling_freq)], axis=0)
        names.append(k)
        
    return df_nonlinear.set_axis(names)


def get_ecg_features(dictionary: FeatureDict, sampling_freq: float=500) -> pd.DataFrame:
    df_stat = ecg_stat_features(dictionary)
    df_time = ecg_time_features(dictionary, sampling_freq)
    df_freq = ecg_freq_features(dictionary, sampling_freq)
    df_nonlinear = ecg_nonlinear_features(dictionary, sampling_freq)
    
    df = pd.concat([df_time, df_stat, df_freq, df_nonlinear], axis=1)
    #df.set_axis(names, inplace=True)
    return df