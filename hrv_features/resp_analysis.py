# -*- coding: utf-8 -*-
"""
akito

2019-11-01
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import biosppy.signals.tools as st
from biosppy import plotting, utils

# local
from opensignalsreader import OpenSignalsReader

def RESP_FEATURES(resp_peaks):
    BRI = resp_peaks[1:] - resp_peaks[:-1]
    resp_features = {}
    #------時系列解析------#
    L = len(resp_peaks)   
    resp_features['BRI_Mean'] = np.mean(BRI)
    resp_features['BRI_Max'] = np.max(BRI)
    resp_features['BRI_Min'] = np.min(BRI)
    resp_features['BRI_SDNN'] = np.std(BRI)
    resp_features['BRI_SDSD'] = np.std(np.diff(BRI))
    resp_features['BRI_rMSSD'] = np.sqrt((1/L) * sum(np.diff(BRI) ** 2))        
    resp_features['BRI_MedianNN'] =np.median(BRI)

    #-----ポアンカレプロット-----#
    _,resp_features['BRI_sd1'],resp_features['BRI_sd2'],resp_features['BRI_sd_ratio'],resp_features['BRI_ellipse_area']=nl.poincare(rpeaks=resp_peaks.astype(int).tolist(),show=False)
    return resp_features



def resp(signal=None, sampling_rate=1000., show=True):
    """Process a raw Respiration signal and extract relevant signal features
    using default parameters.
    Parameters
    ----------
    signal : array
        Raw Respiration signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.
    Returns

    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered Respiration signal.
    zeros : array
        Indices of Respiration zero crossings.
    resp_rate_ts : array
        Inspiration rate time axis reference (seconds).
    resp_rate : array
        Instantaneous respiration rate (Hz).
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)

    # filter signal
    # 0.1 ~~ 0.35 Hzのバンドパスフィルタ
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='bandpass',
                                      order=2,
                                      frequency=[0.1, 0.35],
                                      sampling_rate=sampling_rate)

    # compute zero crossings
    filtered = filtered - np.mean(filtered)

    # zeros
    df = np.diff(np.sign(filtered))
    inspiration = np.nonzero(df > 0)[0]
    expiration = np.nonzero(df < 0)[0]

    if len(inspiration) < 2:
        rate_idx = []
        rate = []
    else:
        # compute resp peaks between inspiration and expiration
        peaks = []
        for i in range(len(inspiration)-1):
            cycle = filtered[inspiration[i]:inspiration[i+1]]
            peaks.append(np.argmax(cycle) + inspiration[i])
        # list to array
        peaks = np.array(peaks)


        # compute respiration rate
        rate_idx = inspiration[1:]
        rate = sampling_rate * (1. / np.diff(inspiration))

        # physiological limits
        # 0.35Hz以下のresp_rateは省かれる
        indx = np.nonzero(rate <= 0.35)
        rate_idx = rate_idx[indx]
        rate = rate[indx]

        # smooth with moving average
        size = 3
        rate, _ = st.smoother(signal=rate,
                              kernel='boxcar',
                              size=size,
                              mirror=True)
         
    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_rate = ts[rate_idx]

    # plot
    if show:
        plotting.plot_resp(ts=ts,
                           raw=signal,
                           filtered=filtered,
                           zeros=zeros,
                           resp_rate_ts=ts_rate,
                           resp_rate=rate,
                           path=None,
                           show=True)
    # output
    args = (ts, filtered, ts_rate, rate,inspiration, expiration,peaks)
    names = ('ts', 'filtered', 'resp_rate_ts', 'resp_rate','inspiration', 'expiration','peaks')
    return utils.ReturnTuple(args, names)

def resp_psd(ts,filtered_signal):
    # welch 法によるスペクトル解析
    plt.figure()
    A = filtered_signal
    filter= filtered_signal - A.mean()
    N = 300*1000
    freq1,power1 = signal.welch(filter[(0<=ts) & (ts<300)], fs=1000.0, window='hanning',nperseg=N)
    plt.plot(freq1,power1,"b")
    freq2,power2 = signal.welch(filter[(300<=ts) & (ts<600)], fs=1000.0, window='hanning',nperseg=N)
    plt.plot(freq2,power2,"r")
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Power/frequency[dB/Hz]")
    plt.xlim(0,10)
    plt.show()



if __name__ == '__main__':
    path = r"Z:\theme\mental_stress\02.BiometricData\2019-10-28\shibata\opensignals_dev_2019-10-28_13-50-02.txt"
    arc = OpenSignalsReader(path)
    result = resp(signal=arc.signal('RESP'), sampling_rate=1000., show=False)

    fig,axes = plt.subplots(2,1,sharex=True)
    axes[0].plot(result['peaks'][1:]*0.001,np.diff(result['peaks'])*0.001)
    axes[1].plot(result['ts'],result['filtered'])
    for ins,exp,peak in zip(result['inspiration'], result['expiration'], result['peaks']):
        #axes[1].axvline(ins*0.001,color= 'b') 
        #axes[1].axvline(exp*0.001,color= 'r')
        axes[1].axvline(peak*0.001,color= 'g')
    plt.show()
