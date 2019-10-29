# -*- coding: utf-8 -*-
"""
biosppy.signals.resp
--------------------
This module provides methods to process Respiration (Resp) signals.
:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
## compat
#from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np

# local
import biosppy.signals.tools as st
from biosppy import plotting, utils
from opensignalsreader import OpenSignalsReader
from scipy import signal
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
        Respiration rate time axis reference (seconds).
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
    zeros, = st.zero_cross(signal=filtered, detrend=True)
    beats = zeros[::2]

    if len(beats) < 2:
        rate_idx = []
        rate = []
    else:
        inspiration = zeros[1:][::2]
        expiration = zeros[::2]

        #recoveries = zeros[::2]
        # peak detection 
        #peaks = []
        #for (onset,recovery) in zip(onsets,recoveries):
        #    if onset >= recovery:
        #        continue
        #    # 最大値のインデックスを取得
        #    index_max = np.argmax(filtered[onset:recovery])
        #    peaks.append(index_max + onset)

        # compute respiration rate
        rate_idx = beats[1:]
        rate = sampling_rate * (1. / np.diff(beats))

        # physiological limits
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
    args = (ts, filtered, zeros, ts_rate, rate,inspiration, expiration)
    names = ('ts', 'filtered', 'zeros', 'resp_rate_ts', 'resp_rate','inspiration', 'expiration')

    return utils.ReturnTuple(args, names)

if __name__ == '__main__':
    path = r"C:\Users\akito\Desktop\test.txt"
    arc = OpenSignalsReader(path)
    result = resp(signal=arc.signal('RESP'), sampling_rate=1000., show=False)


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(result['ts'],result['filtered'])

    for ins,exp in zip(result['inspiration'], result['expiration']):
        plt.axvline(ins*0.001,color= 'b')
        plt.axvline(exp*0.001,color= 'r')
    plt.show()
    #import matplotlib.pyplot as plt
    #import matplotlib as mpl
    ## welch 法によるスペクトル解析
    #plt.figure()
    #A = result['filtered']
    #filter= result['filtered'] - A.mean()
    #N = 300*1000
    #freq1,power1 = signal.welch(filter[(0<=result['ts']) & (result['ts']<300)], fs=1000.0, window='hanning',nperseg=N)
    
    #plt.plot(freq1,power1,"b")

    #freq2,power2 = signal.welch(filter[(300<=result['ts']) & (result['ts']<600)], fs=1000.0, window='hanning',nperseg=N)
    
    #plt.plot(freq2,power2,"r")

    #plt.xlabel("Frequency[Hz]")
    #plt.ylabel("Power/frequency[dB/Hz]")
    #plt.xlim(0,10)
    #plt.show()

    #fig, ax = plt.subplots() 
    #N = 2**12
    #fs = 1000
    #f, t, Sxx = signal.spectrogram(filter,
    #                               nperseg=N,
    #                               #noverlap=N,
    #                               detrend='linear')
    ##ax[0].plot(result['ts'],result['filtered'])
    #pc = ax.pcolormesh(t/fs, f, Sxx, norm=mpl.colors.LogNorm(vmin=Sxx.mean(), vmax=Sxx.max()), cmap='inferno')
    #ax.set_ylim(0,0.5)
    #ax.set_ylabel('Frequency')
    #ax.set_xlabel('Time') 
    ##fig.colorbar(pc)
    #plt.show()