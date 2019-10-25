# Import Package
from biosppy.signals.eda import eda
import numpy as np
import pandas as pd
from opensignalsreader import OpenSignalsReader


def EDA_FEATURES(eda,section):
    eda_features = {}
    onsets = eda[0]
    peaks = eda[1]
    amplitude = eda[2]
    eda_filter = (peaks>= section[0]*1000) & (peaks <= section[1]*1000)
    # ピークの振幅
    eda_features['SCR_Amplitude_Mean'] = np.mean(amplitude[eda_filter])
    eda_features['SCR_Amplitude_Max']  = np.max(amplitude[eda_filter])

    # SCR_Latency
    SCR_Latency = onsets[eda_filter][1:] - onsets[eda_filter][:-1] 
    eda_features['SCR_Latency'] = np.mean(SCR_Latency)

    # SCR_PeakTime
    SCR_PeakTime = peaks[eda_filter][1:] - peaks[eda_filter][:-1] 
    eda_features['SCR_PeakTime'] = np.mean(SCR_PeakTime)

    # SCR_RiseTimes
    SCR_RiseTImes = peaks[eda_filter] - onsets[eda_filter]
    eda_features['SCR_RiseTImes'] = np.mean(SCR_RiseTImes)

    return eda_features

if __name__ == '__main__':
    path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-14\opensignals_dev_2019-10-14_11-40-05.txt"
    arc = OpenSignalsReader(path)
    eda_data = eda(arc.signal('EDA'),show=True)
    for keys in EDA_FEATURES(eda_data):
        print(keys, EDA_FEATURES(eda_data)[keys])