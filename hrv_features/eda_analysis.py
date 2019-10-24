# Import Package
from biosppy.signals.eda import eda
import numpy as np
import pandas as pd


arc = OpenSignalsReader(path)
eda_data = eda.eda(arc.signal('EDA'),show=False)

def EDA_FEATURES(eda, onset=0,duration=300):
    eda_features = {}
    #SCR
    eda_filter = (eda['peaks']>= onset) & (eda['peaks'] <=(onset+duration))

    # ピークの振幅
    eda_features['SCR_Amplitude_Mean'] = np.mean(eda['amplitudes'][eda_filter])
    eda_features['SCR_Amplitude_Max']  = np.max(eda['amplitudes'][eda_filter])

    # SCR_Latency
    SCR_Latency = eda['onsets'][eda_filter][1:] - eda['onsets'][eda_filter][:-1] 
    eda_features['SCR_Latency'] = np.mean(SCR_Latency)

    # SCR_PeakTime
    SCR_PeakTime = eda['peaks'][eda_filter][1:] - eda['peaks'][eda_filter][:-1] 
    eda_features['SCR_PeakTime'] = np.mean(SCR_PeakTime)

    # SCR_RiseTImes
    SCR_RiseTImes = eda['peaks'][eda_filter] - eda['onsets'][eda_filter]
    eda_features['SCR_RiseTImes'] = np.mean(SCR_RiseTImes)

    return eda_features


def cal_eda_parameters(eda):
    #relative peak time
    peak_time = eda['peaks'] - eda['onsets']

    nSRR = cal_nSRR(eda['peaks'])
    
    # Output
    args = (eda['peaks'][:-1], peak_time[:-1]
            , nSRR, eda['amplitudes'][:-1])

    names = ('peaks', 'SRT', 'SRR', 'SRA')
    return utils.ReturnTuple(args, names)



def cal_nSRR(rpeaks=None):
    # Confirm numpy arrays & compute RR intervals
    rpeaks = np.asarray(rpeaks)
    nn_int = np.zeros(rpeaks.size - 1)
    
    for i in range(nn_int.size):
        nn_int[i] = rpeaks[i + 1] - rpeaks[i]
    return nn_int
