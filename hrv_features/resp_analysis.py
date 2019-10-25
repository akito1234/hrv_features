# Import Package

import numpy as np
import pandas as pd
from opensignalsreader import OpenSignalsReader
from biosppy.signals.resp import resp
import pyhrv.time_domain as td
import pyhrv.nonlinear as nl
import pyhrv.tools as tools


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

if __name__ == '__main__':
    path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-14\opensignals_dev_2019-10-14_11-40-05.txt"
    arc = OpenSignalsReader(path)
    resp_data = resp(signal=arc.signal('RESP'), sampling_rate=1000.0,show=False)['resp_rate_ts']
    for keys in RESP_FEATURES(resp_data):
        print(keys, RESP_FEATURES(resp_data)[keys])

