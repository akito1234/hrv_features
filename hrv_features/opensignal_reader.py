# -*- coding: utf-8 -*-
"""
Created 2019.10.12 17:00

@author: akito
"""

# Import OpenSignalsReader
from opensignalsreader import OpenSignalsReader
import pyhrv.tools as tools
from biosppy import signals
# Import BasePlugins
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hrv_analysis as hf


path = r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-11\tohma\opensignals_dev_2019-10-11_17-29-23.txt"
# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)

# peak detection
signal, rpeaks= signals.ecg.ecg(signal= arc.signal("ECG"), sampling_rate=1000.0, show=True)[1:3]


nni = tools.nn_intervals(rpeaks.tolist())
np.savetxt(r"Z:\00_個人用\東間\01.theme\卒論関連\2020年度要旨\nni_opensignals_dev_2019-10-11_17-29-23.csv"
           ,nni, delimiter=',' )


np.savetxt(r"Z:\00_個人用\東間\01.theme\卒論関連\2020年度要旨\ecg_opensignals_dev_2019-10-11_17-29-23.csv"
           ,np.c_[arc.t,signal], delimiter=',' )

## RRIから特徴量を算出する
#nni= np.loadtxt(r"C:\Users\akito\Desktop\shibata_ecg.csv",delimiter=",")
#result = hf.segumentation_freq_features(nni,sample_time=300,time_step=30)

## 結果をエクスポート
##np.savetxt(r"C:\Users\akito\Downloads\opensignals_201808080162_13-58-37_converted.csv"
##           ,nni, delimiter=',' )
#result.to_excel(r"C:\Users\akito\Desktop\features_shibata_300s_30s.xlsx")
