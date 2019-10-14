# -*- coding: utf-8 -*-
"""
Created 2019.10.12 17:00

@author: akito
"""

# Import OpenSignalsReader
from opensignalsreader import OpenSignalsReader
from biosppy import signals
import pyhrv.tools as tools

# Import BasePlugins
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hrv_features as hf

# Read OpenSignals file
path = r"Z:\theme\stress\02.BiometricData\2019-10-14\kojima\opensignals_201806130003_2019-10-14_20-29-40.txt"

# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)
ecg = arc.signal('ECG')


# 心拍データからピークを取り出す
signal, rpeaks = signals.ecg.ecg(signal= ecg , sampling_rate=1000.0, show=False)[1:3]
nni = tools.nn_intervals(rpeaks.tolist())

# RRIから特徴量を算出する
result = hf.segumentation_features(nni,sample_time=120,time_step=30)

# 結果をエクスポート
np.savetxt(r'Z:\theme\stress\03.Analysis\Analysis_BioSignal\RRI_kojima_2019-10-14.csv'
           ,nni, delimiter=',' )

result.to_excel(r"Z:\theme\stress\03.Analysis\Analysis_Features\features_kojima_2019-10-14_120s_windows.xlsx")
