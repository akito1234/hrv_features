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


path = r"D:\disgust_contentment\20190802\log\opensignals_device1_device2_device3_2019-08-02_17-15-43.txt"
fig,axes = plt.subplots(3,1)

kind = "EDA"

# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path,multidevice=0)
ecg = arc.signal(kind)
axes[0].plot(ecg)

arc = OpenSignalsReader(path,multidevice=1)
ecg = arc.signal(kind)
axes[1].plot(ecg)

arc = OpenSignalsReader(path,multidevice=2)
ecg = arc.signal(kind)
axes[2].plot(ecg)

plt.show()

## 心拍データからピークを取り出す
#signal, rpeaks = signals.ecg.ecg(signal= ecg , sampling_rate=1000.0, show=False)[1:3]
#nni = tools.nn_intervals(rpeaks.tolist())

# RRIから特徴量を算出する
#result = hf.segumentation_features(nni,sample_time=120,time_step=30)

# 結果をエクスポート
#np.savetxt(r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\ECG\RRI_takase_2019-11-19_16-38-30.csv"
#           ,nni, delimiter=',' )

#result.to_excel(r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\features_shizuya_2019-10-23_120s_windows.xlsx")
