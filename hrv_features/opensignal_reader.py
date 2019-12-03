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


path = r"Z:\00_個人用\東間\01.theme\Bitalino\ECG_test\opensignals_device3_2019-12-02_14-53-32.txt"

# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)

# peak detection
signal_breast, rpeaks_breast = signals.ecg.ecg(signal= arc.signal("ECG"), sampling_rate=1000.0, show=False)[1:3]
signal_leg, rpeaks_leg = signals.ecg.ecg(signal=arc.signal("ECG1") , sampling_rate=1000.0, show=False)[1:3]

# peaks to nni
nni_breast = tools.nn_intervals(rpeaks_breast.tolist())
nni_leg = tools.nn_intervals(rpeaks_leg.tolist())

fig,axes = plt.subplots(3,1,sharex=True)
# plot nni
axes[0].plot(rpeaks_breast[1:]*0.001,nni_breast)
axes[0].plot(rpeaks_leg[1:]*0.001,nni_leg)

# plot error time
axes[1].plot(rpeaks_leg[1:]*0.001,nni_breast-nni_leg)

# plot raw ecg and peaks
axes[2].plot(arc.t,arc.signal("ECG1"))
for (peak_breast,peak_leg) in zip(rpeaks_breast,rpeaks_leg):
    axes[2].axvline(peak_breast*0.001, color="r")
    axes[2].axvline(peak_leg*0.001, color="b")

plt.legend()
plt.show()


# RRIから特徴量を算出する
#result = hf.segumentation_features(nni,sample_time=120,time_step=30)

# 結果をエクスポート
np.savetxt(r"Z:\00_個人用\東間\01.theme\Bitalino\ECG_test\RRI_ecg_breast.csv"
           ,nni_breast, delimiter=',' )
np.savetxt(r"Z:\00_個人用\東間\01.theme\Bitalino\ECG_test\RRI_ecg_leg.csv"
           ,nni_leg, delimiter=',' )
#result.to_excel(r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\features_shizuya_2019-10-23_120s_windows.xlsx")
