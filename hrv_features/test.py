# -*- coding: utf-8 -*-
"""
Created 2019.10.12 17:00

@author: akito
"""

# Import OpenSignalsReader
from opensignalsreader import OpenSignalsReader
import pyhrv.tools as tools
from biosppy import signals
import matplotlib.pyplot as plt
import numpy as np

#path = r"Z:\theme\mental_stress\02.BiometricData\2019-11-19\shibata\opensignals_201806130003_2019-11-19_14-34-44.txt"

## Read OpenSignals file and plot all signals
#arc = OpenSignalsReader(path).plot()
#plt.show()
## 計測開始時刻の抽出
#sign_button = np.diff(np.sign(arc.signal("RAW")))
#onsets     = np.nonzero( sign_button<0 )[0]
#recoveries = np.nonzero( sign_button>0)[0]
#duration = recoveries - onsets
## 開始時刻の定義
#start = onsets[np.argmax(duration)]* 0.001
#print("start_time = {}".format(start))
#filter_ecg = arc.signal('ECG')[arc.t > start]
## 心拍データからピークを取り出す
#signal, rpeaks = signals.ecg.ecg(signal=filter_ecg , sampling_rate=1000.0, show=False)[1:3]

## 結果をエクスポート
##np.savetxt(r"Z:\theme\rppg_filter_test\rri\ecg\RRI_nofiltered_nolight.csv"
##           ,rpeaks, delimiter=',' )



rri_ecg = np.loadtxt(r"Z:\00_個人用\東間\01.theme\Bitalino\ECG_test\RRI_ecg_breast.csv",delimiter=",")[0:200]
ts_ecg = np.cumsum(rri_ecg)
ts_ecg -= ts_ecg[0]

rri_ppg = np.loadtxt(r"Z:\00_個人用\東間\01.theme\Bitalino\ECG_test\RRI_ppg.csv",delimiter=",")[0:200]
ts_ppg = np.cumsum(rri_ppg)
ts_ppg -= ts_ppg[0]

fig,axes = plt.subplots(2,1)
axes[0].plot(ts_ecg,rri_ecg,"r",label="ecg")
axes[0].plot(ts_ppg,rri_ppg,"b",label="ppg")
axes[1].plot(rri_ecg-rri_ppg)
plt.show()