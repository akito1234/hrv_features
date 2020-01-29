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
from datetime import datetime as dt
from biosinal_summary import biosignal_time_summary


path = r"Z:\00_個人用\柴田\20200114実験データ\実験1\opensignals_201808080162_2020-01-14_14-32-28_teraki_30.txt"
skip= 30
outpath = r"Z:\00_個人用\柴田\20200114実験データ\実験1\RRI\rri_201808080162_2020-01-14_14-32-28_teraki.csv"


#record_path = r"Z:\theme\robot_communication\03_LogData\2020-01-28\shibata\robot_communication_2020_01_28__18_05_32.csv"
#outpath = r"Z:\theme\robot_communication\04_Analysis\Analysis_TimeVaries\features_kaneko_2020-01-28.xlsx"

### Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)
#time_start = dt.strptime(arc.info["date"] +" "+arc.info["time"], '%Y-%m-%d %H:%M:%S.%f')
#time_record = pd.read_csv(record_path,index_col =0)

## 時間のずれを算出
#exp_start = dt.strptime(time_record.loc["Neutral","StartDatetime"], '%Y-%m-%d %H:%M:%S.%f')
#duration = (exp_start - time_start).total_seconds()

#print("bitalino start : {}".format(time_start))
#print("experimence start: {}".format(exp_start))
#print("shift duration [s]: {}\n".format(duration))

#df = biosignal_time_summary(path,duration = 300, overlap = 30, skip_time = duration)
#df.to_excel(outpath)



# peak detection
signal, rpeaks= signals.ecg.ecg(signal= arc.signal("ECG")[int(skip*1000):], sampling_rate=1000.0, show=False)[1:3]
nni = tools.nn_intervals(rpeaks.tolist())
#signal, rpeaks= signals.ecg.ecg(signal= arc.signal("ECG")[int(duration*1000):], sampling_rate=1000.0, show=True)[1:3]



## RRIから特徴量を算出する
#nni= np.loadtxt(r"C:\Users\akito\Desktop\shibata_ecg.csv",delimiter=",")
#result = hf.segumentation_freq_features(nni,sample_time=300,time_step=30)

## 結果をエクスポート
np.savetxt(outpath ,nni, delimiter=',' )
#result.to_excel(r"C:\Users\akito\Desktop\features_shibata_300s_30s.xlsx")
