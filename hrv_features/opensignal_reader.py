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


path = r"Z:\theme\robot_communication\04_Analysis\Analysis_Biosignal\rri_shibata_2020-02-05.csv"
outpath = r"Z:\theme\robot_communication\04_Analysis\Analysis_Biosignal.xlsx"

# RRIから特徴量を算出する
nni= np.loadtxt(path,delimiter=",")
result = hf.segumentation_features(nni,sample_time=120,time_step=30)

# 結果をエクスポート
result.to_excel(outpath)



#text = "shibata_2020-02-05"


#outpath = r"Z:\theme\robot_communication\04_Analysis\Analysis_TimeVaries\features_{}.xlsx".format(text)
#rripath = r"Z:\theme\robot_communication\04_Analysis\Analysis_Biosignal\rri_{}.csv".format(text)

### Read OpenSignals file and plot all signals
#arc = OpenSignalsReader(path)
#time_start = dt.strptime(arc.info["date"] +" "+arc.info["time"], '%Y-%m-%d %H:%M:%S.%f')
#time_record = pd.read_csv(record_path,index_col =0)




## 時間のずれを算出
#exp_start = dt.strptime(time_record.loc["Neutral","StartDatetime"], '%Y-%m-%d %H:%M:%S.%f')
#duration = (exp_start - time_start).total_seconds()

#print("bitalino start : {}".format(time_start))
#print("experimence start: {}".format(exp_start))
#print("shift duration [s]: {}\n".format(duration))


# peak detection
#signal, rpeaks= signals.ecg.ecg(signal= arc.signal("ECG")[int(duration*1000):], sampling_rate=1000.0, show=False)[1:3]
#nni = tools.nn_intervals(rpeaks.tolist())
#np.savetxt(rripath ,nni, delimiter=',' )

#df = biosignal_time_summary(path,duration = 300, overlap = 30, skip_time = duration)
#df.to_excel(outpath)






