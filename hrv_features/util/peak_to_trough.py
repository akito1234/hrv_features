# import package
import numpy as np
import pandas as pd
import seaborn as sns
from biosppy import signals
import matplotlib.pyplot as plt
import pyhrv.tools as tools
# import local
import eda_analysis
import resp_analysis
import pyhrv.frequency_domain as fd
from opensignalsreader import OpenSignalsReader
import biosignal_plot
path = r"C:\Users\akito\Desktop\test.txt"

arc = OpenSignalsReader(path)

# 心拍データからピークを取り出す
ecg_result = signals.ecg.ecg(signal= arc.signal(['ECG']) , sampling_rate=1000.0, show=False)

# 呼吸周波数を取り出す
resp_result = resp_analysis.resp(arc.signal('RESP'),show=False)

# 描画設定
fig,axes = plt.subplots(2,1,sharex=True,figsize = (16,9) )
axes[0].set_title(path)


# Compute NNI series
nni = tools.nn_intervals(ecg_result['rpeaks'].tolist())
filtered = biosignal_plot.detrend(nni,500)

# 心拍変動の描画    
axes[0].plot(ecg_result['heart_rate_ts'].tolist(),filtered,'b')
axes[0].set_ylabel("HR[bpm]")

# 呼吸の描画    
axes[1].plot(resp_result['ts'],resp_result['filtered'])
for ins,exp in zip(resp_result['inspiration'], resp_result['expiration']):
    axes[0].axvline(ins*0.001,color= 'b')
    axes[0].axvline(exp*0.001,color= 'r')
plt.show()