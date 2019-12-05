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


path = r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\moriyama\opensignals_201808080163_2019-12-05_14-44-44.txt"

# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)

# peak detection
signal, rpeaks= signals.ecg.ecg(signal= arc.signal("ECG"), sampling_rate=1000.0, show=True)[1:3]
# peaks to nni
nni = tools.nn_intervals(rpeaks.tolist())



# RRIから特徴量を算出する
#result = hf.segumentation_features(nni,sample_time=120,time_step=30)

# 結果をエクスポート
np.savetxt(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_BioSignal\ECG\RRI_moriyama_2019-12-05_14-44-44.csv"
           ,nni, delimiter=',' )
#result.to_excel(r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\features_shizuya_2019-10-23_120s_windows.xlsx")
