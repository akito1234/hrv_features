
# -*- coding: utf-8 -*-
"""
Created 2019.10.12 17:00

@author: akito
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import local package
import hrv_analysis as hf

## RRIから特徴量を算出する
nni= np.loadtxt(r"C:\Users\akito\Desktop\shibata_ecg.csv",delimiter=",")
result = hf.segumentation_freq_features(nni,sample_time=300,time_step=30)

## 結果をエクスポート
result.to_excel(r"C:\Users\akito\Desktop\features_shibata_300s_30s.xlsx")
