# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:58:24 2019

@author: akito
"""

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import frequency_domain as fd
import pyhrv.time_domain as td
import pyhrv.nonlinear as nl
import pyhrv.tools as tools
from opensignalsreader import OpenSignalsReader
from biosppy import signals

# ファイルの関数を呼び出す
import hrv_analysis
import resp_analysis
import eda_analysis




def features(nni,resp,eda,emotion):
    for i,key in enumerate(emotion.keys()):
        segment_bio_report = segments_parameter(nni,resp,eda,emotion[key])
        if i == 0:
            df = pd.DataFrame([], columns=segment_bio_report.keys())
        df =  pd.concat([df, pd.DataFrame(segment_bio_report , index=[key])])

    return df

def segments_parameter(_nni,_resp,_eda,_section):
    results = {}
    # 心拍変動をセクションごとに分ける
    tmStamps = np.cumsum(_nni)*0.001 #in seconds 
    nni_item =  _nni[(tmStamps>=_section[0]) & (tmStamps<=_section[1])]
    nni_features = hrv_analysis.parameter(nni_item)
    
    # 呼吸をセクションごとに分ける
    resp_item =  _resp[(_resp>=_section[0]) & (_resp<=_section[1])]
    resp_features = resp_analysis.RESP_FEATURES(resp_item)

    # 皮膚コンダクタンスをセクションごとに分ける
    eda_features = eda_analysis.EDA_FEATURES(_eda,_section)

    results.update(**nni_features,**resp_features,**eda_features)
    return results



path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-23\teraki\opensignals_dev_2019-10-23_16-59-10.txt"
emotion = {'Neutral1':[0,300]  ,'Stress':[300,600]
          ,'Neutral2':[600,900] ,'Ammusement':[900,1200]}
arc = OpenSignalsReader(path)
signal, rpeaks = signals.ecg.ecg(signal=arc.signal('ECG') , sampling_rate=1000.0, show=False)[1:3]
nni_data = tools.nn_intervals(rpeaks.tolist())

resp_data = signals.resp.resp(signal=arc.signal('RESP'), sampling_rate=1000.0,show=False)['resp_rate_ts']

eda_data = signals.eda.eda(arc.signal('EDA'),show=False)[2:5]
df = features(nni_data,
              resp_data,
              eda_data,
              emotion)

df.to_excel(r"C:\Users\akito\Desktop\bio_features_teraki_2019-10-23_16-59-10.xlsx")
