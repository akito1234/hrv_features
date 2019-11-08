# -*- coding: utf-8 -*-

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import frequency_domain as fd
import pyhrv.time_domain as td
import pyhrv.nonlinear as nl
import pyhrv.tools as tools
from biosppy import signals

# ファイルの関数を呼び出す
import hrv_analysis
import resp_analysis
import eda_analysis

def features(rri_peaks, resp_peaks, scr_data, emotion):
    for i,key in enumerate(emotion.keys()):
        segment_bio_report = segments_parameter(rri_peaks, 
                                                resp_peaks, 
                                                scr_data,
                                                emotion[key])
        if i == 0:
            df = pd.DataFrame([], columns=segment_bio_report.keys())
        df =  pd.concat([df, pd.DataFrame(segment_bio_report , index=[key])])

    return df

def segments_parameter(_rri_peaks,_resp_peaks,_scr_data,_section):
    results = {}
    # 呼吸，心拍，皮膚抵抗をセクションごとに分割する
    ecg_item  = _rri_peaks[(_rri_peaks>=_section[0]*1000) & (_rri_peaks<=_section[1]*1000)]
    
    resp_item = _resp_peaks[(_resp_peaks>=_section[0]) & (_resp_peaks<=_section[1])]
    
    ts_filter = (_scr_data['ts']>=_section[0]) & (_scr_data['ts']<=_section[1])
    scr_item  = {'sc':_scr_data['sc'][ts_filter],
                 'pathicData':_scr_data['pathicData'][ts_filter],
                 'tonicData':_scr_data['tonicData'][ts_filter]}

    # 心拍変動をセクションごとに分ける
    ecg_features = hrv_analysis.parameter(ecg_item)
    
    # 呼吸をセクションごとに分ける
    resp_features = resp_analysis.resp_features(resp_item)

    # 皮膚コンダクタンスをセクションごとに分ける
    eda_features = eda_analysis.scr_features(scr_item)

    results.update(**ecg_features,**resp_features,**eda_features)
    return results

if __name__ == '__main__':
    from opensignalsreader import OpenSignalsReader
    path = r"C:\Users\akito\Desktop\test.txt"
    
    # セクションを設定する
    #emotion = {'Neutral1':[0,300]  ,'Stress':[300,600]
    #          ,'Neutral2':[600,900] ,'Ammusement':[900,1200]}
    emotion = {'test':[0,100]}
    arc = OpenSignalsReader(path)
    
    # 心拍変動
    rri_peaks = signals.ecg.ecg(arc.signal('ECG') , sampling_rate=1000.0, show=False)['rpeaks']
    # 呼吸変動
    resp_peaks = resp_analysis.resp(arc.signal('RESP'), sampling_rate=1000.0,show=False)['peaks']  
    # 皮膚コンダクタンス
    scr_data = eda_analysis.scr(arc.signal('EDA'), sampling_rate=1000.0, downsamp = 4)

    df = features(rri_peaks,
                  resp_peaks,
                  scr_data,
                  emotion)
    df.to_excel(r"C:\Users\akito\Desktop\bio_features_teraki_2019-10-23_16-59-10.xlsx")
