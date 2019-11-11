# -*- coding: utf-8 -*-

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from biosppy import signals

# ファイルの関数を呼び出す
import hrv_analysis
import resp_analysis
import eda_analysis


def features(rri_peaks, resp_peaks, scr_data, emotion,keywords = None):
    for i,key in enumerate(emotion.keys()):
        # セグメント内での特徴量算出
        segment_bio_report = {}
        if keywords is not None:
            segment_bio_report.update(keywords)
            ## 被験者の名前を取得se            
            #a = os.path.split(path).split('/')
            #subject = os.path.split(path)[-2]

            ## タイムラインを取得
            #date = os.path.basename(path).split('_')[2]
            #time = os.path.basename(path).split('_')[3][:-3]


        bio_parameter = segments_parameter(rri_peaks, 
                                           resp_peaks, 
                                           scr_data,
                                           emotion[key])

        segment_bio_report.update({'emotion':key})
        segment_bio_report.update(bio_parameter)

        if i == 0:
            df = pd.DataFrame([], columns=segment_bio_report.keys())
        section_df = pd.DataFrame(segment_bio_report.values(), index=segment_bio_report.keys()).T
        df =  pd.concat([df, section_df])

    return df

def segments_parameter(_rri_peaks,_resp_peaks,_scr_data,_section):
    results = {}
    # 呼吸，心拍，皮膚抵抗をセクションごとに分割する
    ecg_item  = _rri_peaks[(_rri_peaks>=_section[0]*1000) & (_rri_peaks<=_section[1]*1000)]
    
    resp_item = _resp_peaks[(_resp_peaks>=_section[0]*1000) & (_resp_peaks<=_section[1]*1000)]
    
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
    #path_list = [r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-21\tohma\opensignals_201806130003_2019-10-21_15-16-48.txt",
    #             r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-22\kishida\opensignals_dev_2019-10-22_13-54-50.txt",
    #             r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-23\shizuya\opensignals_dev_2019-10-23_14-09-52.txt",
    #             r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-23\teraki\opensignals_dev_2019-10-23_16-59-10.txt",
    #             r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-29\shibata\opensignals_dev_2019-10-28_13-50-02.txt"
    #             ]

    path_list = [r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-11\kishida\opensignals_dev_2019-10-11_17-06-10.txt",
                 r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-11\tohma\opensignals_dev_2019-10-11_17-29-23.txt",
                 r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-14\kishida\opensignals_dev_2019-10-14_11-40-05.txt",
                 r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-14\tohma\opensignals_dev_2019-10-14_12-09-33.txt"]

    # セクションを設定する
    emotion = {'Neutral1':[0,300],
               'Stress':[300,600],
               'Neutral2':[600,900],
               #'Ammusement':[900,1200]
               }


    for i,path in enumerate(path_list):
        arc = OpenSignalsReader(path)
        print(path + 'detected')

        # 心拍変動
        rri_peaks = signals.ecg.ecg(arc.signal('ECG') , sampling_rate=1000.0, show=False)['rpeaks']
        # 呼吸変動
        resp_peaks = resp_analysis.resp(arc.signal('RESP'), sampling_rate=1000.0,show=False)['peaks']  
        # 皮膚コンダクタンス
        scr_data = eda_analysis.scr(arc.signal('EDA'), sampling_rate=1000.0, downsamp = 4)

        # キーワードを設定
        keyword = {'id':i ,
                   'path_name':path}
        df = pd.DataFrame([])
        df = features(rri_peaks,
                      resp_peaks,
                      scr_data,
                      emotion,
                      keyword)
        if i == 0:
            df_summary = pd.DataFrame([], columns=df.columns)

        # ファイルを結合
        df_summary = pd.concat([df_summary,df],ignore_index=True)

    df_summary.to_excel(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets2.xlsx")
