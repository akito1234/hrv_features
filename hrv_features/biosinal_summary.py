# -*- coding: utf-8 -*-

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os
from biosppy import signals

# Import local packages
import hrv_analysis
import resp_analysis
import eda_analysis
from opensignalsreader import OpenSignalsReader

def biosignal_features(rri_peaks,
                       resp_peaks, 
                       scr_data,
                       emotion,keywords = None):
    for i,key in enumerate(emotion.keys()):
        # セグメント内での特徴量算出
        segment_bio_report = {}
        if keywords is not None:
            segment_bio_report.update(keywords)


        bio_parameter = segments_parameter(rri_peaks, 
                                           resp_peaks, 
                                           scr_data,
                                           emotion[key])
        print("{}... done".format(key))
        segment_bio_report.update({'emotion':key})
        segment_bio_report.update(bio_parameter)

        if i == 0:
            df = pd.DataFrame([], columns=segment_bio_report.keys())

        df =  pd.concat([df, pd.DataFrame(segment_bio_report , index=[key])])
    return df

def segments_parameter(_rri_peaks,_resp_peaks,
                       _scr_data,
                       _section):
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

    results.update(**ecg_features,**resp_features, **eda_features)
    return results

# 生体信号から特徴量を算出し，dataframe型にまとめて返す
def biosignal_summary(path_list,emotion=None,output=None):
    for i,path in enumerate(path_list):
        print(path + ' ....start')

        # ファイル名およぶフォルダ名を取得
        dict_path = os.path.dirname(path)
        fname = os.path.splitext(os.path.basename(path))[0]

        # pathから名前と日付に変換する
        user = dict_path.split("\\")[-1]
        day, time = fname.split("_")[-2:]
        date = datetime.datetime.strptime(day+" "+time, '%Y-%m-%d %H-%M-%S')

        # キーワードを設定
        keyword = {'id':i, 'path_name':path,
                   'user':user,'date':date}

        arc = OpenSignalsReader(path)

        # 心拍変動
        rri_peaks = signals.ecg.ecg(arc.signal('ECG') , sampling_rate=1000.0, show=False)['rpeaks']
        # 呼吸変動
        resp_peaks = resp_analysis.resp(arc.signal('RESP'), sampling_rate=1000.0,show=False)['peaks']  
        # 皮膚コンダクタンス
        scr_data = eda_analysis.scr(arc.signal('EDA'), sampling_rate=1000.0, downsamp = 4)

        df = pd.DataFrame([])
        df = biosignal_features(rri_peaks,
                      resp_peaks,
                      scr_data,
                      emotion,
                      keyword)

        if i == 0:
            if output == None:
                df_summary = pd.DataFrame([], columns=df.columns)
            else:
                df_summary = pd.read_excel(output)

        # ファイルを結合
        df_summary = pd.concat([df_summary,df],ignore_index=True)
        df_summary.to_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_shy.xlsx")
    return df_summary

# 一定時間ごとの特徴量を算出し，dataframe型にまとめて返す
def biosignal_time_summary(path, duration=300,overlap=150):
    # 生体データを取得
    arc = OpenSignalsReader(path)
    
    # 時間変数を作成
    time_ = np.arange(duration, arc.t.max(), overlap)
    section_ = zip((time_ - duration), time_)
    emotion = dict(zip(time_.tolist(), section_))

    # HRV Features
    rri_peaks = signals.ecg.ecg(arc.signal('ECG') , sampling_rate=1000.0, show=False)['rpeaks']
    # RESP Features
    resp_peaks = resp_analysis.resp(arc.signal('RESP'), sampling_rate=1000.0,show=False)['peaks']  
    # EDA Features
    scr_data = eda_analysis.scr(arc.signal('EDA'), sampling_rate=1000.0, downsamp = 4)
    
    df = pd.DataFrame([])
    # 各生体データを時間区間りで算出
    df = biosignal_features(rri_peaks,resp_peaks,scr_data,emotion)
    return df

# 生体信号から特徴量を算出し，dataframe型にまとめて返す
def biosignal_multi_summary(path_list,emotion=None):
    i = 0
    df_summary = None
    for path in path_list:
        print('{} ....start'.format(path))
        for n in range(3):
            arc = OpenSignalsReader(path,multidevice=n)
            print("device= {}".format(n))

            # 心拍変動
            rri_peaks = signals.ecg.ecg(arc.signal('ECG'), show=False)['rpeaks']
            # 呼吸変動
            resp_peaks = resp_analysis.resp(arc.signal('RESP'),show=False)['peaks']  
            # 皮膚コンダクタンス
            scr_data = eda_analysis.scr(arc.signal('EDA'), downsamp = 4)
            # キーワードを設定
            keyword = {'id':i, 'path_name':path}
            
            df = pd.DataFrame([])
            df = biosignal_features(rri_peaks, resp_peaks, scr_data, emotion, keyword)

            if i == 0:
                df_summary = pd.DataFrame([], columns=df.columns)
            # ファイルを結合
            df_summary = pd.concat([df_summary,df],ignore_index=True)
            i+=1
        

    return df_summary



if __name__ == '__main__':
#    path_list = [
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-21\tohma\opensignals_201806130003_2019-10-21_15-16-48.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-22\kishida\opensignals_dev_2019-10-22_13-54-50.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-23\shizuya\opensignals_dev_2019-10-23_14-09-52.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-23\teraki\opensignals_dev_2019-10-23_16-59-10.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-28\shibata\opensignals_dev_2019-10-28_13-50-02.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-19\shizuya\opensignals_201806130003_2019-11-19_16-38-07.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-19\takase\opensignals_device3_2019-11-19_16-38-30.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-20\takase\opensignals_201808080162_2019-11-20_13-39-19.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-20\takase\opensignals_device1_2019-11-20_15-37-40.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-20\teraki\opensignals_device1_2019-11-20_14-40-26.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-20\teraki\opensignals_device2_2019-11-20_13-40-49.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kaneko\opensignals_201806130003_2019-11-21_14-58-59.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kaneko\opensignals_201806130003_2019-11-21_16-01-49.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kaneko\opensignals_201806130003_2019-11-21_16-44-45.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kishida\opensignals_device3_2019-11-21_14-56-56.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kishida\opensignals_device3_2019-11-21_16-00-52.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kojima\opensignals_device2_2019-11-21_14-59-07.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kojima\opensignals_device2_2019-11-21_16-06-09.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\shibata\opensignals_device2_2019-11-21_16-51-13.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\tohma\opensignals_device3_2019-11-21_16-54-54.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\moriyama\opensignals_201808080163_2019-12-05_14-44-44.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\moriyama\opensignals_201808080163_2019-12-05_16-03-59.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\otsuka\opensignals_device2_2019-12-05_14-32-48.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\otsuka\opensignals_device2_2019-12-05_16-01-19.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\tozyo\opensignals_device1_2019-12-05_14-45-10.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\tozyo\opensignals_device1_2019-12-05_16-02-47.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\moriyama\opensignals_201808080162_2019-12-10_15-09-14.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\moriyama\opensignals_device2_2019-12-10_16-40-48.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\otsuka\opensignals_device3_2019-12-10_14-52-41.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\otsuka\opensignals_device3_2019-12-10_16-08-42.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\otsuka\opensignals_device3_2019-12-10_16-39-25.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\tozyo\opensignals_201806130003_2019-12-10_14-54-49.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\tozyo\opensignals_201806130003_2019-12-10_16-10-31.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\tozyo\opensignals_201806130003_2019-12-10_16-37-56.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\kaneko\opensignals_device1_2019-12-11_15-18-00.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\kaneko\opensignals_device1_2019-12-11_16-17-56.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\kishida\opensignals_device3_2019-12-11_13-46-45.txt",
##r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\tohma\opensignals_201808080162_2019-12-11_15-22-53.txt",
##r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\tohma\opensignals_device2_2019-12-11_17-00-24.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\kishida\opensignals_201808080163_2019-12-12_17-50-27.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\kishida\opensignals_201808080163_2019-12-12_18-17-30.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\kojima\opensignals_device3_2019-12-12_13-42-47.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\kojima\opensignals_device3_2019-12-12_14-40-14.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shibata\opensignals_201808080162_2019-12-12_16-00-17.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shibata\opensignals_201808080162_2019-12-12_17-18-11.txt",
#                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shibata\opensignals_201808080162_2019-12-12_19-20-02.txt",
##r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_device2_2019-12-12_11-15-27.txt",
#                 r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_device2_2019-12-12_13-19-00.txt",
#                 r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_201808080162_2019-12-12_14-57-19.txt",
#                 r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\tohma\opensignals_device3_2019-12-12_16-46-59.txt",
#                 r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\tohma\opensignals_201806130003_2019-12-12_19-36-46.txt"
#    ]
    #path_list = [
    #    r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_device2_2019-12-12_11-15-27.txt",
    #    #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_201808080162_2019-12-12_14-57-19.txt",
    #    #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\tohma\opensignals_device3_2019-12-12_16-46-59.txt",
    #    #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\tohma\opensignals_201806130003_2019-12-12_19-36-46.txt"
    #]
    #emotion = {'Neutral1':[0,300],
    #           'Stress':[300,600],
    #           'Neutral2':[600,900],
    #           'Amusement':[900,1220]
    #           }
    #df = biosignal_summary(path_list,emotion)
    #df.to_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_shy.xlsx")
    #print(df)

    df=biosignal_time_summary(r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\tohma\opensignals_201806130003_2019-12-12_19-36-46.txt",
                           duration=300,overlap=30)
    df.to_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_time_Varies_TOHMA.xlsx")

    pass