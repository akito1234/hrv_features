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
        print(key+"... done")
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
def biosignal_summary(path_list,emotion=None):
    for i,path in enumerate(path_list):
        print(path + ' ....start')

        # ファイル名およぶフォルダ名を取得
        dict = os.path.dirname(path)
        fname = os.path.splitext(os.path.basename(path))[0]

        # pathから名前と日付に変換する
        user = dict.split("\\")[-1]
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
            df_summary = pd.DataFrame([], columns=df.columns)

        # ファイルを結合
        df_summary = pd.concat([df_summary,df],ignore_index=True)

    return df_summary



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
    path_list = [
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\kaneko\opensignals_device1_2019-12-11_15-18-00.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\kaneko\opensignals_device1_2019-12-11_16-17-56.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\kishida\opensignals_device3_2019-12-11_13-46-45.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\tohma\opensignals_201808080162_2019-12-11_15-22-53.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\tohma\opensignals_device2_2019-12-11_17-00-24.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\kishida\opensignals_201808080163_2019-12-12_17-50-27.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\kishida\opensignals_201808080163_2019-12-12_18-17-30.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\kojima\opensignals_device3_2019-12-12_13-42-47.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\kojima\opensignals_device3_2019-12-12_14-40-14.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shibata\opensignals_201808080162_2019-12-12_16-00-17.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shibata\opensignals_201808080162_2019-12-12_17-18-11.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shibata\opensignals_201808080162_2019-12-12_19-20-02.txt"#,
        #
        #
        #
        r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_device2_2019-12-12_11-15-27.txt"#,
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_device2_2019-12-12_13-19-00.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_201808080162_2019-12-12_14-57-19.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\tohma\opensignals_device3_2019-12-12_16-46-59.txt",
        #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\tohma\opensignals_201806130003_2019-12-12_19-36-46.txt"
        ]
 
    emotion = {'Neutral1':[0,300],
               'Stress':[300,600],
               'Neutral2':[600,900],
               'Amusement':[900,1250]
               }



    df = biosignal_summary(path_list,emotion)
    df.to_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_2019-12-12_4.xlsx")
    print(df)
    ## セクションを設定する
    #path_list2 = [r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-11\kishida\opensignals_dev_2019-10-11_17-06-10.txt",
    #              r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-11\tohma\opensignals_dev_2019-10-11_17-29-23.txt",
    #              r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-14\kishida\opensignals_dev_2019-10-14_11-40-05.txt",
    #              r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-14\tohma\opensignals_dev_2019-10-14_12-09-33.txt"]
    #emotion2 = {'Neutral1':[120,300],
    #            'Stress':[300,480],
    #            'Neutral2':[720,900]}
    #df_2 = biosignal_summary(path_list2,emotion2)
    #df_2.to_excel(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_dataset_3min_df2.xlsx")


    #df_summary = pd.DataFrame([], columns=df_1.columns)
    #df_summary = pd.concat([df_1, df_2],ignore_index=True)

    #df_summary.to_excel(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_dataset_3min_duration.xlsx")



