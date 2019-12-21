# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:58:24 2019

@author: akito
"""

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyhrv.time_domain as td
import pyhrv.nonlinear as nl
import pyhrv.tools as tools
import datetime
from opensignalsreader import OpenSignalsReader
from biosppy import signals

# local packages
import custom_frequency_domain as fd
from util import detrending

def features(nni,emotion):
    for i,key in enumerate(emotion.keys()):
        segment_hrv_report = segments_parameter(nni,emotion[key])
           
        if i == 0:
            df = pd.DataFrame([], columns=segment_hrv_report.keys())
        df =  pd.concat([df, pd.DataFrame(segment_hrv_report , index=[key])])

    return df


def segments_parameter(_nni,_section):
    tmStamps = np.cumsum(_nni)*0.001 #in seconds 
    nni_item =  _nni[(tmStamps>=_section[0]) & (tmStamps<=_section[1])]
    results = parameter(nni_item)
    return results


#---------------------------------------------
#心拍変動からパラメータを算出する
#---------------------------------------------
def parameter(rpeaks):
    #返り値を初期化
    results = {}
    nni = tools.nn_intervals(rpeaks=rpeaks.tolist())

    # -----------------周波数解析-------------------#
    # welch method
    welch_freqDomain = fd.welch_psd(nni,nfft=2 ** 10, show=False)
    for key in welch_freqDomain.keys():
        results[key] = welch_freqDomain[key]
    # ar method
    ar_freqDomain = fd.ar_psd(nni,nfft=2 ** 10, show=False)
    for key in ar_freqDomain.keys():
        results[key] = ar_freqDomain[key]
    # lomb method
    lomb_freqDomain = fd.lomb_psd(nni,nfft=2 ** 8, show=False)
    for key in lomb_freqDomain.keys():
        results[key] = lomb_freqDomain[key]

    # -----------------時系列解析-------------------#
    timeDomain = td.time_domain(nni)

    for key in timeDomain.keys():
        results[key] = timeDomain[key]

    # -----------------非線形解析-------------------#
    nonlinearDomain = nl.nonlinear(nni=nni)

    for key in nonlinearDomain.keys():
        results[key] = nonlinearDomain[key]

    #不要なパラメータの削除
    del_keylist = ['nni_histogram','poincare_plot','dfa_plot','fft_bands','ar_bands','lomb_bands','sdnn_index','sdann']
    for del_keys in del_keylist:
        del results[del_keys]

    #パラメータの変換
    results = modify_tuple_to_float(results)

    return results

#---------------------------------------------
#パラメータリストをfloat型に変換する
#---------------------------------------------
def modify_tuple_to_float(parameter_list):
    #-------------------------------------------------
    #・パラメータがlist型の場合
    #ar_rel : [A B C]を
    #ar_rel_vlf : A, ar_rel_lf : B, ar_rel_vhf : C
    #に変換
    #※パラメータがfloat型の場合はそのまんま
    #-------------------------------------------------
    results = {}
    for key in parameter_list.keys():
        if type(parameter_list[key]) == tuple:
            #変更するラベル名を定義
            if len(parameter_list[key]) == 2:  labels = ['lf','hf']
            elif len(parameter_list[key]) == 3:labels = ['vlf','lf','hf']

            for (list_item,label) in zip(parameter_list[key],labels):
                #dict名をkey + vlf or lf or hfに変更する
                results[key + '_' + label] = list_item
        else:
            if key == 'Filename':
                #ファイル名からに日にちとユーザ名を取得
                results['date'], results['user'] = fileName_Info(parameter_list[key])

            results[key] = parameter_list[key]
    return results




#-----------------------------------------------------------------------
# 一定時間ごとの特徴量
#-----------------------------------------------------------------------
def segumentation_features(nni,sample_time=300,time_step=30):
    # 一定時間ごとのRRIを取得する
    rri_list = segumentation_rri(nni,sample_time,time_step)

    for i,rri_item in enumerate(rri_list):
        dict_parameters = parameter(rri_item)
        time = sample_time + time_step * i;

        if i == 0:
            df = pd.DataFrame([], columns=dict_parameters.keys())

        df =  pd.concat([df, pd.DataFrame(dict_parameters, index=[time])])

    return df

#-----------------------------------------------------------------------
# 一定時間ごとのRRI
#-----------------------------------------------------------------------
def segumentation_rri(nni,sample_time=300,time_step=30):
    time_inter = sample_time * 1000 # 標本数
    time_step = time_step * 1000 # ステップ時間

    # タイムスタンプの算出
    tmStamps = np.cumsum(nni) #in seconds 
    tmStamps -= tmStamps[0]

    # timeStepごとにデータを取り出す
    start_time = np.arange(start = 0,
                           stop  = tmStamps[-1]-time_inter,
                           step  = time_step)
    list = []
    for i,start in enumerate(start_time):
        end = start + time_inter
        nni_item = nni[(tmStamps>= start) & (tmStamps <=end)];
        list.append(nni_item)
        
    return list


#-----------------------------------------------------------------------
# 一定時間ごとの特徴量　Freq Analysis
# 時間短縮のため作成，後で消しておく
#-----------------------------------------------------------------------
def segumentation_freq_features(nni,sample_time=300,time_step=30):
    # 一定時間ごとのRRIを取得する
    rri_list = segumentation_rri(nni,sample_time,time_step)

    for i,rri_item in enumerate(rri_list):
        dict_parameters = freqparameter(rri_item)
        time = sample_time + time_step * i;
        if i == 0:
            df = pd.DataFrame([], columns=dict_parameters.keys())

        df =  pd.concat([df, pd.DataFrame(dict_parameters, index=[time])])

    return df

#---------------------------------------------
#心拍変動からパラメータを算出する　Freq Analysis
# 時間短縮のため作成，後で消しておく
#---------------------------------------------
def freqparameter(nni):
    #返り値を初期化
    results = {}

    # -----------------周波数解析-------------------#
    welch_freqDomain = fd.welch_psd(nni, nfft=2 ** 10, show=False)
    ar_freqDomain = fd.ar_psd(nni, nfft=2 ** 10, show=False)
    lomb_freqDomain = fd.lomb_psd(nni, nfft=2 ** 8, show=False)

    for key in welch_freqDomain.keys():
        results[key] = welch_freqDomain[key]

    for key in ar_freqDomain.keys():
        results[key] = ar_freqDomain[key]

    for key in lomb_freqDomain.keys():
        results[key] = lomb_freqDomain[key]

    #不要なパラメータの削除
    del_keylist = ['fft_bands','ar_bands','lomb_bands']
    for del_keys in del_keylist:
        del results[del_keys]

    #パラメータの変換
    results = modify_tuple_to_float(results)

    return results

#---------------------------------------------
# Neutrl状態から差分を算出する
#---------------------------------------------
#def biosignal_features(df, emotion,keywords = None):
#    for i,key in enumerate(emotion.keys()):
#        # セグメント内での特徴量算出
#        segment_bio_report = {}
#        if keywords is not None:
#            segment_bio_report.update(keywords)

#        #bio_parameter = segments_parameter(rri_peaks, 
#        #                                   resp_peaks, 
#        #                                   scr_data,
#        #                                   emotion[key])

#        print("----------------"+key+"--------------------")
        
#        segment_bio_report.update({'emotion':key})
#        segment_bio_report.update(bio_parameter)

#        if i == 0:
#            df = pd.DataFrame([], columns=segment_bio_report.keys())

#        df =  pd.concat([df, pd.DataFrame(segment_bio_report , index=[key])])
#    return df


def neutral_detrend(df, emotion, keywords= None,base = "Neutral1"):
    # ベースラインを算出
    df_base = df[ (df.index > emotion[base][0]) & (df.index <= emotion[base][1]) ].mean()
    # 帰り値の初期化
    result = pd.DataFrame()
    
    for i,key in enumerate(emotion.keys()):
        # セクションごとの特徴量
        df_item = df[(df.index > emotion[key][0]) & (df.index <= emotion[key][1])]
        df_item = df_item - df_base.values
        
        # 各特徴量でゼロ以上となる点の平均値を算出する
        df_bool = (df_item >= 0)
        series_abs = (df_item * df_bool).mean()
        
        if keywords is not None:
            series_abs = pd.concat([series_abs, pd.Series(keywords)])
        
        # 感情ラベルを追加
        series_abs['emotion'] = key
        result = pd.concat([result, series_abs], axis=1)
    return result

def freqDomain_base_analysis(pathlist,emotion,base="Neutral1"):
    for i,path in enumerate(pathlist):
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
        rpeaks = signals.ecg.ecg(arc.signal('ECG') , sampling_rate=1000.0, show=False)['rpeaks']
        rri = tools.nn_intervals(rpeaks.tolist())
        df_segumentation = segumentation_freq_features(rri,sample_time=60,time_step=30)

        # ニュートラルの平均よりも高い値の平均を算出する
        result = neutral_detrend(df_segumentation,emotion, keywords=keyword, base = "Neutral1")

        if i == 0:
            df_summary = pd.DataFrame([], columns=result.index)

        # ファイルを結合
        df_summary = pd.concat([df_summary,result.T],ignore_index=True)
        df_summary.to_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\datasets_freq.xlsx")
    return df_summary
    pass


if __name__ == '__main__':
    import os
    pathlist= [
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-21\tohma\opensignals_201806130003_2019-10-21_15-16-48.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-22\kishida\opensignals_dev_2019-10-22_13-54-50.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-23\shizuya\opensignals_dev_2019-10-23_14-09-52.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-23\teraki\opensignals_dev_2019-10-23_16-59-10.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-10-28\shibata\opensignals_dev_2019-10-28_13-50-02.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-19\shizuya\opensignals_201806130003_2019-11-19_16-38-07.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-19\takase\opensignals_device3_2019-11-19_16-38-30.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-20\takase\opensignals_201808080162_2019-11-20_13-39-19.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-20\takase\opensignals_device1_2019-11-20_15-37-40.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-20\teraki\opensignals_device1_2019-11-20_14-40-26.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-20\teraki\opensignals_device2_2019-11-20_13-40-49.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kaneko\opensignals_201806130003_2019-11-21_14-58-59.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kaneko\opensignals_201806130003_2019-11-21_16-01-49.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kaneko\opensignals_201806130003_2019-11-21_16-44-45.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kishida\opensignals_device3_2019-11-21_14-56-56.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kishida\opensignals_device3_2019-11-21_16-00-52.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kojima\opensignals_device2_2019-11-21_14-59-07.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\kojima\opensignals_device2_2019-11-21_16-06-09.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\shibata\opensignals_device2_2019-11-21_16-51-13.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-11-21\tohma\opensignals_device3_2019-11-21_16-54-54.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\moriyama\opensignals_201808080163_2019-12-05_14-44-44.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\moriyama\opensignals_201808080163_2019-12-05_16-03-59.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\otsuka\opensignals_device2_2019-12-05_14-32-48.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\otsuka\opensignals_device2_2019-12-05_16-01-19.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\tozyo\opensignals_device1_2019-12-05_14-45-10.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\tozyo\opensignals_device1_2019-12-05_16-02-47.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\moriyama\opensignals_201808080162_2019-12-10_15-09-14.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\moriyama\opensignals_device2_2019-12-10_16-40-48.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\otsuka\opensignals_device3_2019-12-10_14-52-41.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\otsuka\opensignals_device3_2019-12-10_16-08-42.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\otsuka\opensignals_device3_2019-12-10_16-39-25.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\tozyo\opensignals_201806130003_2019-12-10_14-54-49.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\tozyo\opensignals_201806130003_2019-12-10_16-10-31.txt",
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-10\tozyo\opensignals_201806130003_2019-12-10_16-37-56.txt",
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
                 #r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shibata\opensignals_201808080162_2019-12-12_19-20-02.txt",
#r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_device2_2019-12-12_11-15-27.txt",
#r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_device2_2019-12-12_13-19-00.txt",
#r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\shizuya\opensignals_201808080162_2019-12-12_14-57-19.txt",
                 r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\tohma\opensignals_device3_2019-12-12_16-46-59.txt",
                 r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-12\tohma\opensignals_201806130003_2019-12-12_19-36-46.txt"
    ]
    #rri = np.loadtxt(path,delimiter=',')
    #A = segumentation_freq_features(rri,sample_time=60,time_step=15)
    #fname = os.path.splitext(os.path.basename(path))[0]

    emotion = {'Neutral1':[0,300],
               'Stress':[300,600],
               'Neutral2':[600,900],
               'Amusement':[900,1200]}
    A = freqDomain_base_analysis(pathlist,emotion,base="Neutral1")
    print(A)
    A.to_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\datasets_freq.xlsx")
    
    pass
