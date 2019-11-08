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
def parameter(nni):
    # タイムスタンプの算出
    tmStamps = np.cumsum(nni)*0.001 #in seconds 

    #返り値を初期化
    results = {}

    # -----------------周波数解析-------------------#
    ## Define input parameters for the 'welch_psd()' function
    #kwargs_welch = {'nfft': 2**10, 'detrend': True, 'window': 'hann'}
    ## Define input parameters for the 'lomb_psd()' function
    #kwargs_lomb = {'nfft': 2**8}
    ## Define input parameters for the 'ar_psd()' function
    #kwargs_ar = {'nfft': 2**10}
    #freqDomain = fd.welch_psd(nni=nni.astype(int).tolist(),nfft= 2**10, detrend = True, window =  'hann',show=False)
    
    detrending_rri = detrending.detrend(nni, Lambda= 500)
    ts = np.arange(0,len(detrending_rri)*0.25,step=0.25)
    fs= 4
    start= 300
    duration = 300
    freq_parameter = welch_psd(detrending_rri[(ts > start) &(ts <= (start + duration) )],fs = fs, nfft=2 ** 12)

    for key in freqDomain.keys():
        results[key] = freqDomain[key]

    # -----------------時系列解析-------------------#
    timeDomain = td.time_domain(nni)

    for key in timeDomain.keys():
        results[key] = timeDomain[key]

    # -----------------非線形解析-------------------#
    nonlinearDomain = nl.nonlinear(nni=nni)

    for key in nonlinearDomain.keys():
        results[key] = nonlinearDomain[key]

    #不要なパラメータの削除
    del_keylist = ['fft_bands'
                  ,'fft_nfft','fft_window','fft_resampling_frequency','fft_interpolation'
                  ,'fft_plot'
                  #,'lomb_plot','ar_plot','lomb_bands','ar_bands'
                  ,'nni_histogram','poincare_plot','dfa_plot']
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


#---------------------------------------------
# ニュートラル状態を差分する
#---------------------------------------------
def correction_neutral_before(df_neutral,df_emotion):
    correction_df = (df_emotion - df_neutral)
    return df
    pass


#-----------------------------------------------------------------------
# 一定時間ごとのRRI
#-----------------------------------------------------------------------
def segumentation_rri(nni,sample_time=300,time_step=30):
    time_inter = sample_time * 1000 # 標本数
    time_step = time_step * 1000 # ステップ時間

    # タイムスタンプの算出
    tmStamps = np.cumsum(nni) #in seconds 

    # 5sごとにデータを取り出す
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
# 一定時間ごとの特徴量
#-----------------------------------------------------------------------
def segumentation_features(nni,sample_time=300,time_step=30):
    rri_list = segumentation_rri(nni,sample_time,time_step)

    for i,rri_item in enumerate(rri_list):
        dict_parameters = parameter(rri_item)
        time = sample_time + time_step * i;

        if i == 0:
            df = pd.DataFrame([], columns=dict_parameters.keys())

        df =  pd.concat([df, pd.DataFrame(dict_parameters, index=[time])])

    return df


if __name__ == '__main__':
    path= r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_BioSignal\RRI_kishida_kubios_2019-10-11.csv"
    rri = np.loadtxt(path,delimiter=',')
    A = segumentation_features(rri,sample_time=120,time_step=30)
    A.to_excel(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\features_kishida_2019-10-11_120s_windows_pre.xlsx")
    pass
    ##感情ラベルの時間を定義する
    #emotion = {'Neutral1':[600,900]  ,'Contentment':[900,1200]
    #          ,'Neutral2':[1380,1680] ,'Disgust':[1680,1980]
    #          }
    #nni = np.loadtxt(path,delimiter=',')
    #df = features(nni,emotion)

    ##ニュートラル補正
    #df_2 = (df.loc[['Contentment','Disgust']] - df.loc[['Neutral1','Neutral2']].values)
    ##df_Contentment = df.loc[['Contentment']] - df.loc[['Neutral2']].values

    #df_2.to_excel(r"Z:\theme\hrv_daily_fluctuation\05_Analysis\Analysis_Features\Analysis_Features_Labels\neutral_first_second\tohma_2019-09-26_emotion.xlsx")

    pass
