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


def features(nni,emotion):
    for i,key in enumerate(emotion.keys()):
        segment_hrv_report = segments_parameter(nni, emotion[key] , type=key)

        
        if i == 0:
            df = pd.DataFrame([], columns=segment_hrv_report.keys())
        df =  pd.concat([df, pd.DataFrame(segment_hrv_report , index=[i])])

    return df


def segments_parameter(_nni,_section, type=''):
    tmStamps = np.cumsum(_nni)*0.001 #in seconds 
    nni_item =  _nni[(tmStamps>=_section[0]) & (tmStamps<=_section[1])]
    results = parameter(nni_item,type=type)
    return results


#---------------------------------------------
#心拍変動からパラメータを算出する
#---------------------------------------------
def parameter(nni,type='Neutral'):
    # タイムスタンプの算出
    tmStamps = np.cumsum(nni)*0.001 #in seconds 

    #返り値を初期化
    results = {'emotion' : type }

    # -----------------周波数解析-------------------#
    # Define input parameters for the 'welch_psd()' function
    kwargs_welch = {'nfft': 2**10, 'detrend': True, 'window': 'hann'}
    # Define input parameters for the 'lomb_psd()' function
    kwargs_lomb = {'nfft': 2**8}
    # Define input parameters for the 'ar_psd()' function
    kwargs_ar = {'nfft': 2**10}
    freqDomain = fd.frequency_domain(nni=nni.astype(int).tolist()
                                          ,show=False
                                          ,kwargs_welch=kwargs_welch
                                          ,kwargs_lomb=kwargs_lomb
                                          ,kwargs_ar=kwargs_ar)
    for key in freqDomain.keys():
        results[key] = freqDomain[key]

    # -----------------時系列解析-------------------#
    timeDomain = td.time_domain(nni.astype(int).tolist())

    for key in timeDomain.keys():
        results[key] = timeDomain[key]

    # -----------------非線形解析-------------------#
    nonlinearDomain = nl.nonlinear(nni=nni)

    for key in nonlinearDomain.keys():
        results[key] = nonlinearDomain[key]

    #不要なパラメータの削除
    del_keylist = ['fft_bands','lomb_bands','ar_bands'
                  ,'fft_nfft','fft_window','fft_resampling_frequency','fft_interpolation'
                  ,'fft_plot','lomb_plot','ar_plot'
                  ,'nni_histogram','poincare_plot','dfa_plot']
    for del_keys in del_keylist:
        del results[del_keys]

    #パラメータの変換
    results = modify_tuple_to_float(results)

    return results

#---------------------------------------------
#パラメータリストをfloat型に変換する
#---------------------------------------------
def modify_tuple_to_float(parameter_list,emotion = "None"):
    #-------------------------------------------------
    #・パラメータがlist型の場合
    #ar_rel : [A B C]を
    #ar_rel_vlf : A, ar_rel_lf : B, ar_rel_vhf : C
    #に変換
    #※パラメータがfloat型の場合はそのまんま
    #-------------------------------------------------
    results = {'emotion' : emotion }
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


if __name__ == '__main__':
    path= r"\\Ts3400defc\共有フォルダ\theme\hrv_daily_fluctuation\05_Analysis\Analysis_Biosignals\emotion\RRI_tohma_2019-09-26_emotion.csv"
    emotion = {'Neutral1':[600,900]  ,'Contentment':[900,1200]
              ,'Neutral2':[1380,1680] ,'Disgust':[1680,1980]
              }
    nni = np.loadtxt(path,delimiter=',')
    df = features(nni,emotion)
    df.to_excel(r"\\Ts3400defc\共有フォルダ\theme\hrv_daily_fluctuation\05_Analysis\Analysis_Features\Analysis_Features_Labels\tohma_2019-09-26_emotion.xlsx",index=None)

    pass
