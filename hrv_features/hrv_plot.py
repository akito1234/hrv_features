# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:58:24 2019

@author: akito
"""

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path


import frequency_domain as fd
import pyhrv.time_domain as td
import pyhrv.nonlinear as nl
import pyhrv.tools as tools

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import scipy.io


#----------------------------------------------------------------------------
# 感情実験結果の補正
#----------------------------------------------------------------------------
def _correction(_df_baseline,_df_emotion,label):
    # 基本統計量を算出   
    # return mean,std,25%,50%,75%,max
    statics_parameter = _df_baseline[label].describe()
    

    print(type(statics_parameter.loc['mean']))
    print(type(statics_parameter.loc['std']))
    print(type(_df_emotion))

    print(len(_df_emotion.columns))
    # 補正する
    # (value - 平均値) /  標準偏差
    # pandas列すべてに適応させる方法がわからなかった....

    df_emotion_std = (_df_emotion[label]  - statics_parameter.loc['mean']) / (statics_parameter.loc['max'] )
    return df_emotion_std

#----------------------------------------------------------------------------
# 時系列グラフ出力
#----------------------------------------------------------------------------
def _plot(_df,label='ANN',legend='Subject'):
    #plt.plot(_df['Time'].values, _df[label].values, label=legend)
    plt.plot([1,2,3,4],_df, label=legend)
    plt.xticks([1,2,3,4],['Neutral1','Contentment','Neutral2','Disgust'])
    plt.grid(True)
    pass

#----------------------------------------------------------------------------
# ファイル取得
#----------------------------------------------------------------------------
def get_filepath(dict,label):
    #　フォルダを指定
    baseline_folder = os.path.join(dict, "baseline")
    emotion_folder = os.path.join(dict, "emotion")

    # ファイル名の取得
    baseline_flist = os.listdir(baseline_folder)
    emotion_flist  = os.listdir(emotion_folder)

    return baseline_flist,emotion_flist
  

def main(dict,label,index_col=None):
    baseline_file,emotion_file = get_filepath(dict,label)


    for (b_path,e_path) in zip(baseline_file,emotion_file):
        user_name = b_path.split('_')[0]

        df_emotion_std = _correction(pd.read_excel(os.path.join(dict,'baseline', b_path),index_col=index_col)
                                     ,pd.read_excel(os.path.join(dict,'emotion', e_path), index_col=index_col)
                                     ,label
                                     )
        _plot(df_emotion_std, label=label,legend= user_name)
    pass


if __name__ == '__main__':
    dict =  r"\\Ts3400defc\共有フォルダ\theme\hrv_daily_fluctuation\05_Analysis\Analysis_Features\Analysis_Features_Labels"
    label = 'fft_abs_hf'
    main(dict,label,index_col=1)
    # 全体の体裁
    plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
    plt.rcParams['font.size'] = 12 #フォントサイズを設定
    plt.rcParams['lines.linewidth'] = 3 

    plt.legend()
    plt.show()
    plt.xlabel("Emotion")
    plt.ylabel(label)
    pass


