# -*- coding: utf-8 -*-
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# graph packages
import matplotlib.pyplot as plt
import seaborn as sns

#---------------------------------------------
# ニュートラル状態を差分する
#---------------------------------------------
def correction_neutral_before(df_neutral,df_emotion,identical_parameter):
    identical_df = df_emotion[identical_parameter]

    #不要なパラメータのを除き，Neuralで補正
    df_neutral_features = df_neutral.drop(identical_parameter, axis=1)
    df_emotion_features = df_emotion.drop(identical_parameter, axis=1)
    features_df = (df_emotion_features - df_neutral_features.values)

    result = pd.concat([identical_df,features_df], axis=1,sort=False)
    return result

#---------------------------------------------
# 感情ごとの特徴量からベースラインを引く
#---------------------------------------------
def features_baseline(df,emotion_state=['Stress','Ammusement','Neutral2'],
                      baseline='Neutral1',
                      identical_parameter = ['id','emotion','user','date','path_name']):
    df_summary = None
    for i in range(df['id'].max() + 1):
        # 各実験データを取り出す
        df_item = df[ df['id']  == i]

        # ベースラインを取得
        baseline_df = df_item[ df_item['emotion']  == baseline]
        
        # 初期化 　->　空のDataframeを作成
        if df_summary is None:
            df_summary = pd.DataFrame([], columns=baseline_df.columns)

        for state in emotion_state:
            # 各感情状態の特徴を取り出す
            emotion_df = df_item[ df_item['emotion']  == state]
            if emotion_df.empty:
                continue;

            # 各感情状態からベースラインを引く 
            correction_emotion_df = correction_neutral_before(baseline_df,emotion_df,identical_parameter)

            # ファイルを結合
            df_summary = df_summary.append(correction_emotion_df,ignore_index=True,sort=False)

    return df_summary


#---------------------------------------------
# Plot features list by using barplot
#---------------------------------------------
def features_barplot(df,columns=None,sort_order = ['Stress','Ammusement','Neutral2']):
    fig, axes = plt.subplots(1,len(columns))
    for i,column in enumerate(columns):
        sns.barplot(x='emotion', y=column, data=df, ax=axes[i],order=sort_order, capsize=.1)
        
        


#---------------------------------------------
# コルモゴロフ-スミルノフ検定
#---------------------------------------------
def K_S_test(df,emotion_status = ['Neutral2','Stress'], identical_parameter = ['id','emotion','user','date','path_name']):
    if len(emotion_status) != 2:
        print('emotion_statusには感情名を2つ入れる')
        return False
    for column in df.columns:
        # 関連コラムは飛ばす
        if identical_parameter.count(column) > 0:
            continue
        
        specimen_group_1 = df[ df['emotion'] == emotion_status[0]].drop(identical_parameter,axis=1)
        specimen_group_2 = df[ df['emotion'] == emotion_status[1]].drop(identical_parameter,axis=1)

        # K-S検定の実行
        result = stats.ks_2samp(specimen_group_1[column].values, specimen_group_2[column].values)
        result[0]  # 統計検定量
        result[1]  # p-value
        #print(result[0])
        print('{} : {} '.format(column,result[1]))
    return result



if __name__ == '__main__':
    import matplotlib as mpl
    #plt.style.use('ggplot') 
    font = {'family' : 'meiryo'}
    plt.rcParams["font.size"] = 18

    # Excelファイルから特徴量データを取得
    path = r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\biosignal_datasets.xlsx"
    df = pd.read_excel(path)
    df_features = features_baseline(df,emotion_state=['Neutral2','Stress'],baseline='Neutral1')
    
    # コルモゴロフ-スミルノフ検定
    #K_S_test(df_features,emotion_status = ['Neutral2','Stress'])

    # 描画設定
    columns = ['hr_mean',
               'bvp_mean',
               'fft_ratio',
               'pathicData_mean'
               ]
    sns.pairplot(data=df_features, hue='emotion', vars=columns)
    #colorlist = ["b","r", "g"]
    #ax = sns.barplot(x='id', y=df['pathicData_mean'], hue='emotion', data=df[ df['emotion'] != 'Ammusement'],palette	= colorlist)
    #ax.legend().set_visible(False)
    #features_barplot(df_features[ df_features['emotion'] != 'Ammusement'],
    #                 columns, 
    #                 sort_order = ['Stress','Neutral2'])
    plt.show()

    #features_baseline(df).to_excel(r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\biosignal_datasets_neutral_base.xlsx",index=False)