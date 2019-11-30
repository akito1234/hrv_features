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
    features_df = (df_emotion_features-df_neutral_features.values)

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
def features_barplot(df,columns=None, emotion_status = ['Neutral2','Stress'],annotation=True):
    # データフレームにフィルタをかける
    df = df[df['emotion'].isin(emotion_status)]

    fig, axes = plt.subplots(1,len(columns))
    for i,column in enumerate(columns):
        sns.barplot(x='emotion', y=column, data=df, ax=axes[i], order=emotion_status, capsize=.1)

        if annotation:
            if i == 0:
                # コルモゴロフ-スミルノフ検定 2標本の検定
                p_value = K_S_test(df,emotion_status = emotion_status, identical_parameter = ['id','emotion','user','date','path_name'])
            # 統計指標を表示
            max_value = df[column].mean() + df[column].std()
            y, h, col = max_value, max_value * 0.05, 'k'
            axes[i].plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c=col)
            axes[i].text(.5, y+h, "p={:.3f}".format(p_value[column]), ha='center', va='bottom', color=col)

#---------------------------------------------
# コルモゴロフ-スミルノフ検定
#---------------------------------------------
def K_S_test(df,emotion_status = ['Neutral2','Stress'], identical_parameter = ['id','emotion','user','date','path_name']):
    if len(emotion_status) != 2:
        print('emotion_statusには感情名を2つ入れる')
        return False
    result = {}
    print("Kolmogorov–Smirnov test")
    print("{} vs {}".format(emotion_status[0],emotion_status[1]))
    for column in df.columns:
        # 関連コラムは飛ばす
        if identical_parameter.count(column) > 0:
            continue
        
        specimen_group_1 = df[ df['emotion'] == emotion_status[0]].drop(identical_parameter,axis=1)
        specimen_group_2 = df[ df['emotion'] == emotion_status[1]].drop(identical_parameter,axis=1)

        # K-S検定の実行
        p_value = stats.ks_2samp(specimen_group_1[column].values, specimen_group_2[column].values)[1]
        result[column] = p_value

        print('{} : {} '.format(column,p_value))

    ## convert to Dataframe
    #result = pd.DataFrame.from_dict(result, orient='index')
    return result



if __name__ == '__main__':
    import matplotlib as mpl
    font = {'family' : 'meiryo'}
    plt.rcParams["font.size"] = 18

    # Excelファイルから特徴量データを取得
    #path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets.xlsx"
    path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets_arousal_valence.xlsx"
    df = pd.read_excel(path)

    # 正規化 (個人差補正)
    df_features = features_baseline(df,emotion_state=['Neutral2','Stress','Ammusement'],baseline='Neutral1')

    # 描画設定
    columns = [#'hr_mean',
               #'bvp_mean',
               'fft_ratio',
               'pathicData_mean'
               ]

    #features_barplot(df_features[~df['id'].isin([5,11,13])],columns,emotion_status = ['Stress', 'Ammusement'],annotation=False)
    #[~df['id'].isin([5,11,13])]
    
    #sns.pairplot(data=df_features, 
    #             hue='emotion',
    #             vars=columns
    #            )
    
    # コルモゴロフ-スミルノフ検定
    A = K_S_test(df_features,emotion_status = ['Ammusement','Neutral2'])
    #result_path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\K_S_value3.xlsx"
    #pd.DataFrame.from_dict(A, orient='index').to_excel(result_path)


    colorlist = ["b","r","g","y"]
    ax = sns.barplot(x='id', y=df['fft_ratio'], hue='emotion', 
                     data=df[~(df['emotion'] == 'Ammusement')]
                     ,palette= colorlist)
    ax.legend()#.set_visible(False)
    #features_barplot(df_features[ df_features['emotion'] != 'Ammusement'],
    #                 columns, 
    #                 sort_order = ['Stress','Neutral2'])
    plt.show()
    features_baseline(df).to_excel(r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\biosignal_datasets_neutral_base.xlsx",index=False)