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
    features_df = (df_emotion_features/df_neutral_features.values)

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

        #

    ## convert to Dataframe
    #result = pd.DataFrame.from_dict(result, orient='index')

    # sort by p values
    sort_result = dict(sorted(result.items(), key=lambda x:x[1]))
    for key in sort_result.keys():
        print('{} : {} '.format(key,sort_result[key]))

    return result



if __name__ == '__main__':
    import matplotlib as mpl
    font = {'family' : 'meiryo'}
    plt.rcParams["font.size"] = 18

    # Excelファイルから特徴量データを取得
    path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets_arousal_valence.xlsx"
    df = pd.read_excel(path)

    # 正規化 (個人差補正)
    df_features = features_baseline(df,emotion_state=['Neutral2','Stress','Amusement'],baseline='Neutral1')

    # 描画設定
    columns = ['hr_mean',
               'bvp_sdnn'
               #'fft_norm_lf',
               #'fft_norm_hf',
               #'fft_ratio'
               ]
    #features_barplot(df_features[~df_features['id'].isin([16,17,18,19,20,21,22,23,24])],columns,emotion_status = ['Ammusement','Stress'])
    #2,3,11,14,

    #16,17,18,19,20,21,22,23,24
    sns.pairplot(data = df_features[df_features['emotion'].isin(['Stress','Amusement'])], 
                 hue='emotion',
                 vars=columns
                )
    
    # コルモゴロフ-スミルノフ検定
    #A = K_S_test(df[~df['id'].isin([])],emotion_status = ['Neutral2','Stress'], identical_parameter = ['id','emotion','path_name'])
    #B = K_S_test(df[~df['id'].isin([])],emotion_status = ['Neutral2','Amusement'], identical_parameter = ['id','emotion','path_name'])
    #C = K_S_test(df[~df['id'].isin([])],emotion_status = ['Amusement','Stress'], identical_parameter = ['id','emotion','path_name'])
    #resultA = pd.DataFrame.from_dict(A, orient='index')
    #resultB = pd.DataFrame.from_dict(B, orient='index')
    #resultC = pd.DataFrame.from_dict(C, orient='index')
    #result = pd.concat([resultA,resultB,resultC], axis=1)
    #result.to_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\K_S_value_FreqDomain.xlsx")



    #colorlist = ["b","r","g","y"]
    #hue_order=['Neutral1','Stress','Neutral2'],palette= colorlist
    #column = 'fft_ratio'
    #replace_df = df_features.replace({'Neutral1':"N1", 'Neutral2':"N2", "Stress":"St", "Ammusement":"Am"})
    #ax = sns.catplot(x='emotion', y=column, col="user", col_wrap=2,
    #                 data=replace_df#[~(df['emotion']=='Ammusement') & (~df['id'].isin([16,17,18,19,20,21,22,23,24]))],
    #                 , aspect=1.2,order = ['N2','St','Am'], kind = 'bar'
    #                 )
    #plt.savefig(r"Z:\theme\mental_stress\04.Figure\physiological_parameter_stress_neutral\2019-11-19~21\{}_subjet_independence.png".format(column))

   # ax.legend(loc="upper center", 
  	#	bbox_to_anchor=(0.5,-0.07), # 描画領域の少し下にBboxを設定
			#ncol=3						# 2列
			#)
    #features_barplot(df_features[ df_features['emotion'] != 'Ammusement'],
    #                 columns, 
    #                 sort_order = ['Stress','Neutral2'])
    plt.show()
    #features_baseline(df).to_excel(r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\biosignal_datasets_neutral_base.xlsx",index=False)