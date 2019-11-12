import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------
# ニュートラル状態を差分する
#---------------------------------------------
def correction_neutral_before(df_neutral,df_emotion):

    identical_parameter = ['id','emotion','user','date','path_name']

    identical_df = df_emotion[identical_parameter]

    #不要なパラメータのを除き，Neuralで補正
    df_neutral_features = df_neutral.drop(identical_parameter, axis=1)
    df_emotion_features = df_emotion.drop(identical_parameter, axis=1)
    features_df = (df_emotion_features - df_neutral_features.values)
    result = pd.concat([identical_df,features_df], axis=1,sort=False)
    return result


path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets.xlsx"
df = pd.read_excel(path)

for i in range(df['id'].max() + 1):
    # 各実験データを取り出す
    df_item = df[ df['id']  == i]
    
    # 各セクションごと
    neutral1 = df_item[ df_item['emotion']  == 'Neutral1']
    stress = df_item[ df_item['emotion']  == 'Stress']
    ammusement = df_item[ df_item['emotion']  == 'Ammusement']
    neutral2 = df_item[ df_item['emotion']  == 'Neutral2']


    # 補正　最初のNeutralの特徴量を引く
    cor_stress = correction_neutral_before(neutral1,stress)
    cor_ammusement = correction_neutral_before(neutral1,ammusement)
    cor_neutral2 = correction_neutral_before(neutral1,neutral2)
    if i == 0:
        df_summary = pd.DataFrame([], columns=df_item.columns)
    # ファイルを結合
    df_summary = pd.concat([df_summary,cor_stress, cor_ammusement,cor_neutral2],ignore_index=True,sort=False)

df_summary.to_excel(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets_neutral_base.xlsx")