import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------
# ニュートラル状態を差分する
#---------------------------------------------
def correction_neutral_before(df_neutral,df_emotion):
    correction_df = (df_emotion - df_neutral)
    return correction_df


path = r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\biosignal_datasets.xlsx"
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
    df_summary = pd.concat([cor_stress, cor_ammusement,cor_neutral2],ignore_index=True)

df_summary.to_excel(r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\biosignal_datasets_neutral_base.xlsx")