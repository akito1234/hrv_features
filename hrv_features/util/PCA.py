import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
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


#---------------------------------------
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


dataset_path = r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_1.xlsx"
df = pd.read_excel(dataset_path)
features = features_baseline(df,emotion_state=['Stress','Amusement'],baseline='Neutral1',
                                identical_parameter = ['id','emotion','user','date','path_name']).to_excel(r"C:\Users\akito\Desktop\test.xlsx")
features.drop(['id','emotion','user','date','path_name'],inplace=True,axis=1)
features = preprocessing.StandardScaler().fit_transform(features)

#主成分分析の実行
pca = PCA()
feature = pca.fit(features)
# データを主成分空間に写像
feature = pca.transform(features)

# 主成分得点
print(pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))]).head())