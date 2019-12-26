import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


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


dataset_path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets_1.xlsx"
df = pd.read_excel(dataset_path)
features_org = df[~df["emotion"].isin(["Neutral1","Neutral2"])]
#features_org = features_baseline(df,emotion_state=['Stress','Amusement'],baseline='Neutral1',
#                                identical_parameter = ['id','emotion','user','date','path_name'])
features = features_org.drop(['id','emotion','user','date','path_name'], axis=1)

# 行列の標準化
features = features.apply(lambda x: (x-x.mean())/x.std(), axis=0)
features.drop(features.columns[np.isnan(features).any()],inplace=True, axis=1)
#主成分分析の実行
pca = PCA(n_components=2)
pca_feature = pca.fit(features)
# データを主成分空間に写像
pca_feature = pca.transform(features)

## 主成分得点
#print(pd.DataFrame(pca_feature,
#                   columns=["PC{}".format(x + 1) for x in range(len(features.columns))]).head())
import matplotlib.pyplot as plt

# 第一主成分と第二主成分における観測変数の寄与度をプロットする
#plt.figure(figsize=(6, 6))
#for x, y, name in zip(pca.components_[0], pca.components_[1], features.columns):
#    plt.text(x, y, name)
#plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
#plt.grid()
#plt.xlabel("PC1")
#plt.ylabel("PC2")
#plt.show()

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.show()


#LabelEncoderのインスタンスを生成
le = LabelEncoder()
#ラベルを覚えさせる
le = le.fit(features_org['emotion'])
#ラベルを整数に変換
features_org['emotion'] = le.transform(features_org['emotion'])
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
plt.scatter(pca_feature[:, 0], pca_feature[:, 1], c=list(features_org['emotion']))
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
