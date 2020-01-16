# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
# Local Packages
from src.data_processor import *
from src.config import *

# plot
import matplotlib.pyplot as plt



# 特徴量選択
def boruta_regression(dataset,show=False):
    
    # 特徴量選択用のモデル(RandamForest)の定義
    rf = RandomForestRegressor(n_jobs=-1,max_depth=5,random_state=0)
    
    # BORUTAの特徴量選択
    feat_selector = BorutaPy(rf, n_estimators='auto',
                             verbose=2, two_step=False,
                             random_state=0,max_iter=100)
    # BORUTAを実行
    # 最低5個の特徴量が選ばれるそう
    feat_selector.fit(dataset.features, dataset.targets)

    # 出力
    column_list = dataset.features_label_list
    print('\n Initial features: {}'.format(column_list))

    # 選ばれた特徴量の数
    print('\n Number of select feature: {}'.format(feat_selector.n_features_))

    print ('\n Top %d features:' % feat_selector.n_features_)
    
    feature_df = pd.DataFrame(column_list, columns=['features'])

    feature_df['rank']=feat_selector.ranking_
    feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
    print (feature_df.head(feat_selector.n_features_))

    # check ranking of features
    print ('\n Feature ranking:')
    print (feat_selector.ranking_)

    # 特徴量後のデータセット作成
    selected_label = dataset.features_label_list[feat_selector.support_]
    selected_features = dataset.features[: ,feat_selector.support_]

    if show:
        ## 可視化
        mask = feat_selector.support_
        # マスクを可視化する．黒が選択された特徴量
        plt.matshow(mask.reshape(1, -1), cmap='gray_r')
        plt.tick_params(labelleft = 'off')
        plt.xlabel('Sample Index', fontsize=15)
        plt.show()

    return selected_label, selected_features


def build():
    # 学習データ  target : Valence
    emotion_dataset = load_emotion_dataset()
    
    # ------------------
    # データ整形
    # ------------------
    # 標準化 [重要]
    emotion_dataset.features = preprocessing.StandardScaler().fit_transform(emotion_dataset.features)
    
    # 特徴量選択
    selected_label, selected_features = boruta_regression(emotion_dataset,show=False)
    emotion_dataset.features_label_list = selected_label
    emotion_dataset.features = selected_features


    reg = LinearSVR()

    #reg.fit(emotion_dataset.features, 
    #        emotion_dataset.targets)
    ##predict_result = svm_reg.score(emotion_dataset.features, 
    #                               emotion_dataset.targets)
    
    #print(predict_result)

    # 精度検証
    gkf = split_by_group(emotion_dataset)
    # ハイパーパラメータのチューニング
    # グリッドサーチするパラメータを設定
    params_cnt = 20
    params = {"C":np.logspace(-1,2,100), "epsilon":np.logspace(-1,1,params_cnt)}
    gridsearch = GridSearchCV(SVR(kernel="linear"), params, cv=gkf, scoring="r2", return_train_score=True)
    gridsearch.fit(emotion_dataset.features, emotion_dataset.targets)
    print("C, εのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_)
    print("精度 =", gridsearch.best_score_)

    return reg

if __name__ == "__main__":
    build()
