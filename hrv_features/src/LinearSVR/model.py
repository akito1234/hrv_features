# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
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
    return selected_label, selected_features

def svr_regression(dataset):
    # 特徴量選択用のモデル(Linear SVM)の定義
    svr = LinearSVR(random_state=0,max_iter=100000)

    gkf = split_by_group(dataset)
    feat_selector = RFECV(estimator=svr, cv=gkf,min_features_to_select=5)
    dataset.features = preprocessing.StandardScaler().fit_transform(dataset.features) 

    # 学習の開始
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
    print(selected_label)
   

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
    selected_label, selected_features = svr_regression(emotion_dataset)
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
    
    print("best parameters:", gridsearch.best_params_)
    
    print("accuracy ", gridsearch.best_score_)
    
    y_pred = gridsearch.best_estimator_.predict(emotion_dataset.features)
    mae = mean_absolute_error(emotion_dataset.targets, y_pred)
    print("mean absolute error : {}".format(mae))
    yyplot(emotion_dataset.targets, y_pred)
    return reg

# visualization
# yyplot 作成関数
def yyplot(y_obs, y_pred):
    yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_obs, y_pred)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01])
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('y_observed', fontsize=24)
    plt.ylabel('y_predicted', fontsize=24)
    plt.title('Observed-Predicted Plot', fontsize=24)
    plt.tick_params(labelsize=16)
    plt.show()

    return fig

if __name__ == "__main__":
    build()
