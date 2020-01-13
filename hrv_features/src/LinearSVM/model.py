# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE,RFECV
from sklearn.neighbors import KNeighborsRegressor
from boruta import BorutaPy
from multiprocessing import cpu_count
import pickle
# 描画
import matplotlib.pyplot as plt

# Import Localpackage
#from .. import 
from src.data_processor import load_emotion_dataset,split_by_group,boruta_feature_selection

# Grid Search
def Grid_Search(dataset):
    gkf = split_by_group(dataset)

    #penaltyとlossの組み合わせは三通り
    #              penalty     loss 
    # Standard  |    L2     |   L1
    # LossL2    |    L2     |   L2
    # PenaltyL1 |    L1     |   L2

    # LinearSVCの取りうるモデルパラメータを設定
    C_range= np.logspace(-1, 2, 20)
    #param_grid = [{"penalty": ["l2"],"loss": ["hinge"],"dual": [True],"max_iter":[12000],
    #                'C': C_range, "tol":[1e-3]}, 
    #                {"penalty": ["l1"],"loss": ["squared_hinge"],"dual": [False],"max_iter":[12000],
    #                'C': C_range, "tol":[1e-3]}, 
    #                {"penalty": ["l2"],"loss": ["squared_hinge"],"dual": [True],"max_iter":[12000],
    #                'C': C_range, "tol":[1e-3]}]
    param_grid = {
          'estimator__C': C_range
          }
    #clf = OneVsRestClassifier(LinearSVC())

    clf = OneVsRestClassifier(LinearSVC())
    grid_clf = GridSearchCV(clf, param_grid, cv=gkf)

    #モデル訓練
    # 本来は，訓練データとテストデータに分けてfitさせる
    grid_clf.fit(dataset.features, dataset.targets)

    # 結果を出力
    print("\n----- Grid Search Result-----")
    print("Best Parameters: {}".format(grid_clf.best_params_))
    print("Best Cross-Validation Score: {}".format(grid_clf.best_score_))

    return grid_clf.best_estimator_

def build():
    # データの取得
    emotion_dataset = load_emotion_dataset(normalization=True,
                                           emotion_filter=False)
    # ------------------
    # データ整形
    # ------------------
    # 正規化 [重要]
    emotion_dataset.features = preprocessing.StandardScaler().fit_transform(emotion_dataset.features) 
    # one_hot_encoding
    le = preprocessing.LabelEncoder().fit(np.unique(emotion_dataset.targets))
    emotion_dataset.targets = le.transform(emotion_dataset.targets)
    
    ## one_hot_encoding
    #df_targets = pd.DataFrame(emotion_dataset.targets)
    #enc = preprocessing.OneHotEncoder( sparse=False )
    ## 結果(ndarray)
    #print( enc.fit_transform(df_targets) )
    #emotion_dataset.targets = enc.fit_transform(df_targets)

    # 特徴量選択
    selected_label, selected_features = boruta_feature_selection(emotion_dataset,show=False)
    emotion_dataset.features_label_list = selected_label
    emotion_dataset.features = selected_features
    
    best_model = Grid_Search(emotion_dataset)
    return best_model

def save(file_name="model"):
    best_model = build()
    with open("./model/{}.pickle".format(file_name), mode='wb') as fp:
        pickle.dump(best_model,fp)

if __name__ == "__main__":
    build()
    print("success")