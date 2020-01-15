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

#normalization=True,emotion_filter=Trueの場合に選択された特徴量
#selected_label = ["rmssd","sdnn","lomb_abs_lf","hr_min","hr_mean","nni_max",
#                     "nni_diff_mean","tinn_m","tinn","nni_mean","sd1","sd2",
#                     "nni_counter","ellipse_area","lomb_total","tri_index","ar_rel_vlf"]



# Grid Search
def Grid_Search(dataset):
    gkf = split_by_group(dataset)

    #penaltyとlossの組み合わせは三通り
    #              penalty     loss 
    # Standard  |    L2     |   L1
    # LossL2    |    L2     |   L2
    # PenaltyL1 |    L1     |   L2

    # LinearSVCの取りうるモデルパラメータを設定
    C_range= np.logspace(-2, 2, 10)
    param_grid = [{"penalty": ["l2"],"loss": ["hinge"],"dual": [True],"max_iter":[100000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}, 
                    {"penalty": ["l1"],"loss": ["squared_hinge"],"dual": [False],"max_iter":[100000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}, 
                    {"penalty": ["l2"],"loss": ["squared_hinge"],"dual": [True],"max_iter":[100000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}]
    #param_grid = {
    #      'estimator__C': C_range
    #      }
    #clf = OneVsRestClassifier(LinearSVC())

    clf = LinearSVC()
    grid_clf = GridSearchCV(clf, param_grid, cv=gkf,n_jobs=-1)

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
    emotion_dataset = load_emotion_dataset()
    # ------------------
    # データ整形
    # ------------------
    # 正規化 [重要]
    emotion_dataset.features = preprocessing.StandardScaler().fit_transform(emotion_dataset.features) 
    # one_hot_encoding
    le = preprocessing.LabelEncoder().fit(np.unique(emotion_dataset.targets))
    
    # 出力
    print(np.unique(emotion_dataset.targets))
    print(le.transform(np.unique(emotion_dataset.targets)))
    emotion_dataset.targets = le.transform(emotion_dataset.targets)


    ## one_hot_encoding
    #df_targets = pd.DataFrame(emotion_dataset.targets)
    #enc = preprocessing.OneHotEncoder( sparse=False )
    ## 結果(ndarray)
    #print( enc.fit_transform(df_targets) )
    #emotion_dataset.targets = enc.fit_transform(df_targets)

    # 特徴量選択
    # Boruta省略
    selected_label, selected_features = boruta_feature_selection(emotion_dataset,show=False)
    emotion_dataset.features_label_list = selected_label
    emotion_dataset.features = selected_features
    
    best_model = Grid_Search(emotion_dataset)
    
    # 精度検証
    gkf = split_by_group(emotion_dataset)
    score_result = cross_val_score(best_model, emotion_dataset.features,
                                   emotion_dataset.targets, cv=gkf)
    print("Cross-Varidation score: \n{}".format(score_result))
    print("Cross-Varidation score mean: \n {}".format(score_result.mean()))

    return best_model

# 学習モデルを保存する
def save(file_name="model"):
    best_model = build()
    with open("./model/{}.pickle".format(file_name), mode='wb') as fp:
        pickle.dump(best_model,fp)
    return best_model

# 学習モデルを復元する
def load(file_name):
    with open("./model/{}".format(file_name),mode="rb") as fp:
        clf = pickle.load(fp)
    return clf

if __name__ == "__main__":
    save("linear_svm_apply_all")
    
    print("success")
    # データの取得
    