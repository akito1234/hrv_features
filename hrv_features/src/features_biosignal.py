# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import pickle
# 描画
import matplotlib.pyplot as plt

# Import Localpackage
#from .. import 
from src.data_processor import *
from src.visualization import *
from src.config import *


# Grid Search
def Grid_Search(dataset):
    gkf = split_by_group(dataset)

    #penaltyとlossの組み合わせは三通り
    #              penalty     loss 
    # Standard  |    L2     |   L1
    # LossL2    |    L2     |   L2
    # PenaltyL1 |    L1     |   L2

    # LinearSVCの取りうるモデルパラメータを設定
    C_range= np.logspace(-2, 2, 50)
    param_grid = [{"penalty": ["l2"],"loss": ["hinge"],"dual": [True],"max_iter":[1000000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}, 
                    {"penalty": ["l1"],"loss": ["squared_hinge"],"dual": [False],"max_iter":[1000000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}, 
                    {"penalty": ["l2"],"loss": ["squared_hinge"],"dual": [True],"max_iter":[1000000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}]
    clf = LinearSVC(random_state=1)
    grid_clf = GridSearchCV(clf, param_grid, cv=gkf,n_jobs=-1)

    #モデル訓練
    # 本来は，訓練データとテストデータに分けてfitさせる
    grid_clf.fit(dataset.features, dataset.targets)

    # 結果を出力
    print("\n----- Grid Search Result-----")
    print("Best Parameters: {}".format(grid_clf.best_params_))
    print("Best Cross-Validation Score: {}".format(grid_clf.best_score_))

    return grid_clf.best_estimator_


# 特徴量名一覧から各生体信号に分割する
def Devide_Features_Biosignal(features_label,biosignal_type = ["ECG","RESP","EDA"]):
    print("----------------------------\n")
    print("Separate features for each biological signal")
    apply_features = None
    for type in biosignal_type:
        if type == "RESP":
            selected_features = features_label.str.contains("bvp_")
            print("Append RESP features\n")

        elif type == "EDA":
            selected_features = (features_label.str.contains("sc_") 
                                 | features_label.str.contains("tonicData_") 
                                 | features_label.str.contains("pathicData_"))
            print("Append EDA features\n")
        elif type == "ECG":
            selected_features = ~(features_label.str.contains("sc_") 
                                 | features_label.str.contains("tonicData_") 
                                 | features_label.str.contains("pathicData_")
                                 | features_label.str.contains("bvp_"))
            print("Append ECG features\n")
        # mask処理
        if apply_features is None:
            apply_features = selected_features
        else:
            apply_features = (apply_features | selected_features)

    return apply_features

def build():
    # データの取得
    emotion_dataset = load_emotion_dataset()
    # ------------------
    # データ整形
    # ------------------

    # 標準化 [重要]

    # 外れ値のある特徴量を取り除く
    emotion_dataset.features_label_list = emotion_dataset.features_label_list[~np.isinf(emotion_dataset.features).any(axis=0)]
    emotion_dataset.features = emotion_dataset.features[:, ~np.isinf(emotion_dataset.features).any(axis=0)]
    
    # 生体信号ごとに特徴量を選択する
    biosignal_type = ["ECG","EDA"]
    selected_features = Devide_Features_Biosignal(emotion_dataset.features_label_list,
                                                                    biosignal_type)
    emotion_dataset.features_label_list = emotion_dataset.features_label_list[selected_features]
    
    emotion_dataset.features = emotion_dataset.features[:,selected_features]
    
    emotion_dataset.features = preprocessing.StandardScaler().fit_transform(emotion_dataset.features) 
    
    # label encoding
    le = preprocessing.LabelEncoder().fit(np.unique(emotion_dataset.targets))
    
    # 出力
    print(np.unique(emotion_dataset.targets))
    print(le.transform(np.unique(emotion_dataset.targets)))
    emotion_dataset.targets = le.transform(emotion_dataset.targets)

    # ----------------
    # 特徴量選択
    # ----------------
    selected_label, selected_features = Foward_features_selection(emotion_dataset,False)
    emotion_dataset.features_label_list = selected_label
    emotion_dataset.features = selected_features
    best_model = Grid_Search(emotion_dataset)


    # 重要度描画
    #plot_importance(best_model, emotion_dataset.features_label_list)
    
    # --------------
    # 精度検証
    # --------------
    print("\n------------Result-------------")
    print("Model : Linear SVM")
    gkf = split_by_group(emotion_dataset)
    predict_result = cross_val_predict(best_model, emotion_dataset.features,
                                   emotion_dataset.targets, cv=gkf)
    print("confusion matrix: \n{}".format(confusion_matrix(emotion_dataset.targets, predict_result)))

    score_result = cross_val_score(best_model, emotion_dataset.features,
                                   emotion_dataset.targets, cv=gkf)
    print("Cross-Varidation score: \n{}".format(score_result))
    print("Cross-Varidation score mean: \n {}".format(score_result.mean()))
    print("Cross-Varidation score std: \n {}".format(score_result.std()))
    
    print("Classification Report : \n")
    print(classification_report(predict_result,emotion_dataset.targets,
                                target_names=config.emotion_state))


    return best_model

# 学習モデルを保存する
def save(file_name="model"):
    best_model = build()
    with open("./models/{}.pickle".format(file_name), mode='wb') as fp:
        pickle.dump(best_model,fp)
    print("{}   save...".format("./models/{}.pickle".format(file_name)))
    return best_model

# 学習モデルを復元する
def load(file_name):
    with open("./models/{}".format(file_name),mode="rb") as fp:
        clf = pickle.load(fp)
    return clf

if __name__ == "__main__":
    #save("LinearSVM_TimeWindow_120s_Noralize_ratio")
    build()
    print("success")
    # データの取得
    