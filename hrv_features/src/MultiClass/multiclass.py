# k


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import preprocessing 
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
# Local Package
from src import config
from src.visualization import *
# Import Localpackage
from src.data_processor import *

def multi_Grid_Search(features,targets,gkf):
    #penaltyとlossの組み合わせは三通り
    #              penalty     loss 
    # Standard  |    L2     |   L1
    # LossL2    |    L2     |   L2
    # PenaltyL1 |    L1     |   L2

    # LinearSVCの取りうるモデルパラメータを設定
    C_range= np.logspace(-2,2,10)
    param_grid = [{"penalty": ["l2"],"loss": ["hinge"],"dual": [True],"max_iter":[500000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}, 
                    {"penalty": ["l1"],"loss": ["squared_hinge"],"dual": [False],"max_iter":[500000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}, 
                    {"penalty": ["l2"],"loss": ["squared_hinge"],"dual": [True],"max_iter":[500000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}]

    clf = LinearSVC(random_state=1,class_weight = 'balanced')
    grid_clf = GridSearchCV(clf, param_grid, cv=gkf, n_jobs=-1)

    #モデル訓練
    # 本来は，訓練データとテストデータに分けてfitさせる
    grid_clf.fit(features, targets)

    # 結果を出力
    print("\n----- Grid Search Result-----")
    print("Best Parameters: {}".format(grid_clf.best_params_))
    print("Best Cross-Validation Score: {}".format(grid_clf.best_score_))

    return grid_clf.best_estimator_

def build():
    # データの取得
    emotion_dataset = load_emotion_dataset()

    # --------------
    # 前処理
    # --------------
    # 標準化 [重要]
    emotion_dataset.features = preprocessing.StandardScaler().fit_transform(emotion_dataset.features) 
    # ラベルエンコード   
    emotion_dataset.targets = preprocessing.LabelEncoder().fit_transform(emotion_dataset.targets)

    feature_label = emotion_dataset.features_label_list
    # --------------
    # 特徴量選択
    # --------------
    print("\n--------Features Selection-----------")
    #print(config.selected_label)
    #select_features = emotion_dataset.features_label_list.isin(config.selected_label)
    #emotion_dataset.features = emotion_dataset.features[:,select_features]
     
    # LinearSVM
    selected_label, selected_features = Foward_features_selection(emotion_dataset)
    emotion_dataset.features_label_list = selected_label
    emotion_dataset.features = selected_features

    # --------------
    # 学習モデル
    # --------------
    gkf = split_by_group(emotion_dataset)
    clf = multi_Grid_Search(emotion_dataset.features,emotion_dataset.targets, gkf)
    
    ##best_clf = multi_Grid_Search(features,targets, gkf)
    #best_clf = clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(features, targets)
    
    # 重要度描画
    #plot_importance(clf, emotion_dataset.features_label_list)
    # --------------
    # 精度検証
    # --------------
    print("\n------------Result-------------")
    print("Model : MultiClassification")
    
    predict_result = cross_val_predict(clf, emotion_dataset.features,
                                   emotion_dataset.targets, cv=gkf)
    print("confusion matrix: \n{}".format(confusion_matrix(emotion_dataset.targets, predict_result)))

    score_result = cross_val_score(clf, emotion_dataset.features,
                                   emotion_dataset.targets, cv=gkf)
    print("Cross-Varidation score: \n{}".format(score_result))
    print("Cross-Varidation score mean: \n {}".format(score_result.mean()))
    print("Cross-Varidation score std: \n {}".format(score_result.std()))
    
    print("Classification Report : \n")
    print(classification_report(predict_result,emotion_dataset.targets))
    return clf

# 学習モデルを保存する
def save(file_name="model"):
    best_model = build()
    with open("./models/{}.pickle".format(file_name), mode='wb') as fp:
        pickle.dump(best_model,fp)
    return best_model

# 学習モデルを復元する
def load(file_name):
    with open("./model/{}".format(file_name),mode="rb") as fp:
        clf = pickle.load(fp)
    return clf

if __name__ =="__main__":
    build()
    #save("LinearSVM_multiclassification_emotion_filter_feature_select")
