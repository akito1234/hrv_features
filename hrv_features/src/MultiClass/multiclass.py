
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
# Local Package
from src import config


def multi_split_by_group(features,targets,subjects):
    print(pd.unique(subjects))
    # 被験者ごとにグループ分け
    le = preprocessing.LabelEncoder()
    unique_subject = le.fit(np.unique(subjects))
    trans_subjects = le.transform(subjects)
    gkf = list(GroupKFold(n_splits=len(np.unique(trans_subjects))).split(features,targets,trans_subjects))
    #　出力
    print(np.unique(trans_subjects))
    return gkf

def multi_Grid_Search(features,targets,gkf):
    #penaltyとlossの組み合わせは三通り
    #              penalty     loss 
    # Standard  |    L2     |   L1
    # LossL2    |    L2     |   L2
    # PenaltyL1 |    L1     |   L2

    # LinearSVCの取りうるモデルパラメータを設定
    C_range= np.logspace(-2,2,100)
    param_grid = [{"penalty": ["l2"],"loss": ["hinge"],"dual": [True],"max_iter":[500000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}, 
                    {"penalty": ["l1"],"loss": ["squared_hinge"],"dual": [False],"max_iter":[500000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}, 
                    {"penalty": ["l2"],"loss": ["squared_hinge"],"dual": [True],"max_iter":[500000],
                    'C': C_range, "tol":[1e-3],"random_state":[0]}]

    clf = LinearSVC()
    grid_clf = GridSearchCV(clf, param_grid, cv=gkf, n_jobs=-1)

    #モデル訓練
    # 本来は，訓練データとテストデータに分けてfitさせる
    grid_clf.fit(features, targets)

    # 結果を出力
    print("\n----- Grid Search Result-----")
    print("Best Parameters: {}".format(grid_clf.best_params_))
    print("Best Cross-Validation Score: {}".format(grid_clf.best_score_))

    return grid_clf.best_estimator_


dataset = pd.read_excel(config.features_path)
# Neutral2以外のデータを取り出す
dataset = dataset.query("emotion != ['Neutral2'] ")

# 目標変数
target_label = dataset["emotion"]
targets = preprocessing.LabelEncoder().fit_transform(target_label)

# 説明変数
features = dataset.drop(config.identical_parameter,axis=1)
features_label = features.columns
features = preprocessing.StandardScaler().fit_transform(features) 
#features = preprocessing.MinMaxScaler().fit_transform(features) 

# ANOVA F-Values
#selector = SelectKBest(f_classif, k=10).fit(features, targets)
#mask = selector.get_support()
#print(features_label)
#print("selected: {}".format(features_label[mask]))

#------------------
# 特徴量選択
#------------------
# 特徴量選択用のモデル(RandamForest)の定義
rf = RandomForestClassifier(n_jobs=-1, max_depth=7,random_state=0)
# BORUTAの特徴量選択
feat_selector = BorutaPy(rf, n_estimators='auto',
                         verbose=2, two_step=False,
                         random_state=42,max_iter=100)
feat_selector.fit(features, targets)
features = features[:,feat_selector.support_]
print(features_label[feat_selector.support_])


#------------------
# モデルの構築
#------------------
# GridSearch
gkf = multi_split_by_group(features,targets,dataset["user"])
#best_clf = RandomForestClassifier(n_jobs=-1, max_depth=5)

#best_clf = multi_Grid_Search(features,targets, gkf)
best_clf = clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(features, targets)



# --------------
# 精度検証
# --------------
print("\n------------Result-------------")
print("Model : MultiClassifier")
predict_result = cross_val_predict(best_clf, features,targets, cv=gkf)
print("confusion matrix: \n{}".format(confusion_matrix(targets, predict_result)))

score_result = cross_val_score(best_clf, features,targets, cv=gkf)
print("Cross-Varidation score: \n{}".format(score_result))
print("Cross-Varidation score mean: \n {}".format(score_result.mean()))
print("Cross-Varidation score std: \n {}".format(score_result.std()))
    
print("Classification Report : \n")
print(classification_report(predict_result,targets))
