
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import preprocessing 
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Local Package
from src import config

dataset = pd.read_excel(config.features_path)
# Neutral2以外のデータを取り出す
dataset = dataset.query("emotion != ['Neutral2']")

# 目標変数
target_label = dataset["emotion"]
#targets = OneHotEncoder(sparse=False).fit_transform(target_label)
targets = preprocessing.LabelEncoder().fit_transform(target_label)

# 説明変数
features = dataset.drop(config.identical_parameter,axis=1)
features_label = features.columns
features = preprocessing.StandardScaler().fit_transform(features) 


# ANOVA F-Values
#selector = SelectKBest(f_classif, k=10).fit(features, targets)
#mask = selector.get_support()
#print(features_label)
#print("selected: {}".format(features_label[mask]))

# 特徴量選択用のモデル(RandamForest)の定義
rf = RandomForestClassifier(n_jobs=-1,class_weight='balanced', max_depth=5)

# BORUTAの特徴量選択
feat_selector = BorutaPy(rf, n_estimators='auto',
                            verbose=2, two_step=False,
                            random_state=42,max_iter=70)

# BORUTAを実行
# 最低5個の特徴量が選ばれるそう
feat_selector.fit(features, targets)

print(features_label[feat_selector.support_])


# モデル構築
clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(features[:, feat_selector.support_], targets)
accuracy = clf.score(features[:, feat_selector.support_], targets)
print(accuracy)
