# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold,LeaveOneOut,GroupKFold

# local packages
from biosignal_analysis import features_baseline

# 描画
import matplotlib.pyplot as plt



def get_Questionnaire(path):
    df = pd.read_excel(path,sheet_name="questionnaire")

    pass


# ---------------------------------
# 学習データをインポート
#----------------------------------
dataset = pd.read_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets.xlsx")
# AmusementとStressを判別したいため，Neutral2を取り除く
dataset = dataset[~dataset["emotion"].isin(['Neutral2'])]


# ---------------------------------
# 前処理
#----------------------------------
# Neutral1をベースラインにとる
df = features_baseline(dataset,emotion_state=['Stress','Amusement'],baseline='Neutral1')

# 変換
target_label = "emotion"
targets = preprocessing.LabelEncoder().fit_transform(df[target_label])
feature_label = ["nni_mean"
                 #,"bvp_mean"
                 ,"bvp_sdnn"
                 ]
features = df.loc[:,feature_label]

# Hold out
# 特徴量と正解ラベルを学習データと評価データへ分割(1割をテストデータに)
X_train,X_test,y_train,y_test = train_test_split(features,targets, random_state=0,
                                                 test_size=0.1, stratify=targets)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("-----Strat Machine learning-----")
print("Info: \n train data num: {} \n test data num: {}".format(X_train.shape[0],X_test.shape[0]))
print("\n feature label : \n {}".format(feature_label))
print("\n emotion label : \n Amusement, Stress")


# ---------------------------------
# モデル定義
#----------------------------------
# KNN
knn = KNeighborsClassifier(n_neighbors=6)
# linear svm 
linear_SVM = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-3, C=2.,max_iter=2000)

# ---------------------------------
# モデル評価
#----------------------------------
# 交差検証
# 層化K分割交差検証
kfold = StratifiedKFold(n_splits=6, shuffle=True,random_state=0)
# 1つ抜き交差検証
loo = LeaveOneOut()
# グループ付き交差検証
le = preprocessing.LabelEncoder()
a = le.fit(df["subject"].unique())
print(df["subject"].unique())
print(le.transform(df["subject"].unique()))
subjects = le.transform(df["subject"])



# 交差検証で，スケール合わせる方法が分からない
features_scaled = scaler.transform(features)
scores = cross_val_score(linear_SVM, features_scaled,targets, 
                         subjects,cv=GroupKFold(n_splits=10) 
                         )

print("Cross-Varidation score: \n{}".format(scores))
print("Cross-Varidation score mean: \n {}".format(scores.mean()))

#plt.figure(figsize=(12,7))
#plt.title("emotion recognition plot")
#plt.scatter(features_scaled[(targets==0),0],features_scaled[(targets==0),1])
#plt.scatter(features_scaled[(targets==1),0],features_scaled[(targets==1),1])

clf = linear_SVM.fit(features_scaled,targets)



# 可視化の準備
xmin, xmax, ymin, ymax = (features_scaled[:,0].min()-1, features_scaled[:,0].max()+1,
                            features_scaled[:,1].min()-1, features_scaled[:,1].max()+1)    
x_ = np.arange(xmin, xmax, 0.01)
y_ = np.arange(ymin, ymax, 0.01)
xx, yy = np.meshgrid(x_, y_)


# 予測
zz = clf.predict(np.stack([xx.ravel(), yy.ravel()], axis=1)
                ).reshape(xx.shape)

# 可視化
plt.pcolormesh(xx, yy, zz, cmap="winter", alpha=0.1, shading="gouraud")
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=targets, edgecolors='k', cmap="winter")
plt.show()