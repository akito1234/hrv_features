import pandas as pd
from sklearn import preprocessing,svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# local packages
from biosignal_analysis import features_baseline


# 学習データをインポート
df = pd.read_excel(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets_arousal_valence.xlsx")
df = df[ df["emotion"].isin(['Neutral1','Stress','Amusement']) ]
# 前処理
# Neutral1状態からの差分を算出
pre_df = features_baseline(df,emotion_state=['Stress','Amusement'],baseline='Neutral1')

# カテゴリ特徴量を変換
target_label = "emotion"
target = preprocessing.LabelEncoder().fit_transform(pre_df[target_label])
train  = pre_df.loc[:,["nni_mean","bvp_mean"]]


# 特徴量と正解ラベルを学習データと評価データへ分割(3割をテストデータに)
X_train,X_test,y_train,y_test = train_test_split(train,target,random_state=45,test_size=0.2)

# KNN
clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X_train,y_train)
print(clf.predict(X_test))
print("Score prediction: {}".format(clf.score(X_test,y_test)))



