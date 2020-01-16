# Random Forest Classifier

import pandas as pd
import numpy as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
# Import local packages
from src import config
from src.data_processor import *


def Grid_Search(dataset):
    pass


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


    # --------------
    # 特徴量選択
    # --------------
    print("\n--------Features Selection-----------")
    print(config.selected_label)
    select_features = emotion_dataset.features_label_list.isin(config.selected_label)
    emotion_dataset.features = emotion_dataset.features[:,select_features]


    # --------------
    # 学習モデル
    # --------------
    # Grid Searchはとばし
    clf = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)


    # --------------
    # 精度検証
    # --------------
    print("\n------------Result-------------")
    print("Model : Linea rDiscriminant Analysis")
    gkf = split_by_group(emotion_dataset)
    predict_result = cross_val_predict(clf, emotion_dataset.features,
                                   emotion_dataset.targets, cv=gkf)
    print("confusion matrix: \n{}".format(confusion_matrix(emotion_dataset.targets, predict_result)))

    score_result = cross_val_score(clf, emotion_dataset.features,
                                   emotion_dataset.targets, cv=gkf)
    print("Cross-Varidation score: \n{}".format(score_result))
    print("Cross-Varidation score mean: \n {}".format(score_result.mean()))
    print("Cross-Varidation score std: \n {}".format(score_result.std()))
    
    print("Classification Report : \n")
    print(classification_report(predict_result,emotion_dataset.targets,
                                target_names=["Amusement","Stress"]))

    return clf

# 学習モデルを保存する
def save(file_name="model"):
    best_model = build()
    with open("./models/{}.pickle".format(file_name), mode='wb') as fp:
        pickle.dump(best_model,fp)
    print("{}   save...".format("./models/{}.pickle".format(file_name)))
    return best_model
if __name__ =="__main__":
    #build()

    save("LinearDiscriminantAnalysis")