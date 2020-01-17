# Random Forest Classifier

import pandas as pd
import numpy as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
# Import local packages
from src import config
from src.data_processor import *


def Grid_Search(dataset):
    gkf = split_by_group(dataset)

    ## Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    ## Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    ## Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    #max_depth.append(None)
    ## Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]
    ## Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]
    ## Method of selecting samples for training each tree
    #bootstrap = [True, False]
    ## Create the random grid
    #param_grid = {'n_estimators': n_estimators,
    #               'max_features': max_features,
    #               'max_depth': max_depth,
    #               'min_samples_split': min_samples_split,
    #               'min_samples_leaf': min_samples_leaf,
    #               'bootstrap': bootstrap}
    param_grid={"n_estimators":[15,20,25,30],#バギングに用いる決定木の個数を指定
                     "criterion":["gini","entropy"],#分割基準。gini or entropyを選択。(デフォルトでジニ係数)
                     "max_depth":[2,3,4,5],#木の深さ。木が深くなるほど過学習し易いので、適当なしきい値を設定してあげる。
                     "random_state":[0]} #ランダムseedの設定。seedを設定しないと、毎回モデル結果が変わるので注意。
    clf = RandomForestClassifier(random_state=0)
    print(param_grid)
    
    grid_clf = GridSearchCV(clf, param_grid, cv=gkf, n_jobs=-1)

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
    # Boruta省略
    print(config.selected_label)
    select_features = emotion_dataset.features_label_list.isin(config.selected_label)
    emotion_dataset.features = emotion_dataset.features[:,select_features]

    # Boruta実行
    #selected_label, selected_features = boruta_feature_selection(emotion_dataset,show=False)
    #emotion_dataset.features_label_list = selected_label
    #emotion_dataset.features = selected_features


    # --------------
    # 学習モデル
    # --------------
    # Grid Searchはとばし
    #clf = Grid_Search(emotion_dataset)
    clf = RandomForestClassifier(random_state=0)

    # --------------
    # 精度検証
    # --------------
    print("\n------------Result-------------")
    print("Model : Random Forest Classifier")
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
    save("RandomForestClassifier")