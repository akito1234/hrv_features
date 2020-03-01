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

def build():
    # データの取得
    emotion_dataset = load_emotion_dataset()


    # 追加
    # 検証用のデータ
    test = pd.read_excel(r"C:\Users\akito\Desktop\Analysis_TimeVaries\features_kaneko_2020-01-28.xlsx"
                         ,index_col = 0,header=0).drop(config.remove_features_label,axis=1)
    test_features = test.columns
    datetime = test.index
    # ------------------
    # データ整形
    # ------------------
    # 標準化 [重要]
    emotion_dataset.features_label_list = emotion_dataset.features_label_list[~np.isinf(emotion_dataset.features).any(axis=0)]
    emotion_dataset.features = emotion_dataset.features[:, ~np.isinf(emotion_dataset.features).any(axis=0)]


    sc = preprocessing.StandardScaler()
    emotion_dataset.features = sc.fit_transform(emotion_dataset.features) 

    # 追加
    # 個人差補正
    ## config. ratio
    test = test.iloc[1:,:].values / test.iloc[0,:].values
    test = sc.transform(test)

    #with open("./models/{}.pickle".format("individual_ratio_StandardScaler_2020_02_01"), mode='wb') as fp:
    #        pickle.dump(sc,fp)

    # label encoding
    le = preprocessing.LabelEncoder().fit(np.unique(emotion_dataset.targets))
    
    # 出力
    print(np.unique(emotion_dataset.targets))
    print(le.transform(np.unique(emotion_dataset.targets)))
    emotion_dataset.targets = le.transform(emotion_dataset.targets)

    # ----------------
    # 特徴量選択
    # ----------------
    # Boruta
    #selected_label, selected_features = boruta_feature_selection(emotion_dataset,show=False)
    selected_label, selected_features = Foward_feature_selection(emotion_dataset)
    emotion_dataset.features_label_list = selected_label
    emotion_dataset.features = selected_features
    #selected_label =  ['ar_peak_lf', 'lomb_rel_hf', 'nni_min', 'nni_max', 'hr_max', 'nn50', 'tinn',
                      #'sampen', 'bvp_min', 'sc_mean']

    #selected_label =  ['fft_peak_lf', 'fft_abs_lf', 'ar_peak_lf'
    #                   , 'lomb_abs_vlf', 'lomb_rel_vlf', 'nni_counter', 
    #                   'nni_max', 'tinn', 'sd_ratio']
    #emotion_dataset.features = emotion_dataset.features[:,emotion_dataset.features_label_list.isin(selected_label)]
    #emotion_dataset.features_label_list = emotion_dataset.features_label_list[emotion_dataset.features_label_list.isin(selected_label)]
    # 追加


    # 特徴量選択
    selected_test = test[:,test_features.isin(selected_label)]


    best_model = Grid_Search(emotion_dataset)
    #best_model = LinearSVC(C= 0.2442053094548651, dual= True, loss= 'squared_hinge', max_iter= 1000000, penalty='l2', 
    #                       random_state= 0, tol= 0.001)
    #best_model = load("LinearSVM_ALL_FowardFeaturesSelection_2020_02_01.pickle")


    #np.savetxt(r"Z:\theme\mental_arithmetic\05.Figure\FeaturesSelectionREFCV\ALL_Features_Importance.csv",
    #           best_model.coef_,delimiter=",")
    #np.savetxt(r"Z:\theme\mental_arithmetic\05.Figure\FeaturesSelectionREFCV\ALL_Features_list.csv",
    #             emotion_dataset.features_label_list,delimiter=",")

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
    print(classification_report(predict_result,emotion_dataset.targets))

    # 追加
    accuracy = best_model.predict(selected_test)
    # 精度
    print(accuracy)

    # 確率
    predict_score = best_model.decision_function(selected_test)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(datetime.tolist()[1:],predict_score)
    plt.show()
    print(predict_score)
    np.savetxt(r"C:\Users\akito\Desktop\Analysis_TimeVaries\result\score_kaneko_2020-01-28_2.xlsx"
               ,predict_score,delimiter=",")
    #np.savetxt(r"Z:\theme\robot_communication\04_Analysis\Analysis_PredictEmotion\accuracy_device1_tohma_2020-01-31.csv"
    #           ,accuracy,delimiter=",")

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
    #save("LinearSVM_2020-01-29_noramlizeTrue_selectfeature_SVC")
    print("success")
    # データの取得
    