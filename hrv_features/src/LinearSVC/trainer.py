# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from src.data_processor import load_emotion_dataset,split_by_group
from src.LinearSVC.model import load 
from src import config



# 検証用のデータ
test = pd.read_excel(r"Z:\theme\robot_communication\04_Analysis\Analysis_TimeVaries\features_kaneko_2020-01-28.xlsx"
                     ,index_col = 0,header=0).drop(config.remove_features_label,axis=1)
test_features = test.columns

# 注意!
# 300s
selected_features = ['fft_abs_lf', 'fft_rel_vlf', 'fft_log_vlf', 'ar_peak_vlf', 'lomb_peak_vlf',
                    'lomb_abs_vlf', 'lomb_rel_vlf', 'tinn_n', 'tinn_m', 'sampen', 'bvp_min', 'pathicData_mean', 'pathicData_log_mean']
print("Selected Feature : {}\n".format(selected_features))
#120s
#selected_features = ["nni_mean","nni_counter","nni_max","hr_mean","hr_min","tinn_m"]


#-------------------------------
# 前処理
#-------------------------------
# 個人差補正
test = test.iloc[1:,:] - test.iloc[0,:]
# スケーリング
scale_clf = load("individual_ratio_StandardScaler.pickle")
test = scale_clf.transform(test)
# 特徴量選択
selected_test = test[:,test_features.isin(selected_features)]

#-------------------------------
# 予測
#-------------------------------
# モデル設定
clf = load("LinearSVM_2020-01-29_noramlizeTrue_selectfeature_Foward.pickle")
accuracy = clf.predict(selected_test)
# 精度
print(accuracy)

# 確立
digit_score = clf.decision_function(selected_test)
print(digit_score)
np.savetxt(r"Z:\theme\robot_communication\04_Analysis\Analysis_PredictEmotion\emotion_tohma_2020-01-28.csv"
           ,digit_score,delimiter=",")
