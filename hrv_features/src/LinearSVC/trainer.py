# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn import preprocessing
from src.data_processor import load_emotion_dataset,split_by_group
from src.LinearSVC.model import load 
from src import config
pass
# データの取得
emotion_dataset = load_emotion_dataset()

# 検証用のデータ
test = pd.read_excel(r"Z:\theme\robot_communication\04_Analysis\Analysis_TimeVaries\features_tohma_2020-01-28.xlsx")

# 特徴量選択
# 個人差補正なし


# 注意!
# 300s
#selected_features = ['lomb_rel_hf', 'lomb_total', 'nni_counter', 'nni_mean', 'nni_max', 'hr_mean', 'hr_min',
#                    'nni_diff_mean', 'sdnn', 'rmssd', 'tinn_m', 'tinn', 'tri_index', 'sd1', 'sd2', 'ellipse_area']


selected_features = ['fft_abs_lf',"fft_rel_vlf","fft_log_vlf","ar_peak_vlf","lomb_peak_vlf",
                     "lomb_abs_vlf","lomb_rel_vlf","tinn_n","tinn_"]

#120s
#selected_features = ["nni_mean","nni_counter","nni_max","hr_mean","hr_min","tinn_m"]

selected_test = test.loc[:,selected_features]
print(selected_test)
# 個人差補正
indiv_test = selected_test.iloc[1:,:] - selected_test.iloc[0,:]


train = emotion_dataset.features[:,np.isin(emotion_dataset.features_label_list,selected_features)]

# 前処理
processed_test = preprocessing.StandardScaler().fit(train).transform(selected_test)
print(processed_test)
# モデル
clf = load("LinearSVM_FeaturesSelect_2LabelClassifier.pickle")
accuracy = clf.predict(processed_test)
# 精度
print(accuracy)

# 確立
digit_score = clf.decision_function(processed_test)
print(digit_score)
#np.savetxt(r"Z:\theme\mental_arithmetic\07.Machine_Learning\個人差補正なし\biosignal_datasets_time_Varies_TOHMA_multi_digit_score.csv",digit_score,delimiter=",")
