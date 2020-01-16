# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn import preprocessing
from src.data_processor import load_emotion_dataset,split_by_group
from src.LinearSVM.model import load 
from src import config
pass
# データの取得
emotion_dataset = load_emotion_dataset()

# 検証用のデータ
test = pd.read_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\謎\biosignal_datasets_time_Varies_TOHMA.xlsx")

# 特徴量選択
# 個人差補正なし
selected_features = ['fft_peak_hf', 'lomb_abs_vlf', 'lomb_log_vlf', 'sdnn', 'tinn',
       'tri_index', 'sampen', 'bvp_mean', 'bvp_min', 'bvp_median', 'bvp_sd2',
       'pathicData_mean', 'pathicData_std', 'pathicData_log_mean']
selected_test = test.loc[:,selected_features]
print(selected_test)
# 個人差補正
#indiv_test = selected_test.iloc[1:,:] - selected_test.iloc[0,:]


train = emotion_dataset.features[:,np.isin(emotion_dataset.features_label_list,selected_features)]

# 前処理
processed_test = preprocessing.StandardScaler().fit(train).transform(selected_test)
print(processed_test)
# モデル
clf = load("LinearSVM_multiclassification_emotion_filter_feature_select.pickle")
accuracy = clf.predict(processed_test)
# 精度
print(accuracy)

# 確立
digit_score = clf.decision_function(processed_test)
np.savetxt(r"Z:\theme\mental_arithmetic\07.Machine_Learning\個人差補正なし\biosignal_datasets_time_Varies_TOHMA_multi_digit_score.csv",digit_score,delimiter=",")
