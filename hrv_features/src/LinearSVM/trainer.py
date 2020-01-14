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
test = pd.read_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_time_Varies_TERAKI.xlsx")

# 特徴量選択
selected_features = ["sd1","hr_mean","tinn","hr_min","ellipse_area","sd2",
                     "nni_diff_mean","nni_mean","tri_index","nni_counter",
                     "sdnn","rmssd","sdsd","lomb_total","tinn_m","nni_max"
                     ]


train = emotion_dataset.features[:,np.isin(emotion_dataset.features_label_list,selected_features)]
# 正規化 [重要]
ps = preprocessing.StandardScaler().fit(train)


# 個人差補正
selected_test = test.loc[:,selected_features]
indiv_test = selected_test.iloc[1:,:] - selected_test.iloc[0,:]

# 標準化
processed_test = ps.transform(indiv_test)
print(processed_test)
# モデル
clf = load("linear_svm_apply_all.pickle")
accuracy = clf.predict(processed_test)

# 精度
print(accuracy)