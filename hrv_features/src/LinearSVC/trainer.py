# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from src.data_processor import load_emotion_dataset,split_by_group
from src.LinearSVC.model import load 
from src import config

# ---------------------------------------
#　設定
# ---------------------------------------
predict_path = r"Z:\theme\robot_communication\04_Analysis\Analysis_TimeVaries\features_kishida_2020-01-28.xlsx"
selected_features =  ['ar_peak_lf', 'lomb_rel_hf', 'nni_min', 'nni_max', 'hr_max', 'nn50', 'tinn', 'sampen', 'bvp_min', 'sc_mean']
model = "LinearSVM_ALL_FowardFeaturesSelection_2020_02_01.pickle"
emotion_label = ["Amusement","Neutral", "Stress"]
time_record_path = r"Z:\theme\robot_communication\03_LogData\2020-01-28\kishida\robot_communication_2020_01_28__19_29_33.xlsx"


# 検証用のデータ
test = pd.read_excel(predict_path,index_col = 0,header=0).drop(config.remove_features_label,axis=1)
test_features = test.columns
time = test.index.tolist()


#-------------------------------
# 前処理
#-------------------------------
# 個人差補正
# !重要!
test = test.iloc[1:,:] / test.iloc[0,:]
# スケーリング
scale_clf = load("individual_ratio_StandardScaler_2020_02_01.pickle")
# !重要!
test = scale_clf.transform(test)

# 特徴量選択
print("Selected Feature : {}\n".format(selected_features))
selected_test = test[:,test_features.isin(selected_features)]

#-------------------------------
# 予測
#-------------------------------
# モデル設定
clf = load(model)
accuracy = clf.predict(selected_test)
print("accuracy: {}".format(accuracy))


digit_score = clf.decision_function(selected_test)



# 描画
time_record = pd.read_excel(time_record_path,header=0,index_col = 0)
end_time = (time_record.loc["Amusement","FinishDatetime"] - time_record.loc["Neutral","StartDatetime"]).total_seconds()
stress_start = (time_record.loc["Stress archimetic","FinishDatetime"]  - time_record.loc["Neutral","StartDatetime"]).total_seconds()
stress_finish = (time_record.loc["Stress archimetic","StartDatetime"]  - time_record.loc["Neutral","StartDatetime"]).total_seconds()
amusement_start = (time_record.loc["Amusement","StartDatetime"]  - time_record.loc["Neutral","StartDatetime"]).total_seconds()
amusement_finish = (time_record.loc["Amusement","FinishDatetime"]  - time_record.loc["Neutral","StartDatetime"]).total_seconds()


print("End Time[s] : {}".format(end_time))
plt.figure(figsize= (16,9))

plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]=14
plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"

for i in range(len(emotion_label)):
    plt.plot(time[1:], digit_score[:,i],label = emotion_label[i])

    plt.xlim(300,end_time)
plt.axvspan(amusement_start,amusement_finish,alpha=0.1,color="b",label="Amusement")
plt.axvspan(stress_start, stress_finish,alpha=0.1,color="r",label="Stress")
plt.legend()
plt.tight_layout()
plt.xlabel("Time [s]")
plt.ylabel("Prediction Score [-]")
plt.show()