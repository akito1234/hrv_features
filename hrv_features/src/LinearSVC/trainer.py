# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()


from sklearn import preprocessing
from src.data_processor import load_emotion_dataset,split_by_group
from src.LinearSVC.model import load 
from src import config
from sklearn.svm import SVC
# ---------------------------------------
#　設定
# ---------------------------------------
predict_path = r"Z:\theme\robot_communication\04_Analysis\Analysis_TimeVaries\features_shibata_2020-02-05.xlsx"
selected_features =  ['ar_peak_lf', 'lomb_rel_hf', 'nni_min', 'nni_max', 'hr_max', 'nn50', 'tinn', 'sampen', 'bvp_min', 'sc_mean']
model = "LinearSVM_ALL_FowardFeaturesSelection_2020_02_01.pickle"
emotion_label = ["Amusement","Neutral", "Stress"]
emotion_color = ["b","gray", "r"]
time_record_path = r"Z:\theme\robot_communication\03_LogData\Biosignal\2020-02-05\shibata\robot_communication_2020_02_05__19_46_12.xlsx"


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

#-------------------------------------
# 追加
emotion_dataset = load_emotion_dataset()
emotion_dataset.features = scale_clf.transform(emotion_dataset.features)
emotion_dataset.features = emotion_dataset.features[:,emotion_dataset.features_label_list.isin(selected_features)]



#-------------------------------
# 予測
#-------------------------------
# モデル設定
#clf = load(model)
clf = SVC(decision_function_shape='ovo',kernel="linear",C=0.9545484566618342,probability=True,
          max_iter=-1,random_state= 1)
accuracy = clf.fit(emotion_dataset.features,emotion_dataset.targets).predict(selected_test)
print("accuracy: {}".format(accuracy))


digit_score = clf.predict_proba(selected_test)
print(digit_score)
np.savetxt(r"C:\Users\akito\Desktop\shibata_2.csv",digit_score,delimiter=",")

# 描画
time_record = pd.read_excel(time_record_path,header=0,index_col = 0)
end_time = (time_record.loc["Amusement","FinishDatetime"] - time_record.loc["Neutral","StartDatetime"]).total_seconds()
stress_start = (time_record.loc["Stress archimetic","FinishDatetime"]  - time_record.loc["Neutral","StartDatetime"]).total_seconds()
stress_finish = (time_record.loc["Stress archimetic","StartDatetime"]  - time_record.loc["Neutral","StartDatetime"]).total_seconds()
amusement_start = (time_record.loc["Amusement","StartDatetime"]  - time_record.loc["Neutral","StartDatetime"]).total_seconds()
amusement_finish = (time_record.loc["Amusement","FinishDatetime"]  - time_record.loc["Neutral","StartDatetime"]).total_seconds()

print("End Time[s] : {}".format(end_time))
plt.figure(figsize= (29,18))

plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]=120
plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"
plt.rcParams['xtick.major.width'] = 5.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 5.0#y軸主目盛り線の線幅
plt.rcParams['axes.linewidth'] = 5.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 10
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.labelspacing"] =0.01
plt.rcParams["legend.handletextpad"] = 0.01
plt.tick_params(length = 20)
plt.rcParams['patch.linewidth'] = 5

for i in range(len(emotion_label)):
    plt.plot(time[1:], digit_score[:,i]*100, emotion_color[i],label = emotion_label[i],linewidth = 10)

plt.axvspan(amusement_start,1200,alpha=0.1,color="r"
            #,label="Amusement Section"
            )
plt.axvspan(stress_start, stress_finish,alpha=0.1,color="b"
            #,label="Stress Section"
            )


plt.xlabel("Time [s]")
plt.ylabel("Probability [%]")
plt.yticks(np.arange(0,120,20))
plt.xticks(np.arange(0,1300,200))
plt.ylim(0,100)
plt.xlim(350,1200)
plt.legend(bbox_to_anchor=(0.5,-0.25), loc='upper center',ncol=3, fontsize=100)
plt.show()
#plt.savefig(r"C:\Users\akito\Desktop\Figure_tohma.png", bbox_inches="tight", pad_inches=0.05)