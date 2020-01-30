import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_processor import *
import sklearn.preprocessing
# 特徴量ごとの相関性を確認する
dataset = load_emotion_dataset()
df = pd.DataFrame(dataset.features, columns = dataset.features_label_list)
datalabel = pd.Series(data=dataset.targets)

datalabel = preprocessing.LabelEncoder().fit_transform(datalabel )
gr = pd.plotting.scatter_matrix(df.loc[:,["pathicData_std","tinn_m"]], c=datalabel , figsize=(8,8), marker = 'o', hist_kwds = {'bins':20}, s=60, alpha=.8)
plt.show()


#df = pd.DataFrame(dataset.features,columns=dataset.features_label_list)

#df["y"] = dataset.targets

##相関行列を作成
#corr_matrix = df.corr()
#corr_y = pd.DataFrame({"features":df.columns,"corr_y":corr_matrix["y"]},index=None)
#corr_y = corr_y.reset_index(drop=True)

##相関係数の絶対値をとって、閾値と比較する
#select_cols = corr_y[corr_y["corr_y"].abs()>=0.4]
#print(select_cols)