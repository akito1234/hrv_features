# -*- coding: utf-8 -*-

# load packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# 描画設定
plt.style.use('ggplot') 
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)
path = r"Z:\theme\mental_stress\05.QuestionNaire\QuestionNaire_result.xlsx"
df = pd.read_excel(path, sheet_name="questionnaire", header=0, index_col=0)
ax = df[df["Emotion"] == "stress"].plot(x='Valence', y='Arousal', kind='scatter',figsize=(10,10),c="r",label="Stress", s=50.0)
df[~(df["Emotion"] == "stress")].plot(x='Valence', y='Arousal', kind='scatter',ax=ax,c="y",label="Ammusement",s=50.0)
plt.legend(loc="upper center", 
  		bbox_to_anchor=(0.5,-0.08), # 描画領域の少し下にBboxを設定
			ncol=2						# 2列
			)


##日本語を使う場合は以下の2行でフォントを準備
#from matplotlib.font_manager import FontProperties
#fp = FontProperties(fname='C:\WINDOWS\Fonts\msgothic.ttc', size=14)
#ax = df["trans_Emotion_1"].value_counts().plot(kind="bar")
plt.show()