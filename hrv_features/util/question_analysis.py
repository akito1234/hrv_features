# -*- coding: utf-8 -*-

# load packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 描画設定
plt.style.use('ggplot') 
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)
path = r"Z:\theme\mental_stress\05.QuestionNaire\QuestionNaire_result.xlsx"

#df = pd.read_excel(path, sheet_name="questionnaire", header=0, index_col=0)
#ax = df[df["Emotion"] == "stress"].plot(x='Valence', y='Arousal', kind='scatter',figsize=(10,10),c="r",label="Stress", s=50.0)
#df[~(df["Emotion"] == "stress")].plot(x='Valence', y='Arousal', kind='scatter',ax=ax,c="y",label="Ammusement",s=50.0)
#plt.legend(loc="upper center", 
#  		bbox_to_anchor=(0.5,-0.08), # 描画領域の少し下にBboxを設定
#			ncol=2						# 2列
#			)


###日本語を使う場合は以下の2行でフォントを準備
##from matplotlib.font_manager import FontProperties
##fp = FontProperties(fname='C:\WINDOWS\Fonts\msgothic.ttc', size=14)
##ax = df["trans_Emotion_1"].value_counts().plot(kind="bar")
#plt.show()

def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.xlim(0.5,7.5)
            plt.ylim(0.5,7.5)
            plt.scatter(*args, **kwargs)
            
        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,            
            vertical=True
        )
    # Do also global Hist:
    sns.distplot(
        df[col_x].values,
        ax=g.ax_marg_x,
        color='grey'
    )
    sns.distplot(
        df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color='grey',
        vertical=True
    )
    plt.legend(legends)

if __name__ =="__main__":
    # 描画設定
    plt.style.use('ggplot') 
    font = {'family' : 'meiryo'}
    matplotlib.rc('font', **font)
    path = r"Z:\theme\mental_arithmetic\06.QuestionNaire\QuestionNaire_result.xlsx"

    df = pd.read_excel(path, sheet_name="questionnaire", header=0, index_col=0)
    multivariateGrid('Valence', 'Arousal', 'emotion', df=df,scatter_alpha=1)
    plt.show()