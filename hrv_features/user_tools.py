# -*- coding: utf-8 -*-

# Import packages
import matplotlib.pyplot as plt
# Import packages
import numpy as np
import pandas as pd
import seaborn as sns


def plot_data(nni,resp,eda,subject=''):

    tmStamps = 0.001*np.cumsum(nni) #in seconds 
    
    fig,axes = plt.subplots(3,1,sharex=True,figsize = (16,9),subplot_kw=({"xticks":np.arange(0,2160,100)}) )
    #RRI
    axes[0].plot(tmStamps,nni,'b')
    axes[0].set_xlim(0,2160)
    axes[0].set_ylim(600,1400)
    axes[0].set_ylabel("RRI[ms]")
    axes[0].set_title("{}_RRI".format(subject))

    #RESP
    axes[1].plot(resp[:,0],resp[:,1],'b')
    axes[1].set_ylabel('RESP[Hz]')
    axes[1].set_title("{}_RESP".format(subject))

    #EDA 
    tmStamp = 0.001 * np.arange(0,eda.size)
    axes[2].plot(tmStamp,eda,'b')
    axes[2].set_ylabel('EDA[us]')
    axes[2].set_title("{}_EDA".format(subject))

    #塗りつぶし部分を作成
    for i in range(3):
        axes[i].axvspan(900,1200, alpha=0.3,color="b",label="Contentment")
        axes[i].axvspan(1200,1380, alpha=0.1,color="#333333",label="QuestionNaire")
        axes[i].axvspan(1680,1980, alpha=0.3,color="r",label="Disgust")
        axes[i].axvspan(1980,2280, alpha=0.1,color="#333333")
    plt.legend()

    plt.xlabel("Time[s]")
    plt.show()


def features_plot(columns,subjects, path_list):
    fig,axes = plt.subplots(nrows=len(columns), ncols=len(subjects),sharex=True,figsize = (12,8))

    for j, (subject,path) in enumerate(zip(subjects,path_list)):
        df = pd.read_excel(path)

        for i,column in enumerate(columns):

            #y軸のラベルを作成
            axes[i,0].set_ylabel(column)

            #塗りつぶし部分を作成
            axes[i,j].axvspan(900,1200, alpha=0.1,color="b",label="Contentment")
            axes[i,j].axvspan(1200,1380, alpha=0.1,color="g",label="QuestionNaire")
            axes[i,j].axvspan(1680,1980, alpha=0.1,color="r",label="Disgust")
            axes[i,j].axvspan(1980,2280, alpha=0.1,color="g")
            #データをプロット
            df.plot(x ='Time',y = column,ax = axes[i,j]
                    ,legend = False # 凡例を入れるか
                    )
            #if j == 0:
            #    axes[i].axvspan(900,1200, alpha=1.0,color="#ffcdd2",label="Con")
            #    axes[i].axvspan(1200,1380, alpha=0.5,color="y",label="QuestionNaire")
               
            #    df.plot(x =df.columns[0],y = column,ax = axes[i]
            #            ,legend = False # 凡例を入れるか
            #            )
            #    axes[i].set_ylabel(column)


            #ラベルを表示
            
            #axes[i].axvspan(900,1200, alpha=1.0,color="#ffcdd2",label="Ammusement")
            #axes[i].axvspan(1500,1800, alpha=0.5,color="y",label="Disgust")
            

            

            #axes[i].set_xlim(0,1100)
            #axes[i].set_ylim(400,1200)
            #axes[i].set_title("video{}_{}".format(number,subject))

    plt.show()

#path_list = [r"C:\Users\akito\Desktop\Hashimoto\20190802\summary\300s_30s\features_shizuya.xlsx"
#             ,r"C:\Users\akito\Desktop\Hashimoto\20190802\summary\300s_30s\features_kishida.xlsx"
#             ,r"C:\Users\akito\Desktop\Hashimoto\20190802\summary\300s_30s\features_teraki.xlsx"
#             ]
#subjects = ['shizuya','kishida','teraki']
#columns = ['SD1','SD2','SD_ratio','ellipse_area','sample_entropy','dfa_short','dfa_long']
#features_plot(columns,subjects, path_list)
def box_whisker(columns,sum_contentment,sum_disgust,sum_recovery):
  #コラム一覧を取得する
  x = sum_contentment.columns.values
  x_position = np.arange(len(x))

  contentment = sum_contentment.mean().values
  recovery = sum_recovery.std().values
  disgust = sum_disgust.mean().values

  contentment_std = sum_contentment.std().values
  disgust_std = sum_disgust.std().values
  recovery_std = sum_recovery.std().values

  #---------------------描画設定-----------------------#
  sns.set()
  sns.set_style('whitegrid')
  sns.set_palette('Paired')

  #各統計量を結合する
  plt.rcParams["font.size"] = 10
  error_bar_set = dict(lw =1, capthick = 1, capsize = 4)
  fig ,axes = plt.subplots(4,2,figsize = (10,20))

  for j,(ax,column_item) in enumerate(zip(axes.ravel(),columns)):
      x_position = np.arange(len(column_item))
      ax.bar(x_position,contentment[column_item]
               ,yerr = contentment_std[column_item]
               ,width=0.2
               ,label='contentment'
               ,error_kw=error_bar_set
               )
      #ax.bar(x_position + 0.2
      #         ,recovery[column_item]
      #         , yerr = recovery_std[column_item]
      #         ,width=0.2
      #         , label='recovery'
      #         ,error_kw=error_bar_set
      #         )
      ax.bar(x_position + 0.2
               ,disgust[column_item]
               , yerr = disgust_std[column_item]
               ,width=0.2
               , label='disgust'
               ,error_kw=error_bar_set
               )
      pass
      ax.set_xticks(x_position + 0.2)
      ax.set_xticklabels(x[column_item],fontsize = 11)
  #fig.legend(["CONTENTMENT","RECOVERY","DISGUST"],ncol=3,loc="lower center",borderaxespad=0.1)
  fig.legend(["CONTENTMENT","DISGUST"],ncol=2,loc="lower center",borderaxespad=0.1)
  return fig 


def features_analysis_plot(path,subjects,columns
                           , control = [0,900]
                           , contentment = 1200
                           , recovery = 1680
                           , disgust = 1980):


    #エクセルファイルから数値を取得する
    for i,(path_item,subject) in enumerate(zip(path, subjects)):
        df = pd.read_excel(path_item)

        #各セクションごとに分割
        df_control = df[ ( df["Time"] >= control[0]) & (df["Time"] <= control[1]) ]

        df_contentment = df[ df["Time"]== contentment ]
        df_recovery = df[ df["Time"]== recovery ]
        df_disgust = df[ df["Time"]== disgust ]



        #コラム一覧を取得する
        x = df.columns.values
        x_position = np.arange(len(x))
        
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Paired')

        #平均値＋標準偏差を算出
        control_mu = df_control.mean().values / df_control.mean()
        control_sd = df_control.std().values / df_control.mean()


        contentment_mu = df_contentment.mean().values / df_control.mean()
        #contentment_sd = df_contentment.std().values / df_control.mean()

        recovery_mu = df_recovery.mean().values / df_control.mean()
        #recovery_sd = df_recovery.std().values / df_control.mean()

        disgust_mu = df_disgust.mean().values / df_control.mean()
        #disgust_sd = df_disgust.std().values / df_control.mean()


        
        #if i == 0:
        #   a =pd.DataFrame(columns = df.columns)

        #a = a.append((emotion_mu - baseline_mu) >=0 ,  ignore_index=True )
        


        #各統計量を結合する
        y = np.array([control_mu, contentment_mu, recovery_mu, disgust_mu])
        #e = np.array([emotion_sd, contentment_sd,recover_sd])


        error_bar_set = dict(lw =1, capthick = 1, capsize = 10)

        fig = plt.figure(figsize = (24,12))
        ax = fig.add_subplot(1, 1, 1)
        x_position = np.arange(len(columns))
        ax.bar(x_position, control_mu[columns]
               , yerr = control_sd[columns]
               , width=0.2
               , label='control'
               ,error_kw=error_bar_set
               )

        ax.bar(x_position + 0.2
               , contentment_mu[columns]
               #, yerr = emotion_sd[columns]
               ,width=0.2
               , label='emotion'
               #,error_kw=error_bar_set
               )

        ax.bar(x_position+ 0.4, recovery_mu[columns]
               #, yerr = baseline_sd[columns]
               , width=0.2
               , label='recovery'
               #,error_kw=error_bar_set
               )
        ax.bar(x_position + 0.6
               , disgust_mu[columns]
               #, yerr = emotion_sd[columns]
               ,width=0.2
               , label='disgust'
               #,error_kw=error_bar_set
               )


        ax.legend(fontsize=18)
        ax.set_xticks(x_position + 0.2)
        ax.set_xticklabels(x[columns],fontsize = 18)
        #plot.show()
        plt.savefig(r"Z:\theme\emotion\disgust_contentment\20190807\summary\figure\time_nonlinear_domain_{}_300s.png".format(subject))

        
def features_analysis_plot2(path,columns
                           , control = [0,900]
                           , contentment = 1200
                           , recovery = 1680
                           , disgust = 1980,
                           plot = False):

    #エクセルファイルから数値を取得する
    for i,path_item in enumerate(path):

        #Excelファイルからデータを取得する
        df = pd.read_excel(path_item)

        if i == 0:
            #配列を初期化
            sum_contentment =pd.DataFrame(columns = df.columns)
            sum_disgust =pd.DataFrame(columns = df.columns)
            sum_recovery =pd.DataFrame(columns = df.columns)

        #---各セクションごとに分割---#
        df_control = df[ ( df["Time"] >= control[0]) & (df["Time"] <= control[1]) ]
        df_contentment = df[ df["Time"]== contentment ]
        df_recovery = df[ df["Time"]== recovery ]
        df_disgust = df[ df["Time"]== disgust ]

        #-----Neutral状態における平均値＋標準偏差を算出-----#
        control_mu = df_control.mean()
        control_sd = df_control.std()


        #---標準化---#
        contentment_mu = (df_contentment.mean() - control_mu) / control_mu
        disgust_mu = (df_disgust.mean() - control_mu) / control_mu
        recovery_mu = (df_recovery.mean() - control_mu) / control_mu
        
        #------sumパラメータに追加------#
        sum_contentment = sum_contentment.append(contentment_mu,  ignore_index=True )
        sum_disgust = sum_disgust.append(disgust_mu,  ignore_index=True )
        sum_recovery = sum_recovery.append(recovery_mu,  ignore_index=True )
        
    #box_figure = box_whisker(columns
    #            ,sum_contentment
    #            ,sum_disgust
    #            ,sum_recovery)
    #if plot ==True:
    #    plt.show()

    return {'contentment':sum_contentment,'disgust':sum_disgust ,'recovery':sum_recovery}
    

        
    
if __name__ == '__main__':
    #path_list = [r"Z:\theme\emotion\disgust_contentment\20190802\summary\300s_30s\eda_resp_summary_shizuya.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190802\summary\300s_30s\eda_resp_summary_teraki.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190802\summary\300s_30s\eda_resp_summary_kishida.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\eda_resp_summary_kishida.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\eda_resp_summary_kou.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\eda_resp_summary_otsuka.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\eda_resp_summary_shizuya.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\eda_resp_summary_teraki.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\eda_resp_summary_udagawa.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190822\summary\300s_30s\eda_resp_summary_kishida.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190822\summary\300s_30s\eda_resp_summary_moriyama.xlsx"
    #            ,r"Z:\theme\emotion\disgust_contentment\20190822\summary\300s_30s\eda_resp_summary_takase.xlsx"
    #            ]

    #path_list = [
    #    r"Z:\theme\emotion\disgust_contentment\20190822\summary\300s_30s\eda_resp_summary_moriyama_2.xlsx"
    #    ,r"Z:\theme\emotion\disgust_contentment\20190822\summary\300s_30s\eda_resp_summary_takase_2.xlsx"
    #    ,r"Z:\theme\emotion\disgust_contentment\20190822\summary\300s_30s\eda_resp_summary_tohma.xlsx"
    #    ]
    path_list = [
        r"D:\実験データ\20190806\summary\300s_30s\eda_resp_summary_moriyama.xlsx"
        ,r"D:\実験データ\20190806\summary\300s_30s\eda_resp_summary_shibata.xlsx"
        ,r"D:\実験データ\20190806\summary\300s_30s\eda_resp_summary_tohma.xlsx"
        ]


                #,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\features_kishida.xlsx"
                #,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\features_kou.xlsx"
                #,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\features_otsuka.xlsx"
                #,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\features_shizuya.xlsx"
                #,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\features_teraki.xlsx"
                #,r"Z:\theme\emotion\disgust_contentment\20190807\summary\300s_30s\features_udagawa.xlsx"
                #,r"Z:\theme\emotion\disgust_contentment\20190822\summary\300s_30s\features_kishida.xlsx"
                #,r"Z:\theme\emotion\disgust_contentment\20190822\summary\300s_30s\features_moriyama_1.xlsx"
                #,r"Z:\theme\emotion\disgust_contentment\20190822\summary\300s_30s\features_takase_1.xlsx"
     #           ]
    #path_list = [r"\\Ts3400defc\共有フォルダ\theme\emotion\disgust_contentment\20190822\summary\300s_30s\features_moriyama_2.xlsx"
    #             ,r"\\Ts3400defc\共有フォルダ\theme\emotion\disgust_contentment\20190822\summary\300s_30s\features_takase_2.xlsx"
    #             ,r"\\Ts3400defc\共有フォルダ\theme\emotion\disgust_contentment\20190822\summary\300s_30s\features_tohma.xlsx"
    #             ]
    
    columns =[[2,3,4],[8,9,10]
              ,[11,12,13],[34,35,36]
              ,[37,41,43],[44,45,46]
              ,[47,48,49]
              ]
    #subjects = ['shizuya','teraki',  'kishida','kishida2','kou','otsuka','shizuya2','teraki2','udagawa']
    df =  features_analysis_plot2(path_list,columns
                                  ,recovery=900,disgust=900,contentment=1170
                                  #,recovery = 1590, disgust = 1890
                                  )
    #df['disgust'].to_excel(r'\\Ts3400defc\共有フォルダ\theme\emotion\disgust_contentment\summary\ave_summary_disgust_resp_3.xlsx')
    df['contentment'].to_excel(r'C:\Users\akito\Desktop\Hashimoto\summary\ave_summary_contentment_resp_3.xlsx')
    #df['recovery'].to_excel(r'\\Ts3400defc\共有フォルダ\theme\emotion\disgust_contentment\summary\ave_summary_recovery_resp_3.xlsx')



    #sum_contentment = pd.read_excel(r"D:\実験データ\summary\summary\個人差検討\Z変数スケール\summary_contentment.xlsx")
    #sum_disgust = pd.read_excel(r"D:\実験データ\summary\summary\個人差検討\Z変数スケール\summary_disgust.xlsx")
    #sum_recovery = pd.read_excel(r"D:\実験データ\summary\summary\個人差検討\Z変数スケール\summary_recovery.xlsx")

    #box_whisker(columns
    #            ,sum_contentment
    #            ,sum_disgust
    #            ,sum_recovery)
    #plt.show()

def features_analysis(path, start = [0,1200], emotion = [1200,1500], end = [1680,1980]):
    #エクセルファイルから数値を取得する
    subjects = ['sizuya','kaneko','kojima']

    for i,(path_item,subject) in enumerate(zip(path, subjects)):

        df = pd.read_excel(path_item)

        #各セクションごとに分割
        #df_start = df[df['Time'<= 1200]]
        #df_emotion = df[df[('Time'>= 1200) & 'Time'<= 1500]]
        #df_end = df[df[('Time'>= 1680) & 'Time'<= 1980]]

        #分割したセクションごとに平均値，標準偏差を算出
        #各列ごとに平均値と標準偏差を算出する

        data = pd.DataFrame(columns = df.columns)

        for sections in [start,emotion,end]:
            df_section = df[ ( df["Time"] >= sections[0]) & (df["Time"] <= sections[1]) ]
        
            data = data.append(df_section.std() ,  ignore_index=True )
            pass
    
        data.to_excel(r'C:\Users\akito\Desktop\Hashimoto\20190715\Feature_summary\300s_10s\feature_{}_300s_10s_std.xlsx'.format(subject),index = None)
    #for column_name, item in df.iteritems():
    #    item.mean()





#path_list  = [r"Z:\theme\emotion\disgust_amusement\summary\features_kishida_300s_10s.xlsx"
#              #,
#              ]

#features_plot(#columns = [1,2,3,7,8,9,10,11,12]
#              #columns = [33,34,35,36,37,38,39,40,41,42,43,44,45]
#              columns = [42,43,44,45,46,47,48]
#             ,subjects = ['kishida']
#             ,path_list=path_list
#              )


#nni = np.loadtxt(r"C:\Users\akito\Desktop\Hashimoto\20190717\RRI_shibata.csv",delimiter=",")
#resp = np.loadtxt(r"C:\Users\akito\Desktop\Hashimoto\20190717\RESP_shibata.csv",delimiter=",")

#log_data = np.loadtxt(r"C:\Users\akito\Desktop\Hashimoto\20190717\opensignals_201806130003_2019-07-17_15-40-31_shibata.txt")
#signal = log_data[:,5:9]
###A1->RESP
###A2->ECG
###A3->EDA
###A4->PPG

##signal_PZT = Scale.PZT(signal[:,0],10)
##signal_ECG = Scale.ECG(signal[:,1],10)
#signal_EDA = Scale.EDA(signal[:,2],10)

##signal_PPG = signal[:,3]

#plot_data(nni,resp,signal_EDA,'shibata')


#lagged poincare

