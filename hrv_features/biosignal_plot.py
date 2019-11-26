# import package
import numpy as np
import pandas as pd
import seaborn as sns
from biosppy import signals
import matplotlib.pyplot as plt

# import local
import eda_analysis
import resp_analysis
import pyhrv.frequency_domain as fd
from opensignalsreader import OpenSignalsReader




def plot_signal(path):
    arc = OpenSignalsReader(path)

    # 心拍データからピークを取り出す
    ecg_result = signals.ecg.ecg(signal= arc.signal(['ECG']) , sampling_rate=1000.0, show=False)
    # 皮膚コンダクタンス からSCRを取り出す
    eda_result = eda_analysis.scr(arc.signal(['EDA']))

    # 呼吸周波数を取り出す
    resp_result = resp_analysis.resp(arc.signal('RESP'),show=False)

    # 描画設定
    fig,axes = plt.subplots(3,1,sharex=True,figsize = (16,9),subplot_kw=({"xticks":np.arange(0,1200,100)}) )
    axes[0].set_title(path)
    # 心拍変動の描画    
    axes[0].plot(ecg_result['heart_rate_ts'],ecg_result['heart_rate'],'b')
    axes[0].set_xlim(0,1200)
    axes[0].set_ylabel("HR[bpm]")

    # 皮膚コンダクタンスの描画  
    axes[1].plot(eda_result['ts'], eda_result['filtered'])
    axes[1].set_ylabel('SCR[us]')

    # 呼吸の描画    
    axes[2].plot(resp_result['resp_rate_ts'],resp_result['resp_rate'],'b')
    axes[2].set_ylabel('RESP[Hz]')
    axes[2].set_ylim(0,0.5)


    for i in range(3):
        axes[i].axvspan(300,600,alpha=0.3,color="r",label="Stress")
        axes[i].axvspan(900,1200,alpha=0.3,color="b",label="Amusement")
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Time[s]")
    return plt

def plot_raw(path,emotion_section={"Stress":[300,600],"Amusement":[900,1200]}):
    arc = OpenSignalsReader(path)

    # 心拍データからピークを取り出す
    ecg_result = signals.ecg.ecg(signal= arc.signal(['ECG']) , sampling_rate=1000.0, show=False)
    # 皮膚コンダクタンス からSCRを取り出す
    eda_filtered = eda_analysis.eda_preprocess(arc.signal(['EDA']),1000.)

    # 呼吸周波数を取り出す
    resp_result = resp_analysis.resp(arc.signal('RESP'),show=False)

    # 描画設定
    fig,axes = plt.subplots(3,1,sharex=True,figsize = (16,9),subplot_kw=({"xticks":np.arange(0,1200,100)}) )
    axes[0].set_title(path)
    # 心拍変動の描画    
    axes[0].plot(ecg_result['heart_rate_ts'],ecg_result['heart_rate'],'b')
    axes[0].set_xlim(0,1200)
    axes[0].set_ylabel("HR[bpm]")

    # 皮膚コンダクタンスの描画  
    axes[1].plot(arc.t,arc.signal(['EDA']))
    axes[1].set_ylabel('[us]')
    axes[1].set_ylim(0,25)
    # 呼吸の描画    
    axes[2].plot(arc.t,arc.signal(['RESP']))
    axes[2].set_ylabel('RESP[Hz]')
    axes[2].set_ylim(-50,50)

    for i in range(3):
        for key in emotion_section.keys():
            if key == "Stress":
                axes[i].axvspan(emotion_section[key][0], #start
                                emotion_section[key][1], #end
                                alpha=0.3,color="r",label="Stress")
            elif key == "Amusement":
                axes[i].axvspan(emotion_section[key][0], #start
                                emotion_section[key][1], #end
                                alpha=0.3,color="b",label="Amusement")
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Time[s]")
    return plt



def plot_hrv(path):
    df = pd.read_excel(path,index_col=0)
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(16,9))
    fig,axes = plt.subplots(2,1,sharex=True,figsize = (16,9),subplot_kw=({"xticks":np.arange(0,1200,100)}) )
    axes[0].plot(df.index.values, df['fft_ratio'].values, color='black',label='LF/HF')
    axes[0].set_xlim(0,1200)
    axes[0].set_ylabel("LF/HF [-]")

    axes[1].plot(df.index.values, df['fft_abs_hf'].values, color='red',label='HF')
    axes[1].plot(df.index.values, df['fft_abs_lf'].values, color='blue',label='LF')
    axes[1].set_ylabel('Power Specktraum[ms2]') # ラベルを設定
    for i in range(2):
        axes[i].axvspan(300,600,alpha=0.3,color="r",label="Stress")
        axes[i].axvspan(900,1200,alpha=0.3,color="b",label="Amusement")
    plt.legend()
    plt.xlabel("Time[s]")
    plt.grid()
    return plt

if __name__ == '__main__':
    path = r"Z:\theme\mental_stress\02.BiometricData\2019-11-20\teraki\opensignals_device2_2019-11-20_13-40-49.txt"
    plt = plot_raw(path,emotion_section={"Amusement":[300,600],"Stress":[900,1200]})
    #plt.show()
    plt.savefig(r"Z:\theme\mental_stress\04.Figure\raw\teraki_2019-11-20_13-40-49_RawData.png")