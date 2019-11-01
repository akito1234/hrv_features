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



def detrend(signal, Lambda):
  """applies a detrending filter.
   
  This code is based on the following article "An advanced detrending method with application
  to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
  
  Parameters
  ----------
  signal: numpy.ndarray
    The signal where you want to remove the trend.
  Lambda: int
    The smoothing parameter.

  Returns
  ------- 
  filtered_signal: numpy.ndarray
    The detrended signal.
  
  """
  signal_length = signal.shape[0]

  # observation matrix
  # 単位行列を作成
  H = np.identity(signal_length) 

  # second-order difference matrix
  from scipy.sparse import spdiags

  ones = np.ones(signal_length)
  
  minus_twos = -2*np.ones(signal_length)
  diags_data = np.array([ones, minus_twos, ones])
  diags_index = np.array([0, 1, 2])
  D = spdiags(diags_data, diags_index, (signal_length-2), signal_length).toarray()
  filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda**2) * np.dot(D.T, D))), signal)
  return filtered_signal 




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
    axes[1].plot(eda_result['ts'], eda_result['src'])
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
    path = r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\RRI_tohma_2019-10-21.csv"
    path2 = r"C:\Users\akito\Desktop\test_filter.csv"
    nn = np.loadtxt(path,delimiter=',')
    kubios_filtered = np.loadtxt(path2,delimiter=',')


    filtered = detrend(nn,500)
    tmStamps = np.cumsum(nn)*0.001 #in seconds 
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(tmStamps,filtered,'r')
    plt.plot(tmStamps,(nn-filtered),'b')

    plt.show()
    #plt = plot_signal(path).savefig(r"C:\Users\akito\Desktop\stress\04.Figure\summary\kishida_2019-10-22.png")
