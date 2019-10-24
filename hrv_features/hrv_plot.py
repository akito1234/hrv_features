
import pyhrv.frequency_domain as fd
import numpy as np
from opensignalsreader import OpenSignalsReader
from biosppy import signals
import matplotlib.pyplot as plt
import pandas as pd
# Import packages
import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns

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
    heart_rate_ts,heart_rate = signals.ecg.ecg(signal= arc.signal(['ECG']) , sampling_rate=1000.0, show=False)[5:7]


    fig,axes = plt.subplots(3,1,sharex=True,figsize = (16,9),subplot_kw=({"xticks":np.arange(0,1200,100)}) )
    axes[0].plot(heart_rate_ts,heart_rate,'b')
    axes[0].set_xlim(0,1200)
    axes[0].set_ylabel("HR[bpm]")

    axes[1].plot(arc.t,
                 arc.signal(['EDA'])
                 ,'b')
    axes[1].set_ylim(0,25)
    axes[1].set_ylabel('EDA[us]')


    axes[2].plot(arc.t,
                 arc.signal(['RESP'])
                 ,'b')
    axes[2].set_ylabel('RESP[%]')
    axes[2].set_ylim(-50,50)


    for i in range(3):
        axes[i].axvspan(300,600,alpha=0.3,color="r",label="Stress")
        axes[i].axvspan(900,1200,alpha=0.3,color="b",label="Amusement")

    plt.legend()
    plt.xlabel("Time[s]")
    plt.show()


def plot_hrv(path):
    df = pd.read_excel(path,index_col=0)
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(24,8))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() # 二つ目の軸を定義

    ax1.plot(df.index.values, df['fft_ratio'].values, color='black',label='LF/HF')
    ax1.set_ylabel('LF / HF [-]') # ラベルを設定
    ax1.set_xlabel("Time[s]")
    ax2.plot(df.index.values, df['fft_abs_hf'].values, color='red',label='HF')
    ax2.plot(df.index.values, df['fft_abs_lf'].values, color='blue',label='LF')
    ax2.set_ylabel('LF, HF [ms^2]') 

    # 塗りつぶしを指定
    plt.axvspan(300,600,alpha=0.3,color="r")
    plt.axvspan(900,1200,alpha=0.3,color="b",label="Amusement")
    # 凡例
    # グラフの本体設定時に、ラベルを手動で設定する必要があるのは、barplotのみ。plotは自動で設定される＞
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    # 凡例をまとめて出力する
    ax1.legend(handler1 + handler2, label1 + label2, loc=2, borderaxespad=0.)
    plt.grid()
    plt.title('Subject1 - 2st')
    plt.show()

path = r"\\Ts3400defc\共有フォルダ\theme\mental_stress\02.BiometricData\2019-10-23\teraki\opensignals_dev_2019-10-23_16-59-10.txt"
plot_signal(path)
