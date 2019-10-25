
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

    resp_data = signals.resp.resp(arc.signal('RESP'),show=False)
    axes[2].plot(resp_data['resp_rate_ts'],
                 resp_data['resp_rate'],'b')
    axes[2].set_ylabel('RESP[Hz]')
    axes[2].set_ylim(0,0.5)

    for i in range(3):
        axes[i].axvspan(300,600,alpha=0.3,color="r",label="Stress")
        axes[i].axvspan(900,1200,alpha=0.3,color="b",label="Amusement")

    plt.legend()
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

path = r"Z:\theme\mental_stress\03.Analysis\Analysis_Features\features_kishida_2019-10-22_120s_windows.xlsx"
plt = plot_hrv(path)
plt.savefig(r"C:\Users\akito\Desktop\features_kishida_2019-10-23.png")
