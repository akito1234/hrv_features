from hrv.classical import frequency_domain
from hrv.io import read_from_text
import pyhrv.frequency_domain as fd
import numpy as np
from opensignalsreader import OpenSignalsReader
from biosppy import signals
import matplotlib.pyplot as plt


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


path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-11\tohma\opensignals_dev_2019-10-11_17-29-23.txt"
arc = OpenSignalsReader(path)

# 心拍データからピークを取り出す
heart_rate_ts,heart_rate = signals.ecg.ecg(signal= arc.signal(['ECG']) , sampling_rate=1000.0, show=False)[5:7]


fig,axes = plt.subplots(3,1,sharex=True,figsize = (16,9),subplot_kw=({"xticks":np.arange(0,900,100)}) )
axes[0].plot(heart_rate_ts,heart_rate,'b')
axes[0].set_xlim(0,900)
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


plt.legend()
plt.xlabel("Time[s]")
plt.show()

 
