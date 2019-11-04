# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy import signal,interpolate

# トレンド除去
def detrend(raw_rri, Lambda):
  """applies a detrending filter.
   
  This code is based on the following article "An advanced detrending method with application
  to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
  
  Parameters
  ----------
  rri: numpy.ndarray
    The rri where you want to remove the trend.
  Lambda: int
    The smoothing parameter.

  Returns
  ------- 
  filtered_rri: numpy.ndarray
    The detrended rri.
  
  """
  rri = resample_to_4Hz(raw_rri)
  rri_length = rri.shape[0]

  # observation matrix
  H = np.identity(rri_length) 

  # second-order difference matrix
  
  ones = np.ones(rri_length)
  minus_twos = -2*np.ones(rri_length)
  diags_data = np.array([ones, minus_twos, ones])
  diags_index = np.array([0, 1, 2])
  D = spdiags(diags_data, diags_index, (rri_length-2), rri_length).toarray()
  filtered_rri = np.dot((H - np.linalg.inv(H + (Lambda**2) * np.dot(D.T, D))), rri)
  return filtered_rri 

# RRIの補間処理
def resample_to_4Hz(rri,sample_rate=4.):
    tmStamp = np.cumsum(rri)
    tmStamp -= tmStamp[0]
    
    # 3次のスプライン補間
    rri_spline = interpolate.interp1d(tmStamp, rri, 'cubic')
    t_interpol = np.arange(tmStamp[0], tmStamp[-1], 1000./sample_rate)
    rri_interpol = rri_spline(t_interpol)
    return rri_interpol

def welch_method(rri,resampled=False,sample_rate=4.):
    if resampled:
        rri = resample_to_4Hz(rri,sample_rate=sample_rate)
    freq, power = signal.welch(rri, sample_rate)
    plt.plot(freq, power,'b')
    plt.xlim(0,0.50)
    

if __name__ == '__main__':
    path = r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\RRI_tohma_2019-10-14.csv"
    rri = np.loadtxt(path,delimiter=',')
    filtered = detrend(rri,Lambda= 500)
    np.savetxt(r"C:\Users\akito\Desktop\filtered_rri.txt",filtered)