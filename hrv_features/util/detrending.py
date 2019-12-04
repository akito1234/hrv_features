# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy import signal,interpolate

# トレンド除去
def detrend(rri, Lambda):
  """applies a detrending filter.
   
  This code is based on the following article "An advanced detrending method with application
  to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
  
  Parameters
  ----------
  rri: numpy.ndarray
    The rri where you want to remove the trend. 
    ***  This rri needs resampling  ***
  Lambda: int
    The smoothing parameter.

  Returns
  ------- 
  filtered_rri: numpy.ndarray
    The detrended rri.
  
  """
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
    dict = r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\ECG"
    path_list = ["RRI_kishida_2019-10-22.csv",
                "RRI_shizuya_2019-10-23.csv",
                "RRI_shibata_2019-10-28.csv",
                "RRI_teraki_2019-10-23.csv",
                "RRI_tohma_2019-10-21.csv"
                ]
    import os
    
    fig,axes = plt.subplots(len(path_list),2,sharex=True)
    for i,path in enumerate(path_list):
        abs_path = os.path.join(dict,path)
        rri = np.loadtxt(abs_path,delimiter=',')
        ts = np.cumsum(rri) * 0.001
        ts = ts - ts[0]

        filtered = detrend(rri,Lambda= 500)
        filtered_ts = np.arange(0,len(filtered)*0.25,step=0.25)
        axes[i,0].plot(ts,rri)
        axes[i,0].set_title(path)
        axes[i,0].set_ylabel('RRI[ms]')
        axes[i,0].plot(filtered_ts,(resample_to_4Hz(rri) - filtered),'r')


        axes[i,1].plot(filtered_ts,filtered)
        axes[i,1].set_title('filtered - '+path)
        axes[i,1].set_ylabel('RRI[ms]')

        for j in range(2):
            axes[i,j].axvspan(300,600,alpha=0.3,color="r",label="Stress")
            axes[i,j].axvspan(900,1200,alpha=0.3,color="b",label="Amusement")

    plt.tight_layout()
    plt.legend()
    fig.suptitle('Appling detrending method to RRI signal')
    plt.show()
    