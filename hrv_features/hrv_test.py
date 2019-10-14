from hrv.classical import frequency_domain
from hrv.io import read_from_text
import pyhrv.frequency_domain as fd
import numpy as np




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



path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_BioSignal\RRI_kishida_2019-10-11.csv"


rri = np.loadtxt(path,delimiter=',')
np.savetxt(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_BioSignal\RRI_kishida_test.csv"
           ,detrend(rri,500)
           ,delimiter= ",")




#results = frequency_domain(
#    rri=rri,
#    fs=4.0,
#    method='welch',
#    interp_method='cubic',
#    detrend='linear'
#)

#print("***********************************************")
#print("HRV analysis result")
#print("***********************************************")
#for key in results.keys():
#    print(key, results[key])


#print("***********************************************")
#print("pyHRV analysis result")
#print("***********************************************")

#results = fd.welch_psd(rri, nfft=2**10)
#for key in results.keys():
#    print(key, results[key])