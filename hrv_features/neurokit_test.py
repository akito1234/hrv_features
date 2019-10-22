# Import packages
import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from opensignalsreader import OpenSignalsReader
from biosppy.signals import eda


# Download data
path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-21\opensignals_201806130003_2019-10-21_15-16-48.txt"

# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)
eda2 = arc.signal('EDA')
eda.eda(eda2, sampling_rate=1000.0, show=True, min_amplitude=0.1)
plt.show()

## Process the resp signals
#processed_eda = nk.eda_process(eda)['EDA']

#plt.figure(figsize=(24,8))
#plt.plot(arc.t,eda)
#for onset, peak, recovery in zip(processed_eda['SCR_Onsets'],
#                                 processed_eda['SCR_Peaks_Indexes'],
#                                 processed_eda['SCR_Recovery_Indexes']):
#    plt.axvline(onset*0.001,color="r")
#    plt.axvline(peak*0.001,alpha=0.3,color="b")
##   plt.axvline(recovery*0.001,alpha=0.3,color="y")
#plt.show()


#eda_data  = np.c_[processed_eda['SCR_Onsets'],
#                   processed_eda['SCR_Peaks_Indexes'],
#                   processed_eda['SCR_Peaks_Amplitudes'],
#                   processed_eda['SCR_Recovery_Indexes']
#                   ]

#np.savetxt(r"C:\Users\akito\Desktop\eda_tohma.csv"
#            ,eda_data,delimiter=',')