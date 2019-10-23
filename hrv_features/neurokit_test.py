# Import packages
import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from opensignalsreader import OpenSignalsReader

# Download data
path = r"C:\Users\akito\Desktop\test.txt"

# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)
processed_rsp = nk.rsp_process(arc.signal('RESP'), sampling_rate=1000)['RSP']

# Process the signals
rsp_data  = np.c_[processed_rsp['Cycles_Onsets'],
                  processed_rsp['Cycles_Length'],
                  processed_rsp['Expiration_Onsets']
                   ]

np.savetxt(r"C:\Users\akito\Desktop\eda_tohma.csv"
            ,rsp_data,delimiter=',')



#eda.eda(eda2, sampling_rate=1000.0, show=True, min_amplitude=0.1)
#plt.show()

# Process the resp signals
#processed_eda = nk.eda_process(eda_data)['EDA']

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