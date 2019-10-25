# Import packages
import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from opensignalsreader import OpenSignalsReader

# Download data
path = r"C:\Users\akito\Desktop\test.txt"


#--呼吸--#
# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)
#processed_rsp = nk.rsp_process(arc.signal('RESP'), sampling_rate=1000)['RSP']

# Process the signals
# 呼吸周期のみを評価
rsp_data  = np.c_[np.array(processed_rsp['Cycles_Onsets'])[1:],
                  processed_rsp['Cycles_Length']]

# 呼吸ピークの解析
#Expiration_Onsets = np.array(processed_rsp['Expiration_Onsets'])
#Expiration_Length = Expiration_Onsets[:-1] - Expiration_Onsets[1:]
#rsp_data  = np.c_[Expiration_Onsets,Expiration_Length]

np.savetxt(r"C:\Users\akito\Desktop\eda_tohma.csv"
            ,rsp_data,delimiter=',')

# 皮膚コンダクタンス--#
arc = OpenSignalsReader(path)
processed_eda = nk.rsp_process(arc.signal('EDA'), sampling_rate=1000)['EDA']
#eda.eda(eda2, sampling_rate=1000.0, show=True, min_amplitude=0.1)
plt.show()

# Process the resp signals
processed_data = nk.eda_process(arc.signal('EDA'))

processed_eda = processed_data['EDA']
processed_filter =processed_data['df']
plt.figure(figsize=(24,8))
plt.plot(arc.t,processed_filter['EDA_Filtered'])
for onset, peak, recovery in zip(processed_eda['SCR_Onsets'],
                                 processed_eda['SCR_Peaks_Indexes'],
                                 processed_eda['SCR_Recovery_Indexes']):
    plt.axvline(onset*0.001,color="r")
    plt.axvline(peak*0.001,alpha=0.3,color="b")
    plt.axvline(recovery*0.001,alpha=0.3,color="y")
plt.show()


eda_data  = np.c_[processed_eda['SCR_Onsets'],
                  processed_eda['SCR_Peaks_Indexes'],
                  processed_eda['SCR_Peaks_Amplitudes'],
                  processed_eda['SCR_Recovery_Indexes']
                  ]

np.savetxt(r"C:\Users\akito\Desktop\eda_tohma.csv"
            ,eda_data,delimiter=',')

## 期間を指定する
#arc = OpenSignalsReader(path)
#bio = nk.bio_process(ecg=arc.signal('ECG'), rsp=arc.signal('RESP'), eda=arc.signal('EDA'), sampling_rate=1000)
#onsets = np.array([0,300,600,900])
#epochs = nk.create_epochs(bio["df"],onsets, duration=300)
#data = {}  # Initialize an empty dict
#for epoch_index in epochs:
#    data[epoch_index] = {}  # Initialize an empty dict for the current epoch
#    epoch = epochs[epoch_index]

#    # ECG
#    baseline = epoch["ECG_RR_Interval"].ix[-100:0].mean()  # Baseline
#    rr_max = epoch["ECG_RR_Interval"].ix[0:400].max()  # Maximum RR interval
#    data[epoch_index]["HRV_MaxRR"] = rr_max - baseline  # Corrected for baseline

#    # EDA - SCR
#    scr_max = epoch["SCR_Peaks"].ix[0:600].max()  # Maximum SCR peak
#    if np.isnan(scr_max):
#        scr_max = 0  # If no SCR, consider the magnitude, i.e.  that the value is 0
#    data[epoch_index]["SCR_Magnitude"] = scr_max


#data = pd.DataFrame.from_dict(data, orient="index")  # Convert to a dataframe
#data["Condition"] = ["Neutral","Stress",  "Neutral","Ammusement"]  # Add the conditions
##print(data)