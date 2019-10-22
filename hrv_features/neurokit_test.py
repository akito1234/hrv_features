# Import packages
import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from opensignalsreader import OpenSignalsReader

# Download data
path = r"\\Ts3400defc\共有フォルダ\theme\mental_stress\02.BiometricData\2019-10-22\kishida\opensignals_dev_2019-10-22_13-54-50.txt"

# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)
eda = arc.signal('EDA')

# Process the resp signals
onsets, peaks, amplitudes, recoveries = nk.eda_scr(eda)


plt.figure(figsize=(24,8))
plt.plot(arc.t,eda)
for onset, peak, amplitude, recovery in zip(onsets, peaks, amplitudes, recoveries):
    plt.axvline(onset*0.001,color="r")
    plt.axvline(peak*0.001,alpha=0.3,color="b")
    plt.axvline(recovery*0.001,alpha=0.3,color="y")
plt.show()


resp_data  = np.c_[np.array(onsets),
                   np.array( peaks),
                   np.array(amplitudes),
                   np.array(recoveries)
                   ]
np.savetxt(r"C:\Users\akito\Desktop\eda_kishida.csv"
            ,resp_data,delimiter=',')