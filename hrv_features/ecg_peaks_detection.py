from ecgdetectors import Detectors
from opensignalsreader import OpenSignalsReader
import numpy as np

path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-11-21\kaneko\opensignals_201806130003_2019-11-21_14-58-59.txt"

# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)
detectors = Detectors(1000.)

result = detectors.pan_tompkins_detector(arc.signal("ECG"))
np.savetxt(r"C:\Users\akito\Desktop\kishida_2019-11-21_16-00-52_pan_tompkins.csv",np.diff(result),delimiter=",")

import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,1)
axes[0].plot(result[1:],np.diff(result),"r",label = "pan_tompkins")
#axes[0].plot(real_ts,real_rri,"b",label = "original")

axes[1].plot(arc.t,arc.signal("ECG"))
for peaks in result:
    axes[1].axvline(peaks*0.001, color='r')

#for peaks in real_ts:
#    axes[1].axvline(peaks*0.001, color='b')
plt.legend()
plt.show()
