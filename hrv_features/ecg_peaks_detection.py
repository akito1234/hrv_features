from ecgdetectors import Detectors
from opensignalsreader import OpenSignalsReader
import numpy as np

path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-23\teraki\opensignals_dev_2019-10-23_16-59-10.txt"
#path = r"C:\Users\akito\Desktop\test2.txt"
# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)
detectors = Detectors(1000.)
result = detectors.pan_tompkins_detector(arc.signal("ECG"))
real_rri = np.loadtxt(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_BioSignal\ECG\RRI_teraki_2019-10-23.csv",delimiter=",")
real_ts = np.cumsum(real_rri)
#np.savetxt(r"C:\Users\akito\Desktop\teraki_pan_tompkins.csv",np.diff(result),delimiter=",")
import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,1)
axes[0].plot(result[1:],np.diff(result),"r",label = "pan_tompkins")
axes[0].plot(real_ts,real_rri,"b",label = "original")

axes[1].plot(arc.t,arc.signal("ECG"))
for peaks in result:
    axes[1].axvline(peaks*0.001, color='r')

for peaks in real_ts:
    axes[1].axvline(peaks*0.001, color='b')
plt.legend()
plt.show()
