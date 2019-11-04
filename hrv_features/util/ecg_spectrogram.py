# Import libraries
import numpy as np
from scipy import signal,interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt

#諸元の設定
# FFTのサンプル数
N = 2**8
# FFTで用いるハミング窓
hammingWindow = np.hamming(N)
# 補間の周波数
fs = 4


# Get some data
nn = np.loadtxt(r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\RRI_teraki_2019-10-23.csv")

import detrending

tmStamp = np.cumsum(nn)
tmStamp -= tmStamp[0]
nn = detrending.detrend(nn,500)
#detrend
nn = nn - np.mean(nn)



# Plot spectrogram
fig, ax = plt.subplots() 
f, t, Sxx = signal.spectrogram(nn, nperseg=N,noverlap=N-1,detrend='linear')
pc = ax.pcolormesh(t, f, Sxx, norm=mpl.colors.LogNorm(vmin=Sxx.mean(), vmax=Sxx.max()), cmap='inferno')
#pc = ax.pcolormesh(t/fs, f, Sxx, norm=mpl.colors.LogNorm(vmin=1, vmax=10**4), cmap='inferno')

ax.set_ylabel('Frequency')
ax.set_xlabel('Time') 
fig.colorbar(pc)
plt.show()

## スペクトログラムを描画
#pxx, freq, bins, t = plt.specgram(nn_interpol, NFFT=N, Fs=fs, noverlap=N-10, window=hammingWindow)
## pxx, freq, bins, t = plt.specgram(nn_interpol, Fs=fsmmingWindow)
#plt.xlabel("time [second]")
#plt.ylabel("frequency [Hz]")
#plt.xlim(tmStamp[0]*0.001, tmStamp[-1]*0.001)
#plt.ylim(0, 0.5)
## x軸の目盛設定
#plt.xticks(np.arange(tmStamp[0]*0.001, tmStamp[-1]*0.001,100))
#plt.title("{}_video_{}".format(subject,id))
