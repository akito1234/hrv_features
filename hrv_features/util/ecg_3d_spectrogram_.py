import numpy as np
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PolyCollection
from mpl_toolkits import Axes3D


fs = 11240.
t = 10
time = np.arange(fs*t) / fs
frequency = 1000.
mysignal = np.sin(2.0 * np.pi * frequency * time)

nperseg = 2**14
noverlap = 2**13
f, t, Sxx = signal.spectrogram(mysignal, fs, nperseg=nperseg,noverlap=noverlap)

myfilter = (f>800) & (f<1200)

f = f[myfilter]
Sxx = Sxx[myfilter, ...]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(f[:, None], t[None, :], 10.0*np.log10(Sxx), cmap=cm.coolwarm)
plt.show()