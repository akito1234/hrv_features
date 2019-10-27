# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:58:24 2019

@author: akito
"""

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from biosppy import signals 
import frequency_domain as fd
import pyhrv.time_domain as td
import pyhrv.nonlinear as nl
import pyhrv.tools as tools
from opensignalsreader import OpenSignalsReader

# mpl_toolkitsのインストール
import importlib
importlib.import_module('mpl_toolkits.mplot3d').__path__
from mpl_toolkits.mplot3d import Axes3D


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-3, 3, 0.25)
y = np.arange(-3, 3, 0.25)
X, Y = np.meshgrid(x, y)
Z = np.sin(X)+ np.cos(Y)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X,Y,Z) #<---ここでplot

plt.show()


# Read OpenSignals file
path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-21\tohma\opensignals_201806130003_2019-10-21_15-16-48.txt"

# Read OpenSignals file and plot all signals
arc = OpenSignalsReader(path)
ecg = arc.signal('ECG')
# 心拍データからピークを取り出す
ecg_data = signals.ecg.ecg(signal= ecg , sampling_rate=1000.0, show=False)
fig,axes = plt.subplots(2,1,sharex=True)
axes[0].plot(ecg_data['heart_rate_ts'],
             ecg_data['filtered'][ecg_data['rpeaks']][1:])
axes[1].plot(ecg_data['heart_rate_ts'],ecg_data['heart_rate'])
plt.show()