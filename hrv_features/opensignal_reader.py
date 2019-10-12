# -*- coding: utf-8 -*-
"""
Created 2019.10.12 17:00

@author: akito
"""

# Import OpenSignalsReader
from opensignalsreader import OpenSignalsReader
from 

# Import BasePlugins
import matplotlib.pyplot as plt



# Read OpenSignals file
path = r"C:\Users\akito\Desktop\stress\02.BiometricData\opensignals_dev_2019-10-11_17-06-10.txt"

# Read OpenSignals file and plot all signals
acq = OpenSignalsReader(path)
acq.plot(['ECG', 'EDA','RESP'])

