# Import packages
import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Download data
df = pd.read_csv("https://raw.githubusercontent.com/neuropsychology/NeuroKit.py/master/examples/Bio/bio_100Hz.csv")
# Process the signals
bio = nk.bio_process(ecg=df["ECG"], rsp=df["RSP"], eda=df["EDA"], add=df["Photosensor"], sampling_rate=100)

nk.z_score(bio["df"][["ECG_Filtered", "RSP_Filtered", "RSA"]])[1000:2500].plot()
plt.show()