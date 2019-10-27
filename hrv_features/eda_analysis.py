# Import Package
from biosppy.signals import eda
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from opensignalsreader import OpenSignalsReader


def EDA_FEATURES(eda,section):
    eda_features = {}
    onsets = eda[0]
    peaks = eda[1]
    amplitude = eda[2]
    eda_filter = (peaks>= section[0]*1000) & (peaks <= section[1]*1000)
    # ピークの振幅
    eda_features['SCR_Amplitude_Mean'] = np.mean(amplitude[eda_filter])
    eda_features['SCR_Amplitude_Max']  = np.max(amplitude[eda_filter])

    # SCR_Latency
    SCR_Latency = onsets[eda_filter][1:] - onsets[eda_filter][:-1] 
    eda_features['SCR_Latency'] = np.mean(SCR_Latency)

    # SCR_PeakTime
    SCR_PeakTime = peaks[eda_filter][1:] - peaks[eda_filter][:-1] 
    eda_features['SCR_PeakTime'] = np.mean(SCR_PeakTime)

    # SCR_RiseTimes
    SCR_RiseTImes = peaks[eda_filter] - onsets[eda_filter]
    eda_features['SCR_RiseTImes'] = np.mean(SCR_RiseTImes)

    return eda_features

if __name__ == '__main__':
    path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-10-23\teraki\opensignals_dev_2019-10-23_16-59-10.txt"
    arc = OpenSignalsReader(path)
    #np.savetxt(r"C:\Users\akito\Downloads\sample_E4\EDA.csv",arc.signal('EDA'),
    #           delimiter = ",")
    eda_filtered = eda.eda(arc.signal('EDA'),show=False)
    eda_data = eda.basic_scr(eda_filtered['filtered'], sampling_rate=1000.0)
    fig,axs = plt.subplots(2,1,sharex=True)
    axs[0].plot(arc.t,eda_filtered['filtered'])
    for onset, peak in zip(eda_data['onsets'],eda_data['peaks']):
        axs[0].axvline(onset*0.001,color="r")
        axs[0].axvline(peak*0.001,color="b")
    onsets_amplitide = eda_filtered['filtered'][eda_data['onsets'].tolist()]
    axs[1].plot(eda_data['peaks']*0.001,eda_data['amplitudes']-onsets_amplitide)
    plt.show()

eda_output  = np.c_[eda_data['onsets'],eda_data['peaks'],(eda_data['amplitudes']-onsets_amplitide)]

np.savetxt(r"C:\Users\akito\Desktop\eda_tohma.csv"
            ,eda_output,delimiter=',')


    #for keys in EDA_FEATURES(eda_data):
    #    print(keys, EDA_FEATURES(eda_data)[keys])