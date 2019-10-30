# Import Package
import biosppy.signals.tools as st
import numpy as np
import ledapy

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

def eda_preprocess(signal,sampling_rate):
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    aux, _, _ = st.filter_signal(signal=signal,
                                 ftype='butter',
                                 band='lowpass',
                                 order=4,
                                 frequency=5,
                                 sampling_rate=sampling_rate)

    # smooth
    sm_size = int(0.75 * sampling_rate)
    filtered, _ = st.smoother(signal=aux,
                              kernel='boxzen',
                              size=sm_size,
                              mirror=True)
    return filtered

def scr(signal,sampling_rate=1000.,result_type='phasicdata',downsamp = 4):
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # apply preprocessing
    filtered = eda_preprocess(signal,sampling_rate)
    
    # caliculate parameters
    # 最適化された結果を出力する
    phasicdata = ledapy.runner.getResult(filtered, result_type, sampling_rate, downsample=downsamp, optimisation=2)

    # create time stamps
    length = len(phasicdata)
    T = (length-1) / downsamp
    ts = np.linspace(0, T, length, endpoint=True)

    return {'ts':ts,
            'src':phasicdata,
            'filtered':filtered}


if __name__ == '__main__':
    from opensignalsreader import OpenSignalsReader
    path = r"C:\Users\akito\Desktop\test.txt"
    arc = OpenSignalsReader(path)
    result = scr(arc.signal(['EDA']),sampling_rate=1000.)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(result['ts'], result['src'])
    plt.show()

#    eda_filtered = eda.eda(arc.signal('EDA'),show=False)
#    eda_data = eda.basic_scr(eda_filtered['filtered'], sampling_rate=1000.0)
#    fig,axs = plt.subplots(2,1,sharex=True)
#    axs[0].plot(arc.t,eda_filtered['filtered'])
#    for onset, peak in zip(eda_data['onsets'],eda_data['peaks']):
#        axs[0].axvline(onset*0.001,color="r")
#        axs[0].axvline(peak*0.001,color="b")
#    onsets_amplitide = eda_filtered['filtered'][eda_data['onsets'].tolist()]
#    axs[1].plot(eda_data['peaks']*0.001,eda_data['amplitudes']-onsets_amplitide)
#    plt.show()

#eda_output  = np.c_[eda_data['onsets'],eda_data['peaks'],(eda_data['amplitudes']-onsets_amplitide)]

#np.savetxt(r"C:\Users\akito\Desktop\eda_tohma.csv"
#            ,eda_output,delimiter=',')


#    #for keys in EDA_FEATURES(eda_data):
#    #    print(keys, EDA_FEATURES(eda_data)[keys])