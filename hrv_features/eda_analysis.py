# Import Package
import biosppy.signals.tools as st
import numpy as np
import ledapy

def scr_features(scr):
    eda_features = {}
    # ピークの振幅
    eda_features['scr_mean'] = np.mean(scr)
    eda_features['scr_max']  = np.max(scr)
    eda_features['scr_std'] = np.std(scr)

    # convert to log scale
    eda_features['scr_log_mean'] = np.log10(1 + np.mean(scr))
    eda_features['scr_log_max']  = np.log10(1 + np.max(scr))
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

def scr(signal,sampling_rate=1000.,downsamp = 4,plot=False):
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # apply preprocessing
    filtered = eda_preprocess(signal,sampling_rate)
    
    # caliculate parameters -> 最適化された結果を出力する
    sc, pathicData, tonicData = ledapy.runner.getResult(filtered, sampling_rate, downsample=downsamp, optimisation=2)

    # create time stamps
    length = len(sc)
    T = (length - 1) * downsamp / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    if plot:
        fig, axes = plt.subplots(3,1)
        axes[0].plot(scr_data['ts'],scr_data['sc'])
        axes[1].plot(scr_data['ts'],scr_data['pathicData'])
        axes[2].plot(scr_data['ts'],scr_data['tonicData'])
        plt.show()

    return {'ts':ts,
            'sc':sc,
            'pathicData':pathicData,
            'tonicData':tonicData}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from opensignalsreader import OpenSignalsReader
    path = r"Z:\theme\mental_stress\02.BiometricData\2019-10-23\shizuya\opensignals_dev_2019-10-23_14-09-52.txt"
    arc = OpenSignalsReader(path)
    scr_data = scr(arc.signal('EDA'))
    fig, axes = plt.subplots(3,1)
    axes[0].plot(scr_data['ts'],scr_data['sc'])
    axes[1].plot(scr_data['ts'],scr_data['pathicData'])
    axes[2].plot(scr_data['ts'],scr_data['tonicData'])
    plt.show()

    #path = r"Z:\theme\mental_stress\02.BiometricData\2019-10-28\shibata\opensignals_dev_2019-10-28_13-50-02.txt"
    #arc = OpenSignalsReader(path)
    #result = scr(arc.signal(['EDA']),result_type='phasicdriver')

    #np.savetxt(r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\EDA\SCR_shibata_2019-10-28.csv"
    #           ,np.c_[result['ts'],result['src']])

    #import matplotlib.pyplot as plt
    #fig,axs = plt.subplots(2,1,sharex=False)
    #axs[0].plot(result['ts'],result['src'])
    #axs[1].plot(arc.t,arc.signal(['EDA']))
    #plt.show()



    #import matplotlib.pyplot as plt
    #data = np.loadtxt(r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\EDA\SCR_kishida_2019-10-22.csv")
    #ts = data[:,0]
    #scr_signal = data[:,1]
    #scr_signal = scr_signal - scr_signal[ts <= 300].mean()


    ### SCR (skin conductance response)の1次微分を算出 
    ##scr_first_time_derivative = np.diff(scr_signal) / np.diff(ts)

    ## 微分した値を符号関数に変換する
    #scr_sgn = np.sign(scr_signal)

    #plt.plot(ts ,scr_signal)
    #plt.plot(ts,scr_sgn)
    #plt.show()
    
    #pass
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