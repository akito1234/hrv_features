# Import Package
import biosppy.signals.tools as st
import numpy as np
import ledapy

# 前処理
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

# CDA法によるskin conducatance responseの算出
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

# 特徴量の算出
def scr_features(scr_data):
    eda_features = {}
    for label in ['sc','tonicData','pathicData']:
        scr_signal = scr_data[label]
        # ピークの振幅
        eda_features[label+'_mean'] = np.mean(scr_signal)
        eda_features[label+'_max']  = np.max(scr_signal)
        eda_features[label+'_std'] = np.std(scr_signal)

        # convert to log scale
        eda_features[label+'_log_mean'] = np.log10(1 + np.mean(scr_signal))
        eda_features[label+'_log_max']  = np.log10(1 + np.max(scr_signal))
    return eda_features

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from opensignalsreader import OpenSignalsReader

    path = r"C:\Users\akito\Desktop\test.txt"
    arc = OpenSignalsReader(path)

    scr_data = scr(arc.signal('EDA'),
                sampling_rate=sampling_rate,
                downsamp = downsamp)

    scr_data = scr_features(scr_data)
    
    for a in scr_data.keys():
        print(a,scr_data[a])
    
    #fig, axes = plt.subplots(3,1)
    #axes[0].plot(scr_data['ts'],scr_data['sc'])
    #axes[1].plot(scr_data['ts'],scr_data['pathicData'])
    #axes[2].plot(scr_data['ts'],scr_data['tonicData'])
    #plt.show()

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
