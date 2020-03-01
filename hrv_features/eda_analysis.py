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
def scr(signal,sampling_rate=1000.0,downsamp = 4,plot=False):
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

    path = r"C:\Users\akito\Desktop\stress\02.BiometricData\2019-11-21\tohma\opensignals_device3_2019-11-21_16-54-54.txt"
    arc = OpenSignalsReader(path)

    scr_data = scr(arc.signal('EDA'))
    plt.plot(scr_data['ts'],scr_data['sc'])
    plt.plot(scr_data['ts'],scr_data['tonicData'])
    plt.plot(scr_data['ts'],scr_data['pathicData'])

    plt.show()
