
import numpy as np
sub = "RRI_tohma_2019-11-21_16-54-54"
data = np.loadtxt(r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\ECG\{}.csv".format(sub),delimiter=",")
data = data * 0.001
np.savetxt(r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\ECG\text\{}.txt".format(sub),data )