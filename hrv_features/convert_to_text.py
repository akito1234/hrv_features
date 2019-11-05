
import numpy as np

data = np.loadtxt(r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\ECG\RRI_tohma_2019-10-21.csv",delimiter=",")
data = data * 0.001
np.savetxt(r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\ECG\text\RRI_tohma_2019-10-21.txt",data )