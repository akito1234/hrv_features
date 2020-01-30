# -*- coding: utf-8 -*-

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os
from biosppy import signals

# Import local packages
from  biosinal_summary import *

path = r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-11\kaneko\opensignals_device1_2019-12-11_16-17-56.txt"
df = biosignal_time_summary(path, duration=300,overlap=30)
df.to_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features_TimeVaries\biosignal_kaneko_device1_2019-12-11_16-17-56.xlsx")