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

path = r"Z:\theme\mental_arithmetic\03.BiometricData\2019-12-05\tozyo\opensignals_device1_2019-12-05_16-02-47.txt"
df = biosignal_time_summary(path, duration=120,overlap=30)
df.to_excel(r"biosignal_tozyo_device1_2019-12-05_16-02-47.xlsx")