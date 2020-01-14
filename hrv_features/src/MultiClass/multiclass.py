
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import OneHotEncoder

# Local Package
from src import config

dataset = pd.read_excel(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_1.xlsx")
# Neutral2以外のデータを取り出す
dataset = dataset.query("emotion != 'Neutral2'")

# 目標変数
target_label = dataset["emotion"]
targets = OneHotEncoder(sparse=False).fit_transform(target_label)

# 説明変数
features = dataset.drop(config.identical_parameter,axis=1).values

# ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=5)

# Apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(features, targets)

# Show results
#print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])