# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
import eli5
from eli5.sklearn import PermutationImportance

def plot_importance(classifier, feature_names, top_features=2):
    dset = pd.DataFrame()
    dset['attr'] = feature_names
    dset['importance'] = classifier.coef_.ravel()

    dset = dset.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(16, 14))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)
    plt.show()

def plot_single_histgraph(dataset,features_label):
    df = pd.DataFrame(dataset.features, columns = dataset.features_label_list)
    df["emotion"] = dataset.targets
    sns.distplot(df[df["emotion"] == "Amusement"][features_label],kde=True,rug=True,color="r")
    sns.distplot(df[df["emotion"] == "Stress"][features_label],kde=True,rug=True,color="g")
    sns.distplot(df[df["emotion"] == "Neutral2"][features_label],kde=True,rug=True,color = "b")
    plt.show()

def plot_pairplot(dataset):
    df = pd.DataFrame(dataset.features, columns = dataset.features_label_list)
    df["emotion"] = dataset.targets
    sns.pairplot(df, hue="emotion")
    plt.show()

def plot_corr(dataset):
    df = pd.DataFrame(dataset.features, columns = dataset.features_label_list)
    fig, ax = plt.subplots(figsize=(12, 9)) 
    sns.heatmap(df.corr(), square=True, vmax=1, vmin=-1, center=0)
    plt.show()


def plot_permutation_importance(dataset,clf):
    perm = PermutationImportance(clf, random_state=1).fit(dataset.features, dataset.targets)
    df = pd.Series(perm.feature_importances_, index= dataset.features_label_list)
    df.plot.barh()
    plt.show()
 


if __name__ == '__main__':
    from src.data_processor import load_emotion_dataset
    dataset = load_emotion_dataset()

    #selected_features = ["fft_abs_lf", "fft_rel_vlf", "fft_log_vlf","lomb_peak_vlf",
    #                     "lomb_abs_vlf", "lomb_rel_vlf","tinn_n","tinn_m","sampen", "bvp_min",
    #                     "pathicData_mean","pathicData_log_mean"
    #                     ]
       
    #dataset.features = dataset.features[:,dataset.features_label_list.isin(selected_features)]
    #dataset.features_label_list = dataset.features_label_list[dataset.features_label_list.isin(selected_features)]

    plot_single_histgraph(dataset,features_label="pathicData_std")