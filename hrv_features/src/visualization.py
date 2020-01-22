# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC


def plot_importance(classifier, feature_names, top_features=2):
    coef = classifier.coef_.ravel()

    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    plt.figure(figsize=(18, 7))
    colors = ['green' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1 + 2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
    plt.show()


if __name__ == '__main__':
    print(df.drop(['Outcome'], axis = 1).columns.values)

    trainedsvm = svm.LinearSVC().fit(X, Y)
    feature_plot(trainedsvm, df.drop(['Outcome'], axis = 1).columns.values)