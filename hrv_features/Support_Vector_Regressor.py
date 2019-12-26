# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE,RFECV
from sklearn.neighbors import KNeighborsRegressor

from boruta import BorutaPy
from multiprocessing import cpu_count

# local packages
from biosignal_analysis import features_baseline
import Support_Vector_Classifier as ml

# 描画
import matplotlib.pyplot as plt

def model_tuning(df, target_label="Valence", type="number",model=KNeighborsRegressor()):
    targets = ml.get_targets(df, target_label,type)
    features = ml.get_features(df)

    gkf = ml.split_by_group(df)
    #penaltyとlossの組み合わせは三通り
    #              penalty     loss 
    # Standard  |    L2     |   L1
    # LossL2    |    L2     |   L2
    # PenaltyL1 |    L1     |   L2
    # LinearSVCの取りうるモデルパラメータを設定
    #C_range= np.logspace(-2, 2, 30)
    #param_grid = [{"penalty": ["l2"],"loss": ["hinge"],"dual": [True],"max_iter":[8000],
    #                'C': C_range, "tol":[1e-3]}, 
    #                {"penalty": ["l1"],"loss": ["squared_hinge"],"dual": [False],"max_iter":[8000],
    #                'C': C_range, "tol":[1e-3]}, 
    #                {"penalty": ["l2"],"loss": ["squared_hinge"],"dual": [True],"max_iter":[8000],
    #                'C': C_range, "tol":[1e-3]}]
    
    param_grid = {"n_neighbors":np.arange(1,20)}
    grid_clf = GridSearchCV(model, param_grid, cv=gkf)

    #モデル訓練
    # 本来は，訓練データとテストデータに分けてfitさせる
    grid_clf.fit(features, targets)

    # 結果を出力
    print("\n----- Grid Search Result-----")
    print("Best Parameters: {}".format(grid_clf.best_params_))
    print("Best Cross-Validation Score: {}".format(grid_clf.best_score_))

    return grid_clf.best_params_


if __name__ == "__main__":
    question_path = r"Z:\theme\mental_arithmetic\06.QuestionNaire\QuestionNaire_result.xlsx"
    dataset_path = r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_1.xlsx"
    dataset = ml.get_datasets(question_path,dataset_path,
                              normalization=True,emotion_filter=False)
    dataset.to_excel(r"C:\Users\akito\Desktop\test.xlsx")
    
    # info
    print("\n----- Strat Machine learning -----")
    print("Info: \n data num: {} \n feature num: {}".format(dataset.shape[0],dataset.shape[1]))
    print("\n feature label : \n {}".format(dataset.columns))
    print("\n emotion label : {}".format(dataset["emotion"].unique()))

    targets =  ml.get_targets(dataset, target_label="Valence", type="number")
    features = ml.get_features(dataset)


    from sklearn.linear_model import Ridge,Lasso
    X_train,X_test,y_train ,y_test = train_test_split(features,targets,random_state=42)
    ridge = Lasso().fit(X_train,y_train)
    print("Training set score: {}".format(ridge.score(X_train,y_train)))
    print("Test set score: {}".format(ridge.score(X_test,y_test)))


    # モデル構築
    #model_tuning(dataset,target_label="Valence",type="number")
    pass