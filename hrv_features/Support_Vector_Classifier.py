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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE,RFECV
from sklearn.neighbors import KNeighborsRegressor

from boruta import BorutaPy
from multiprocessing import cpu_count

# local packages
from biosignal_analysis import features_baseline

# 描画
import matplotlib.pyplot as plt




##-----------------------------------
## データセットの作成
##-----------------------------------
def get_datasets(question_path,dataset_path, remove_label=["Neutral2"],
                 normalization =True,emotion_filter=False,filter_type= "both"):

    # アンケート結果を取得する
    questionnaire = pd.read_excel(question_path,sheet_name="questionnaire",usecols=[i for i in range(13)])
    questionnaire = preprocessing_questionnaire(questionnaire,emotion_filter,filter_type)


    # 特徴量を取得する
    features = pd.read_excel(dataset_path)
    features = features[~features["emotion"].isin(remove_label)]
    # 不要な特徴量を削除
    features =  features.drop(["nni_diff_min"],axis=1)
    if normalization:
        # 正規化 (個人差補正)
        features = features_baseline(features,emotion_state=['Stress','Amusement'],baseline='Neutral1',
                                     identical_parameter = ['id','emotion','user','date','path_name'])

    dataset = pd.merge(questionnaire, features)
    return dataset


##-----------------------------------
## 前処理
##-----------------------------------
def preprocessing_questionnaire(questionnaire,emotion_filter,filter_type):
    # Flag処理
    questionnaire = questionnaire.query("is_bad == 1")
    # Affect Gridを拡張
    questionnaire = extension_affectgrid(questionnaire)
    # 主観評価用フィルタ
    if emotion_filter:
        questionnaire = emotion_label_filter(questionnaire, filter_type)
    #不要な項目を削除
    questionnaire = questionnaire.drop(["id","exp_id","trans_Emotion_1","trans_Emotion_2","Film","is_bad"] ,axis=1)
    return questionnaire



# アンケート結果に基づいたデータの選定
def emotion_label_filter(qna,filter_type):
    '''
    Ggross and levenson	
    1	楽しさ
    2	興味
    3	幸福
    4	怒り
    5	嫌悪
    6	軽蔑
    7	恐怖
    8	悲しみ
    9	驚き
    10	満足
    11	安心
    12	苦しみ
    13	混乱
    14	困惑
    15	緊張
    16	中立
    中立はamusementとstress
    '''

    df_amusement = qna.query("emotion == 'Amusement'")
    df_stress = qna.query("emotion == 'Stress'")
    print("\n---------Selection of data by subjective evaluation---------")
    print("\nNumber of original data")
    print(" Stress data : {}\n Amusement data : {}\n Sum data : {}".format(df_stress.shape[0],
                                                                         df_amusement.shape[0],
                                                                         (df_stress.shape[0]+df_amusement.shape[0])))
    # Affect Gridによるフィルタ
    # Amusementは第一象限，Stressは第二象限
    if filter_type == "both" or filter_type == "affect_grid":
        df_amusement = df_amusement.query('Arousal >= 4 & Valence > 4')
        df_stress = df_stress.query('Arousal >= 4 & Valence < 4')
    
    # 感情ラベルによるフィルタ
    if filter_type == "both" or filter_type == "emotion_label":
        df_amusement = df_amusement.query('Emotion_1 in (1, 2, 3, 9, 10, 11, 15)')
        df_stress = df_stress.query('Emotion_1 in (4, 5, 6, 7, 8, 12, 13, 14)')
    
    # 出力
    print("\nNumber of selected data")
    print(" Stress data : {}\n Amusement data : {}\n Sum data : {}".format(df_stress.shape[0],
                                                                         df_amusement.shape[0],
                                                                         (df_stress.shape[0]+df_amusement.shape[0])))
    
    print("\ndata filter result by subject")
    print("\nbefore")
    print("{}".format(qna["user"].value_counts()))
    print("\nafter")
    result = pd.concat([df_amusement,df_stress],axis = 0)
    print("{}".format(result["user"].value_counts()))
    return result


##------------------------------------
##　ターゲット処理
##------------------------------------
# データセットからターゲットを抽出
def get_targets(df, target_label = "emotion",type="label"):
    if type == "label":
        # 0 1 にする
        le = preprocessing.LabelEncoder().fit(df[target_label].unique())
        targets = le.transform(df[target_label])
        print("Unique : {}".format(df[target_label].unique()))
        print("Transform : {}".format(le.transform(df[target_label].unique())))

    elif type == "multilabel":
        # pattern 1
        targets = df["emotion"].where((df['emotion'] == 'Stress') & (df['Valence'].isin([1,2])),"StH")
        targets = df["emotion"].where((df['emotion'] == 'Stress') & (df['Valence'].isin([3,4])),"StL") 
        print(targets)
        targets = df["emotion"].where((df['emotion'] == 'Amusement') & (df['Valence'].isin([4,5]) ),"AmH")
        targets = df["emotion"].where((df['emotion'] == 'Amusement') & (df['Valence'].isin([6,7]) ),"AmL")
        
        #df["emotion"].where( df["Valence"] == 4 ,"Neutral",inplace=True)
        targets = pd.get_dummies(df["emotion"]).values

    elif type == "number":
        targets = df[target_label].values
    else:
        return 0
    
    return targets

# データセットから特徴量を抽出
# 最終的には，ここで特徴量選択を行う
def get_features(df,selected_feature = None,drop_features = ['id','emotion','user','date','path_name','Valence','Arousal',
                                     'Emotion_1','Emotion_2',"angle","strength"],scale=True):
    df_features = df.drop(drop_features, axis=1)
    # 使う特徴量選択
    if selected_feature is not None:
        df_features = df_features[selected_feature]
    
    # スケール調整
    # 平均値を0,標準偏差を1 
    if scale:
        features = preprocessing.StandardScaler().fit_transform(df_features)
    else:
        features = df_features.values
    return features


##--------------------------
## モデル作成
##--------------------------
# 特徴量選択
def feature_selection(dataset,show=False):
    drop_label = ['id','emotion','user','date','path_name','Valence','Arousal',
                                    'Emotion_1','Emotion_2',"angle","strength"]

    targets = get_targets(dataset, target_label = "emotion",type="multilabel")
    features = get_features(dataset,scale=False)
    gkf = split_by_group(dataset)

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=int(cpu_count()/2), max_depth=7)

    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, two_step=False, random_state=42,max_iter=100)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(features, targets)

    # 出力
    column_list = dataset.columns.drop(drop_label).tolist()
    print('\n Initial features: {}'.format(column_list))
    # number of selected features
    print('\n Number of select feature: {}'.format(feat_selector.n_features_))

    print ('\n Top %d features:' % feat_selector.n_features_)
    feature_df = pd.DataFrame(column_list, columns=['features'])
    feature_df['rank']=feat_selector.ranking_
    feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
    print (feature_df.head(feat_selector.n_features_))

    # check ranking of features
    print ('\n Feature ranking:')
    print (feat_selector.ranking_)

    # 特徴量後のデータセット作成
    selected = dataset.drop(drop_label, axis=1).columns[feat_selector.support_]
    #selected = dataset.drop(drop_label, axis=1).columns[feat_selector.support_]
    selected_dataset = dataset[selected]
    result = pd.concat([dataset[drop_label],selected_dataset], axis=1)

    if show:
        ## 可視化
        mask = feat_selector.support_
        # マスクを可視化する．黒が選択された特徴量
        plt.matshow(mask.reshape(1, -1), cmap='gray_r')
        plt.tick_params(labelleft = 'off')
        plt.xlabel('Sample Index', fontsize=15)
        plt.show()

    return selected

# グループ分け
def split_by_group(dataset):
    targets = get_targets(dataset, target_label = "emotion",type="multilabel")
    features = get_features(dataset)

    # 被験者ごとにグループ分け
    le = preprocessing.LabelEncoder()
    group = dataset["user"]
    unique_subject = le.fit(group.unique())
    subjects = le.transform(group)
    gkf = list(GroupKFold(n_splits=len(group.unique())).split(features,targets,subjects))
    
    #　出力
    print(group.unique())
    print(le.transform(group.unique()))
    return gkf

# 線形サポートベクタを用いて最適なパラメータを推定する
def model_tuning(dataset, selected_feature =None, target_label = "emotion", type="label"):
    targets = get_targets(dataset, target_label = "emotion",type="multilabel")
    features = get_features(dataset,selected_feature)
    gkf = split_by_group(dataset)
    
    #penaltyとlossの組み合わせは三通り
    #              penalty     loss 
    # Standard  |    L2     |   L1
    # LossL2    |    L2     |   L2
    # PenaltyL1 |    L1     |   L2
    # LinearSVCの取りうるモデルパラメータを設定
    C_range= np.logspace(-1, 2, 20)
    #param_grid = [{"penalty": ["l2"],"loss": ["hinge"],"dual": [True],"max_iter":[12000],
    #                'C': C_range, "tol":[1e-3]}, 
    #                {"penalty": ["l1"],"loss": ["squared_hinge"],"dual": [False],"max_iter":[12000],
    #                'C': C_range, "tol":[1e-3]}, 
    #                {"penalty": ["l2"],"loss": ["squared_hinge"],"dual": [True],"max_iter":[12000],
    #                'C': C_range, "tol":[1e-3]}]
    param_grid = {
          'estimator__C': [0.5, 1.0, 1.5],
          'estimator__tol': [1e-3, 1e-4, 1e-5],
          }
    clf = OneVsRestClassifier(LinearSVC())
    grid_clf = GridSearchCV(clf, param_grid, cv=gkf)

    #モデル訓練
    # 本来は，訓練データとテストデータに分けてfitさせる
    grid_clf.fit(features, targets)

    # 結果を出力
    print("\n----- Grid Search Result-----")
    print("Best Parameters: {}".format(grid_clf.best_params_))
    print("Best Cross-Validation Score: {}".format(grid_clf.best_score_))

    return grid_clf.best_estimator_


##---------------------------
## 可視化
##---------------------------
def plot_2d_separator():
    plt.figure(figsize=(12,7))
    plt.title("emotion recognition plot")
    plt.scatter(features_scaled[(targets==0),0],features_scaled[(targets==0),1])
    plt.scatter(features_scaled[(targets==1),0],features_scaled[(targets==1),1])

    clf = linear_SVM.fit(features_scaled,targets)

    # 可視化の準備
    xmin, xmax, ymin, ymax = (features_scaled[:,0].min()-1, features_scaled[:,0].max()+1,
                                features_scaled[:,1].min()-1, features_scaled[:,1].max()+1)    
    x_ = np.arange(xmin, xmax, 0.01)
    y_ = np.arange(ymin, ymax, 0.01)
    xx, yy = np.meshgrid(x_, y_)

    # 予測
    zz = clf.predict(np.stack([xx.ravel(), yy.ravel()], axis=1)
                    ).reshape(xx.shape)

    # 可視化
    plt.pcolormesh(xx, yy, zz, cmap="winter", alpha=0.1, shading="gouraud")
    plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=targets, edgecolors='k', cmap="winter")
    plt.show()


if __name__ =="__main__":
    #  # Valenceは残す
    #df = dataset.drop(['id','emotion','user','date','path_name','Arousal',
    #                          'Emotion_1','Emotion_2'],axis=1)
    #res=df.corr().to_excel(r"C:\Users\akito\Desktop\cor_python.xlsx")   # pandasのDataFrameに格納される
    #print(res)

    question_path = r"C:\Users\akito\Desktop\stress\05.QuestionNaire\QuestionNaire_result.xlsx"
    dataset_path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets_1.xlsx"
    dataset = get_datasets(question_path,dataset_path,
                           normalization=True,emotion_filter=False)
    
    dataset.to_excel(r"C:\Users\akito\Desktop\test.xlsx") 
    # info
    print("\n----- Strat Machine learning -----")
    print("Info: \n data num: {} \n feature num: {}".format(dataset.shape[0],dataset.shape[1]))
    print("\n feature label : \n {}".format(dataset.columns))
    print("\n emotion label : {}".format(dataset["emotion"].unique()))
    
    print("\n----- Model Tuning -----")
    # モデル作成
    targets = get_targets(dataset, target_label = "emotion",type="multilabel")
    np.savetxt(r"C:\Users\akito\Desktop\2019-01-09.csv",targets,delimiter=","
               )
    #features = get_features(dataset,scale=True)
    pass

    # 特徴量選択
    #selected_feature = feature_selection(dataset)
    
    # GridSearch
    # 精度検証
    #coef_2 = model_tuning(dataset)

    #print("\n特徴量選択後")
    # 相関ある特徴量と，ありそうな特徴量を使用
    # 引数を特徴量にしたい
    best_clf = model_tuning(dataset,selected_feature=None)

    # 精度検証
    gkf = split_by_group(dataset)
    targets = get_targets(dataset, target_label = "emotion",type="multilabel")
    features = get_features(dataset)
    score_result = cross_val_score(best_clf,features, targets, cv=gkf)
    print("Cross-Varidation score: \n{}".format(score_result))
    print("Cross-Varidation score mean: \n {}".format(score_result.mean()))
    #np.savetxt(r"Z:\00_個人用\東間\02.discussion\20191226\score_normalization_false",score_result,delimiter=",")
    
    
    ## モデル作成
    #print("------Model Accuracy------")
    #test_dataset = pd.read_excel(test_dataset_path,index_col=0)
    ## Neutral(300s)のデータを差分する
    
    ## スケールの調節していない
    #test_dataset = test_dataset.iloc[1:, :] - test_dataset.iloc[0, :]
    
    #predict_result = best_clf.predict(test_dataset[selected_feature])
    #print(predict_result)
    #np.savetxt(r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\predict_result2.csv",
    #           predict_result,delimiter=",")

    ## モデルを保存する
    ##filename = 'finalized_model.sav'
    ##pickle.dump(model, open(filename, 'wb'))