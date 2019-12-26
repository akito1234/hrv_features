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

# 描画
import matplotlib.pyplot as plt




##-----------------------------------
## データセットの作成
##-----------------------------------
def get_datasets(question_path,dataset_path, remove_label=["Neutral2"],
                 normalization =True,emotion_filter=False):

    # アンケート結果を取得する
    questionnaire = pd.read_excel(question_path,sheet_name="questionnaire",usecols=[i for i in range(12)])
    questionnaire = extension_affectgrid(questionnaire)
    # 主観評価によるデータの選定を行う
    if emotion_filter:
        questionnaire = emotion_label_filter(questionnaire,emotion_label=True,affect_grid=True)

    #不要な項目を削除
    questionnaire = questionnaire.drop(["id","exp_id","trans_Emotion_1","trans_Emotion_2","Film"] ,axis=1)

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
# アンケート結果によるフィルタ
def extension_affectgrid(questionnaire):
    # AFFECT GRIDの感情の強さと，方向を決める
    _arousal = questionnaire["Arousal"].values
    _valence = questionnaire["Valence"].values
    questionnaire["strength"] = np.sqrt( np.square(_arousal) + np.square(_valence) )
    questionnaire["angle"]  = np.arctan(_arousal/ _valence)
    return questionnaire

# アンケート結果に基づいたデータの選定
def emotion_label_filter(qna,emotion_label=True,affect_grid=True):
    #    Gross And Levenson	
    #1	楽しさ
    #2	興味
    #3	幸福
    #4	怒り
    #5	嫌悪
    #6	軽蔑
    #7	恐怖
    #8	悲しみ
    #9	驚き
    #10	満足
    #11	安心
    #12	苦しみ
    #13	混乱
    #14	困惑
    #15	緊張
    #16	中立
    # 中立はAmusementとStress

    org_df_amusement = qna.query("emotion == 'Amusement'")
    org_df_stress = qna.query("emotion == 'Stress'")

    # Affect Gridによるフィルタ
    # Amusementは第一象限，Stressは第二象限
    if affect_grid:
        df_amusement = org_df_amusement.query('Arousal >= 4 & Valence > 4')
        df_stress = org_df_stress.query('Arousal >= 4 & Valence < 4')
    
    # 感情ラベルによるフィルタ
    if emotion_label:
        df_amusement = df_amusement.query('Emotion_1 in (1, 2, 3, 9, 10, 11, 15)')
        df_stress = df_stress.query('Emotion_1 in (4, 5, 6, 7, 8, 12, 13, 14)')
    
    # 出力
    print("\n---------Selection of data by subjective evaluation---------")
    print("\nNumber of original data")
    print(" Stress data : {}\n Amusement data : {}\n Sum data : {}".format(org_df_stress.shape[0],
                                                                         org_df_amusement.shape[0],
                                                                         (org_df_stress.shape[0]+org_df_amusement.shape[0])))
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


def emotion_info(questionnaire):
    
    pass

# Affect Grid の拡張
def Extension_AffectGrid():
    # 感情の強さ
    # 軸を45度変えるなど
    pass

# データセットからターゲットを抽出
def get_targets(df, target_label = "emotion",type="label"):
    if type == "label":
        # 0 1 にする
        targets = preprocessing.LabelEncoder().fit_transform(df[target_label])
    elif type == "number":
        targets = df[target_label].values
    else:
        return 0
    
    return targets

# データセットから特徴量を抽出
# 最終的には，ここで特徴量選択を行う
def get_features(df,drop_features = ['id','emotion','user','date','path_name','Valence','Arousal',
                                     'Emotion_1','Emotion_2',"angle","strength"],scale=True):
    df_features = df.drop(drop_features, axis=1)
    # スケール調整
    # 平均値を0,標準偏差を1 
    if scale:
        features = preprocessing.StandardScaler().fit_transform(df_features)
    else:
        features = df_features.values
    return features


##--------------------------
## モデル構築
##--------------------------
# 線形サポートベクタを用いて最適なパラメータを推定する
def model_tuning(dataset, target_label = "emotion", type="label",model=LinearSVC()):
    targets = get_targets(dataset, target_label = "emotion",type="label")
    features = get_features(dataset)
    gkf = split_by_group(dataset)
    #penaltyとlossの組み合わせは三通り
    #              penalty     loss 
    # Standard  |    L2     |   L1
    # LossL2    |    L2     |   L2
    # PenaltyL1 |    L1     |   L2
    # LinearSVCの取りうるモデルパラメータを設定
    C_range= np.logspace(-1, 2, 20)
    param_grid = [{"penalty": ["l2"],"loss": ["hinge"],"dual": [True],"max_iter":[12000],
                    'C': C_range, "tol":[1e-3]}, 
                    {"penalty": ["l1"],"loss": ["squared_hinge"],"dual": [False],"max_iter":[12000],
                    'C': C_range, "tol":[1e-3]}, 
                    {"penalty": ["l2"],"loss": ["squared_hinge"],"dual": [True],"max_iter":[12000],
                    'C': C_range, "tol":[1e-3]}]

    grid_clf = GridSearchCV(model, param_grid, cv=gkf)

    #モデル訓練
    # 本来は，訓練データとテストデータに分けてfitさせる
    grid_clf.fit(features, targets)

    # 結果を出力
    print("\n----- Grid Search Result-----")
    print("Best Parameters: {}".format(grid_clf.best_params_))
    print("Best Cross-Validation Score: {}".format(grid_clf.best_score_))

    return grid_clf.best_estimator_

# グループ分け
def split_by_group(dataset):
    targets = get_targets(dataset, target_label = "emotion",type="label")
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

# 特徴量選択
def feature_selection(dataset,show=False):
    drop_label = ['id','emotion','user','date','path_name','Valence','Arousal',
                                    'Emotion_1','Emotion_2',"angle","strength"]

    targets = get_targets(dataset, target_label = "emotion",type="label")
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

    return result

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

## ---------------------------------
## モデル定義
##----------------------------------
## KNN
#knn = KNeighborsClassifier(n_neighbors=6)
## linear svm 
#linear_SVM = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-3, C=2.,max_iter=2000)

## ---------------------------------
## モデル評価
##----------------------------------
## 交差検証
## 層化K分割交差検証
#kfold = StratifiedKFold(n_splits=6, shuffle=True,random_state=0)
## 1つ抜き交差検証
#loo = LeaveOneOut()
## グループ付き交差検証
#le = preprocessing.LabelEncoder()
#a = le.fit(df["subject"].unique())
#print(df["subject"].unique())
#print(le.transform(df["subject"].unique()))
#subjects = le.transform(df["subject"])



## 交差検証で，スケール合わせる方法が分からない
#features_scaled = scaler.transform(features)
#scores = cross_val_score(linear_SVM, features_scaled,targets, 
#                         subjects,cv=GroupKFold(n_splits=10) 
#                         )

#print("Cross-Varidation score: \n{}".format(scores))
#print("Cross-Varidation score mean: \n {}".format(scores.mean()))
#clf = linear_SVM.fit(features_scaled,targets)


if __name__ =="__main__":
    #  # Valenceは残す
    #df = dataset.drop(['id','emotion','user','date','path_name','Arousal',
    #                          'Emotion_1','Emotion_2'],axis=1)
    #res=df.corr().to_excel(r"C:\Users\akito\Desktop\cor_python.xlsx")   # pandasのDataFrameに格納される
    #print(res)


    question_path = r"Z:\theme\mental_arithmetic\06.QuestionNaire\QuestionNaire_result.xlsx"
    dataset_path = r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_1.xlsx"
    dataset = get_datasets(question_path,dataset_path,
                           normalization=False,emotion_filter=True)
    
    dataset.to_excel(r"C:\Users\akito\Desktop\test.xlsx") 
    # info
    print("\n----- Strat Machine learning -----")
    print("Info: \n data num: {} \n feature num: {}".format(dataset.shape[0],dataset.shape[1]))
    print("\n feature label : \n {}".format(dataset.columns))
    print("\n emotion label : {}".format(dataset["emotion"].unique()))
    
    print("\n----- Model Tuning -----")
    # モデル構築

    # 特徴量選択
    dataset = feature_selection(dataset)
    
    # GridSearch
    # 精度検証
    #coef_2 = model_tuning(dataset)

    print("\n特徴量選択後")
    # 相関ある特徴量と，ありそうな特徴量を使用
    best_clf = model_tuning(dataset)

    # 精度検証
    gkf = split_by_group(dataset)
    targets = get_targets(dataset, target_label = "emotion",type="label")
    features = get_features(dataset)
    score_result = cross_val_score(best_clf,features, targets, cv=gkf)
    print("Cross-Varidation score: \n{}".format(score_result))
    print("Cross-Varidation score mean: \n {}".format(score_result.mean()))
    np.savetxt(r"Z:\00_個人用\東間\02.discussion\20191226\score_normalization_false",score_result,delimiter=",")
    # 可視化

