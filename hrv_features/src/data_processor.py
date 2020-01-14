import numpy as np
import pandas as pd
import os, glob
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import cpu_count
import src.config as config





class EmotionRecognition:
    #感情クラス
    #前処理全般
    def __init__(self, features_path,question_path
                 ,normalization=True,emotion_filter=True,filter_type="both"
                 ):
        '''
        感情判別用のデータセット作成

        '''
        self.targets = None
        self.targets_name = "emotion"
        self.targets_user = None
        self.features = None
        self.features_label_list = None
        self.identical_parameter = config.identical_parameter
        self.remove_features_label = config.remove_features_label
        self.emotion_filter = emotion_filter
        self.filter_type = filter_type
        self.bool_normalization = normalization
        self.emotion_baseline = "Neutral1"
        #self.emotion_state = ['Stress','Amusement','Neutral2']
        self.emotion_state = ['Stress','Amusement']
        self.individual_parameter = config.individual_parameter

       
        
        # 特徴量取得
        if features_path is not None:
            # 主観評価結果取得
            emotion_label = Emotion_Label(question_path,emotion_filter,filter_type)
            
            self._read_dataset(features_path)
            self._set_parameter(emotion_label)
            
            print("---------------------")
            print("Project Info\n")
            print(" normalization :{}".format(normalization))
            print(" emotion filter :{}".format(emotion_filter))
            print(" individual_parameter :{}\n".format(config.individual_parameter))
            print("---------------------")

    #キー(インスタンス変数)を取得するメソッド
    def keys(self):
        print("[targets, target_names, features]")

    def _set_parameter(self,emotion_label):
        # オプション設定
        # マージ，余計なデータを省く
        # Neutralを使う場合の処理は未実装
        dataset = pd.merge(emotion_label.questionnaire, self.features, on=["user","date","emotion"])
        # Dataframe をNumpy形式に変換する
        self.targets = dataset[self.targets_name].values
        self.targets_user = dataset["user"].values
        self.features_label_list = self.features.drop(self.identical_parameter,axis=1).columns
        self.features = dataset[self.features_label_list].values

    def _read_dataset(self,features_path):
        self.features = pd.read_excel(features_path)
        # 不要な特徴量を削除
        self.features = self.features.drop(["nni_diff_min"],axis=1)
        if self.bool_normalization:
            # 正規化 (個人差補正)
            self.normalization()

    def normalization(self,emotion_state=['Stress','Ammusement','Neutral2'],
                          baseline='Neutral1'):
        # 個人差の補正 (感情ごとの特徴量からベースラインを引く)
        if self.features is None:
            raise ValueError("Features Dataset does not found...")

        df_summary = pd.DataFrame([], columns=self.features.columns)
        groupby_features = self.features.groupby(["user","date"])
        for item in groupby_features:
            df_item = self.individual_difference_correction(item)
            df_summary = df_summary.append(df_item,ignore_index=True,sort=False)
        
        self.features = df_summary
    
    def individual_difference_correction(self, item):
        # 個人差補正(ベースライン差分)
        # ベースラインを取得
        baseline = item[1][ item[1]['emotion']  == self.emotion_baseline]
        df_baseline = baseline.drop(self.identical_parameter, axis=1)
        
        df_result = pd.DataFrame([],columns=baseline.columns)
        

        for state in self.emotion_state:
            emotion = item[1][ item[1]['emotion']  == state]
            if emotion.empty:
                print("Skip ... {}".format(state))
                continue;
            # キーラベルを別にする
            df_emotion = emotion.drop(self.identical_parameter, axis=1)
            identical_df = emotion[self.identical_parameter]

            # 個人差補正
            if self.individual_parameter == "diff":
                detrend_features = (df_emotion - df_baseline.values)
            elif self.individual_parameter == "ratio":
                detrend_features = (df_emotion / df_baseline.values)
            else:
                break
            
            result_item = pd.concat([identical_df,detrend_features], axis=1,sort=False)
            df_result = df_result.append(result_item,ignore_index=True,sort=False)

        return df_result



# 主観評価の処理クラス
class Emotion_Label:
    def __init__(self, questionnaire_path,emotion_filter=False,
                 filter_type="both",target_name="emotion"):
        self.target = None
        self.target_name = target_name
        self.questionnaire = None
        self.emotion_filter = emotion_filter
        self.emotion_filter_type = filter_type
        self.remove_questionnaire_label = config.remove_questionnaire_label

        # 前処理実行
        self.preprocessing_questionnaire(questionnaire_path)
        if self.questionnaire is not None:
            self.get_emotion_target()

    def preprocessing_questionnaire(self,question_path):
        self.questionnaire = pd.read_excel(question_path,sheet_name="questionnaire",usecols=range(13))

        # フラグ処理
        self.questionnaire = self.questionnaire.query("is_bad == 1")
        # Affect Gridの拡張
        self.extension_affectgrid()
        # 主観評価用フィルタ
        if self.emotion_filter:
            self.filter()
        self.questionnaire = self.questionnaire.drop(self.remove_questionnaire_label,axis=1)

    def extension_affectgrid(self):
        # AFFECT GRIDの感情の強さと，方向を決める
        _arousal = self.questionnaire["Arousal"].values
        _valence = self.questionnaire["Valence"].values
        self.questionnaire["strength"] = np.sqrt( np.square(_arousal) + np.square(_valence) )
        self.questionnaire["angle"]  = np.arctan(_arousal/ _valence)

    
    def filter(self):
        # 主観評価結果に基づいたデータの選定
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

        # 入力
        df_amusement = self.questionnaire.query("emotion == 'Amusement'")
        df_stress = self.questionnaire.query("emotion == 'Stress'")
        filtered_df_amusement = df_amusement
        filtered_df_stress = df_stress
        # Affect Gridによるフィルタ
        # Amusementは第一象限，Stressは第二象限
        if self.emotion_filter_type == "both" or self.emotion_filter_type == "affect_grid":
            filtered_df_amusement = filtered_df_amusement.query('Arousal >= 4 & Valence >= 4')
            filtered_df_stress = filtered_df_stress.query('Arousal >= 4 & Valence <= 4')
    
        # 感情ラベルによるフィルタ
        if self.emotion_filter_type == "both" or self.emotion_filter_type == "emotion_label":
            filtered_df_amusement = filtered_df_amusement.query('Emotion_1 in (1, 2, 3, 9, 10, 11)')
            filtered_df_stress = filtered_df_stress.query('Emotion_1 in (4, 5, 6, 7, 8, 12, 13, 14, 15)')
        
        self.questionnaire = pd.concat([filtered_df_amusement,filtered_df_stress],axis = 0,ignore_index=True)
        
        # 入力
        print("\n---------Selection of data by subjective evaluation---------")
        print("\nNumber of original data")
        print(" Stress data : {}\n Amusement data : {}\n Sum data : {}".format(df_stress.shape[0],
                                                                             df_amusement.shape[0],
                                                                             (df_stress.shape[0]+df_amusement.shape[0])))
        
        # 出力
        print("\nNumber of selected data")
        print("Filter Type: {}".format(self.emotion_filter_type))
        print(" Stress data : {}\n Amusement data : {}\n Sum data : {}".format(filtered_df_stress.shape[0],
                                                                               filtered_df_amusement.shape[0],
                                                                               (filtered_df_stress.shape[0]+filtered_df_amusement.shape[0])))
    
        print("\ndata filter result by subject")
        print("{}".format(self.questionnaire["user"].value_counts()))

    # 感情ラベルを作成
    def get_emotion_target(self):
        self.target = self.questionnaire[self.target_name]

        

# ExcelデータをNumpy形式に変換してデータセットを作成
def load_emotion_dataset():
    emotion_dataset = EmotionRecognition(config.features_path,
                                         config.questionnaire_path,
                                         config.normalization,
                                         config.emotion_filter,
                                         config.filter_type)
    print("dataset info :")
    print("target shape : {}".format(emotion_dataset.targets.shape))
    print("features shape : {}".format(emotion_dataset.features.shape))
    return emotion_dataset

def split_by_group(dataset):
    # 被験者ごとにグループ分け
    le = preprocessing.LabelEncoder()
    group = dataset.targets_user
    unique_subject = le.fit(np.unique(group))
    subjects = le.transform(group)
    gkf = list(GroupKFold(n_splits=len(np.unique(group))).split(dataset.features,
                                                                dataset.targets,
                                                                subjects))
    #　出力
    print(np.unique(group))
    print(le.transform(np.unique(group)))
    return gkf

# 特徴量選択
def boruta_feature_selection(dataset,show=False):
    # 特徴量選択用のモデル(RandamForest)の定義
    rf = RandomForestClassifier(n_jobs=-1,class_weight='balanced', max_depth=5)

    # BORUTAの特徴量選択
    feat_selector = BorutaPy(rf, n_estimators='auto',
                             verbose=2, two_step=False,
                             random_state=42,max_iter=70)

    # BORUTAを実行
    # 最低5個の特徴量が選ばれるそう
    feat_selector.fit(dataset.features, dataset.targets)

    # 出力
    column_list = dataset.features_label_list
    print('\n Initial features: {}'.format(column_list))

    # 選ばれた特徴量の数
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
    selected_label = dataset.features_label_list[feat_selector.support_]
    #selected = dataset.drop(drop_label, axis=1).columns[feat_selector.support_]
    selected_features = dataset.features[: ,feat_selector.support_]

    if show:
        ## 可視化
        mask = feat_selector.support_
        # マスクを可視化する．黒が選択された特徴量
        plt.matshow(mask.reshape(1, -1), cmap='gray_r')
        plt.tick_params(labelleft = 'off')
        plt.xlabel('Sample Index', fontsize=15)
        plt.show()

    return selected_label, selected_features




if __name__ =="__main__":
    load_emotion_dataset()
    test = Emotion_Label(config.questionnaire_path)
    print(test.target)
    print("success")
    pass