import numpy as np
import pandas as pd

import os, glob

questionnaire_path = r"C:\Users\akito\Desktop\stress\05.QuestionNaire\QuestionNaire_result.xlsx"
features_path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets.xlsx"
identical_features = ['id','emotion','user','date','path_name']
remove_label = ["Neutral2"]
remove_features_label = ["nni_diff_min"]


#感情クラス
#前処理全般
class EmotionRecognition:
    def __init__(self, features_path,question_path):
        questionnaire = pd.read_excel(question_path,sheet_name="questionnaire",usecols=range(13))
        questionnaire_isbool = questionnaire.query("is_bad == 0")

        datasets = pd.read_excel(features_path)
        # これもどうするか考える必要あり
        datasets = datasets[~datasets["emotion"].isin(remove_label)]
        # 不要な特徴量を削除
        datasets =  datasets.drop(remove_features_label,axis=1)
        
        datasets[(datasets["date"] == questionnaire_isbool["date"].values) 
                 & (datasets["user"] == questionnaire_isbool["user"].values)]

        ## データセットからis_bad=0 となるデータを取り除く
        #dataset.update(df2.pivot('col1', 'col3', 'col2'))
        datasets.to_excel(r"C:\Users\akito\Desktop\output.xlsx")
        self.targets = datasets["emotion"].values
        self.target_user = datasets["user"].values
        self.features = datasets.drop(identical_features,axis=1).values
        self.feature_names = datasets.columns
        self.identical_features = identical_features

    #,remove_label,normalization
    # targetを定義する
    def get_target(self):
        
        pass

    #キー(インスタンス変数)を取得するメソッド
    def keys(self):
        print("[targets, target_names, features]")
    def get_target_label(self):
        self.targets = ""
        pass
    def emotion_filter(self):
        # 主観評価値から使うデータラベルを選択
        pass
    # アンケート結果によるフィルタ
    def extension_affectgrid(questionnaire):
        # AFFECT GRIDの感情の強さと，方向を決める
        _arousal = questionnaire["Arousal"].values
        _valence = questionnaire["Valence"].values
        questionnaire["strength"] = np.sqrt( np.square(_arousal) + np.square(_valence) )
        questionnaire["angle"]  = np.arctan(_arousal/ _valence)
        return questionnaire

    # 個人差の補正 (感情ごとの特徴量からベースラインを引く)
    def normalization(self,df,emotion_state=['Stress','Ammusement','Neutral2'],
                          baseline='Neutral1',
                          identical_parameter = ['id','emotion','user','date','path_name']):
        df_summary = None
        for i in range(df['id'].max() + 1):
            # 各実験データを取り出す
            df_item = df[ df['id']  == i]

            # ベースラインを取得
            baseline_df = df_item[ df_item['emotion']  == baseline]
        
            # 初期化 　->　空のDataframeを作成
            if df_summary is None:
                df_summary = pd.DataFrame([], columns=baseline_df.columns)

            for state in emotion_state:
                # 各感情状態の特徴を取り出す
                emotion_df = df_item[ df_item['emotion']  == state]
                if emotion_df.empty:
                    continue;

                # 各感情状態からベースラインを引く 
                correction_emotion_df = correction_neutral_before(baseline_df,emotion_df,identical_parameter)

                # ファイルを結合
                df_summary = df_summary.append(correction_emotion_df,ignore_index=True,sort=False)

        return df_summary
    # ニュートラル状態を差分する
    def correction_neutral_before(self,df_neutral,df_emotion,identical_parameter):
        identical_df = df_emotion[identical_parameter]

        #不要なパラメータのを除き，Neuralで補正
        df_neutral_features = df_neutral.drop(identical_parameter, axis=1)
        df_emotion_features = df_emotion.drop(identical_parameter, axis=1)
        features_df = (df_emotion_features/df_neutral_features.values)

        result = pd.concat([identical_df,features_df], axis=1,sort=False)
        return result


# ExcelデータをNumpy形式に変換してデータセットを作成
def load_emotion_dataset():
    emotion_dataset = EmotionRecognition(features_path,questionnaire_path)
    # 個人差の補正処理
    #emoton_dataset.normalization()

    return emotion_dataset
    pass
load_emotion_dataset()