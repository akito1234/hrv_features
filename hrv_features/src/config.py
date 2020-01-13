"""
プログラムの設定一覧
"""

# 主観評価データへのパス
questionnaire_path = r"Z:\theme\mental_arithmetic\06.QuestionNaire\QuestionNaire_result.xlsx"

# 特徴量データへのパス
features_path = r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_1.xlsx"

# 特徴量データと主観評価を連結する用のキー
identical_parameter = ['id','emotion','user','date','path_name']

# 取り除く感情名
remove_label = ["Neutral2"]

# 不要な特徴量
remove_features_label = ["nni_diff_min"]

# 主観評価データから取り除くカラム
remove_questionnaire_label = ["id","exp_id","trans_Emotion_1","trans_Emotion_2","Film","is_bad"]
