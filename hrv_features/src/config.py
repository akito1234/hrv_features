"""
プログラムの設定一覧
"""
# 主観評価データへのパス
questionnaire_path = r"C:\Users\akito\Desktop\stress\05.QuestionNaire\QuestionNaire_result.xlsx"

# 特徴量データへのパス
features_path = r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_Features\biosignal_datasets_1.xlsx"

# 重要 出力される感情ラベルの種類
emotion_state = ['Amusement','Stress']

emotion_baseline = "Neutral1"

# 特徴量データと主観評価を連結する用のキー
identical_parameter = ['id','emotion','user','date','path_name']

# 不要な特徴量
remove_features_label = ["nni_diff_min"]

# 主観評価データから取り除くカラム
remove_questionnaire_label = ["id","exp_id","trans_Emotion_1","trans_Emotion_2","Film","is_bad"]

# 目的変数
target_name = "emotion"
#target_name = "angle"
#target_name = "strength"
#target_name = "Valence"

# 注意　3_labelおよび4_labelはアンケートフィルタ必須
#target_name = "4label_emotion"

# 個人差補正 diff or ratio
#individual_parameter = "diff"
individual_parameter = "ratio"

# 個人差補正
normalization = True

# 主観評価によるフィルタ
emotion_filter = True

# フィルタの種類
filter_type="both"