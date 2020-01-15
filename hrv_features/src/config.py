"""
プログラムの設定一覧
"""

#--------------------------------
# プログラム設定
#--------------------------------
# 個人差補正 diff or ratio
#individual_parameter = "diff"
individual_parameter = "ratio"

# 個人差補正
normalization=True

# 主観評価によるフィルタ
emotion_filter=True

# フィルタの種類
filter_type="both"

# Borutaにより選ばれた特徴量
selected_label = ["rmssd","sdnn","lomb_abs_lf","hr_min","hr_mean","nni_max",
                  "nni_diff_mean","tinn_m","tinn","nni_mean","sd1","sd2",
                  "nni_counter","ellipse_area","lomb_total","tri_index","ar_rel_vlf"]

#--------------------------------
# 諸設定
#--------------------------------
# 主観評価データへのパス
questionnaire_path = r"Z:\theme\mental_arithmetic\06.QuestionNaire\QuestionNaire_result.xlsx"


# 特徴量データへのパス
#features_path = r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_120s.xlsx"
features_path = r"Z:\theme\mental_arithmetic\04.Analysis\Analysis_Features\biosignal_datasets_300s.xlsx"

# 特徴量データと主観評価を連結する用のキー
identical_parameter = ['id','emotion','user','date','path_name']

# 取り除く感情名
remove_label = ["Neutral2"]

# 不要な特徴量
remove_features_label = ["nni_diff_min"]

# 主観評価データから取り除くカラム
remove_questionnaire_label = ["id","exp_id","trans_Emotion_1","trans_Emotion_2","Film","is_bad"]
