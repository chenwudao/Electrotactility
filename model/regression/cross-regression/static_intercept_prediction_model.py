"""模型脚本。

文件：static_intercept_prediction_model.py
统一使用项目内的相对路径。
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import catboost as cb

# === 配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_FILE = os.path.join(
    BASE_DIR,
    r"feature-enginnering/linear-features/experiment1.0-50features/X_结合相关互相关_spe_0.9_rfr_50.xlsx",
)
LABEL_FILE = os.path.join(BASE_DIR, r"feature-enginnering/linear-features/experiment1.0-50features/Y.xlsx")
SUBJECT_COL = "受试者编号"

# 基础模型配置（与当前最佳跨被试RF一致）
RF_CONFIG_CROSS = {
    "criterion": "friedman_mse",
    "max_depth": 5,
    "max_features": 3,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 7,
    "random_state": 1435,
}

# 截距模型配置（用静态特征预测 Alpha）
INTERCEPT_MODEL_TYPE = "catboost"  # 'rf', 'ridge', 'catboost'；ridge 更稳、易收缩
INTERCEPT_RIDGE_ALPHA = 200.0   # 越大截距预测越往 0 收缩；
# 预测截距的收缩系数：y_final = y_base + lambda * alpha_pred
LAMBDA_SHRINK = 0.4
# lambda 选择方式：'loo' 表示留一被试选 lambda（更稳），'in_sample' 表示在训练集按 MSE 选择（更易过拟合）
LAMBDA_TUNE_METHOD = "loo"
LAMBDA_CANDIDATES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]  

# 动态特征（来自当前最佳的20维特征集）
ORIGINAL_12_FEATURES = [
    "v1-v2", "LOG_R2", "指围-指长", "45. FFM% of Right Arm", 
    "199. Measured Circumference of Neck", "206. Measured Circumference of Left Thigh",
    "GLCM_t_0_d_2_SVD15_465", "STHOS_r_2_a_45_LRHGLE_22",
    "LBP_r3_pt_17_342", "LBP_r3_pt_18_343", "ZER_r3_pt_4_80", "ZER_r3_pt_19_95",
]
INITIAL_LOG_FEATURES = ["45. FFM% of Right Arm", "199. Measured Circumference of Neck"]
TOP3_ADDITIONAL = ["ZER_r3_pt_15_91", "GLCM_t_0_d_4_SVD14_524", "STHOS_r_2_a_135_LRLGLE_43"]
NEW_LOG_FEATURES = ["206. Measured Circumference of Left Thigh", "STHOS_r_2_a_45_LRHGLE_22", "LOG_R2"]

# 用于截距预测的静态特征
# ----------------------------------------
# 选取理由：
# 1) 必须是“被试内基本不变”的变量（或按被试取均值后有意义），才能对新被试预测其截距。
# 2) 来自 label_noise_report：与阈值相关最强的几个是 颈围(r≈0.45)、右臂FFM%(0.43)、大腿围(0.23)、
#    LOG_R2(-0.13)。前三个是体测/形态学，同一被试多次测量可视为常数，故选为静态特征。
# 3) 指围-指长：同属形态学，且在 ORIGINAL_12 里，跨被试常用，故加入。
# 未选：LOG_R2、v1-v2 等若为试次级或定义上随试次变，则不宜作为“纯静态”；若你确认某列为被试级，
#       可加入下方列表。也可从 50 维特征表中选其他体测/人口学列名加入。
STATIC_FEATURE_NAMES = [
    "199. Measured Circumference of Neck",
    "206. Measured Circumference of Left Thigh",
    "45. FFM% of Right Arm",
    "指围-指长",
]
# 可选扩展（若 df_feat 中有且确认为被试级）：例如 "v1-v2", "LOG_R2"（需按被试取均值）
# STATIC_FEATURE_NAMES_EXTRA = ["v1-v2", "LOG_R2"]

KF_RANDOM_STATE_CROSS = 427
N_SPLITS = 5

def mean_absolute_percentage_error(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def safe_log(values):
    v = np.asarray(values, dtype=float)
    m = np.min(v)
    if m <= 0: return np.log(v - m + 1.0)
    return np.log(v)

def load_data():
    df_feat = pd.read_excel(FEAT_FILE)
    tf = pd.read_excel(LABEL_FILE, index_col=0)
    
    if "阈值" in tf.columns:
        y = tf["阈值"].to_numpy(dtype=float)
    else:
        y = tf.iloc[:, 3].to_numpy(dtype=float)
    
    subjects = tf[SUBJECT_COL].to_numpy()
    
    # 构建20维动态特征
    X_12 = df_feat[ORIGINAL_12_FEATURES].to_numpy()
    X_init = np.column_stack([safe_log(df_feat[f].values) for f in INITIAL_LOG_FEATURES])
    X_top3 = df_feat[TOP3_ADDITIONAL].to_numpy()
    X_new = np.column_stack([safe_log(df_feat[f].values) for f in NEW_LOG_FEATURES])
    X_dynamic = np.hstack([X_12, X_init, X_top3, X_new])
    
    # 构建静态特征矩阵（形状：N_samples x N_static）
    # 使用 df_feat 中 STATIC_FEATURE_NAMES 对应的原始值
    X_static = df_feat[STATIC_FEATURE_NAMES].to_numpy()
    
    return X_dynamic, X_static, y, subjects

def get_subject_static_features(X_static, subjects):
    """
    Returns a dictionary: {subject_id: static_features_vector}
    Assumes static features are constant per subject. If not, takes the mean.
    """
    sub_static = {}
    unique_subs = np.unique(subjects)
    for s in unique_subs:
        mask = subjects == s
        # 为抑制轻微波动或噪声，对同一受试者取均值
        mean_static = np.mean(X_static[mask], axis=0)
        sub_static[s] = mean_static
    return sub_static

def train_intercept_model(X_train_static_agg, alpha_train, model_type="ridge", ridge_alpha=10.0):
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)  # 浅一点防过拟合
    elif model_type == "ridge":
        model = Ridge(alpha=ridge_alpha, random_state=42)
    elif model_type == "catboost":
        model = cb.CatBoostRegressor(
            iterations=500, depth=3, learning_rate=0.05,
            l2_leaf_reg=1.0, verbose=False, allow_writing_files=False,
            random_state=42,
        )
    else:
        raise ValueError("Unknown Intercept Model Type")
    model.fit(X_train_static_agg, alpha_train)
    return model

def main():
    X_dynamic, X_static, y, subjects = load_data()

    # 动态特征：全局 MinMax（与主回归脚本一致）
    scaler_dyn = MinMaxScaler()
    X_dynamic_scaled = scaler_dyn.fit_transform(X_dynamic)

    unique_subjects = np.unique(subjects)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=KF_RANDOM_STATE_CROSS)
    results = []

    print("Starting Cross-Subject Intercept Prediction Strategy (improved)...")
    print(f"Dynamic Model: RF | Intercept Model: {INTERCEPT_MODEL_TYPE} (ridge_alpha={INTERCEPT_RIDGE_ALPHA})")
    print(f"Lambda tune: {LAMBDA_TUNE_METHOD} | candidates: {LAMBDA_CANDIDATES}")
    print("-" * 60)

    for fold_idx, (train_sub_idx, test_sub_idx) in enumerate(kf.split(unique_subjects)):
        train_subs = unique_subjects[train_sub_idx]
        test_subs = unique_subjects[test_sub_idx]
        train_mask = np.isin(subjects, train_subs)
        test_mask = np.isin(subjects, test_subs)

        # 本 fold 内：静态特征仅在训练被试上 fit scaler，再 transform 全体（避免泄露）
        scaler_stat = StandardScaler()
        X_static_train = X_static[train_mask]
        scaler_stat.fit(X_static_train)
        X_static_scaled_full = np.zeros_like(X_static, dtype=float)
        X_static_scaled_full[train_mask] = scaler_stat.transform(X_static_train)
        X_static_scaled_full[test_mask] = scaler_stat.transform(X_static[test_mask])
        sub_static_dict = get_subject_static_features(X_static_scaled_full, subjects)

        X_d_train = X_dynamic_scaled[train_mask]
        X_d_test = X_dynamic_scaled[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        subs_train_fold = subjects[train_mask]
        subs_test_fold = subjects[test_mask]

        # 1. 训练全局模型
        base_model = RandomForestRegressor(**RF_CONFIG_CROSS)
        base_model.fit(X_d_train, y_train)
        y_pred_train = base_model.predict(X_d_train)
        y_pred_base_test = base_model.predict(X_d_test)
        residuals_train = y_train - y_pred_train

        train_sub_alphas = []
        train_sub_static_feats = []
        for s in train_subs:
            s_mask = subs_train_fold == s
            if np.sum(s_mask) > 0:
                true_alpha = np.mean(residuals_train[s_mask])
                train_sub_alphas.append(true_alpha)
                train_sub_static_feats.append(sub_static_dict[s])
        train_sub_alphas = np.array(train_sub_alphas)
        train_sub_static_feats = np.array(train_sub_static_feats)

        # 2. 训练截距模型（全训练被试，用于测试集预测）
        intercept_model = train_intercept_model(
            train_sub_static_feats, train_sub_alphas,
            INTERCEPT_MODEL_TYPE, ridge_alpha=INTERCEPT_RIDGE_ALPHA,
        )

        # 3. 在训练集上得到每个样本的 alpha_pred，用于选择 lambda
        # LOO：留一被试预测 alpha，避免“见过该被试”的过拟合，选出的 lambda 更稳
        if LAMBDA_TUNE_METHOD == "loo":
            alpha_pred_LOO = {}
            for j, s in enumerate(train_subs):
                feats_loo = np.array([train_sub_static_feats[i] for i in range(len(train_subs)) if i != j])
                alphas_loo = np.array([train_sub_alphas[i] for i in range(len(train_subs)) if i != j])
                if len(alphas_loo) < 2:
                    alpha_pred_LOO[s] = 0.0
                    continue
                g_loo = train_intercept_model(feats_loo, alphas_loo, INTERCEPT_MODEL_TYPE, ridge_alpha=INTERCEPT_RIDGE_ALPHA)
                alpha_pred_LOO[s] = g_loo.predict(train_sub_static_feats[j].reshape(1, -1))[0]
            alpha_pred_train = np.array([alpha_pred_LOO[s] for s in subs_train_fold], dtype=float)
        else:
            alpha_pred_train = np.zeros(len(y_train), dtype=float)
            for i, s in enumerate(subs_train_fold):
                alpha_pred_train[i] = intercept_model.predict(sub_static_dict[s].reshape(1, -1))[0]

        # 4. 在测试集上预测 alpha（使用全训练被试训练得到的 intercept_model）
        alpha_pred_per_sample = np.zeros(len(y_test), dtype=float)
        for i, s in enumerate(subs_test_fold):
            alpha_pred_per_sample[i] = intercept_model.predict(sub_static_dict[s].reshape(1, -1))[0]

        # 5. 选择 lambda：用 alpha_pred_train 在训练集上遍历 LAMBDA_CANDIDATES，取 MSE 最小值
        if len(LAMBDA_CANDIDATES) > 0:
            best_lam = LAMBDA_SHRINK
            best_mse = np.inf
            for lam in LAMBDA_CANDIDATES:
                y_fit = y_pred_train + lam * alpha_pred_train
                mse = mean_squared_error(y_train, y_fit)
                if mse < best_mse:
                    best_mse = mse
                    best_lam = lam
            lambda_used = best_lam
        else:
            lambda_used = LAMBDA_SHRINK

        y_pred_final = y_pred_base_test + lambda_used * alpha_pred_per_sample

        r2_base = r2_score(y_test, y_pred_base_test)
        r2_final = r2_score(y_test, y_pred_final)
        mae_base = mean_absolute_error(y_test, y_pred_base_test)
        mae_final = mean_absolute_error(y_test, y_pred_final)
        mape_base = mean_absolute_percentage_error(y_test, y_pred_base_test)
        mape_final = mean_absolute_percentage_error(y_test, y_pred_final)

        print(
            f"Fold {fold_idx}: Base R2={r2_base:.4f} -> Final R2={r2_final:.4f} "
            f"(Diff: {r2_final-r2_base:+.4f}) lambda={lambda_used} | "
            f"MAE: {mae_base:.4f} -> {mae_final:.4f} | "
            f"MAPE: {mape_base:.4f} -> {mape_final:.4f}"
        )

        results.append({
            "fold": fold_idx,
            "r2_base": r2_base,
            "r2_final": r2_final,
            "mae_base": mae_base,
            "mae_final": mae_final,
            "mape_base": mape_base,
            "mape_final": mape_final,
            "lambda_used": lambda_used,
        })

    df_res = pd.DataFrame(results)
    print("-" * 60)
    print("Summary:")
    print(f"Mean Base R2 : {df_res['r2_base'].mean():.4f} +/- {df_res['r2_base'].std():.4f}")
    print(f"Mean Final R2: {df_res['r2_final'].mean():.4f} +/- {df_res['r2_final'].std():.4f}")
    print(f"Mean Base MAE : {df_res['mae_base'].mean():.4f} +/- {df_res['mae_base'].std():.4f}")
    print(f"Mean Final MAE: {df_res['mae_final'].mean():.4f} +/- {df_res['mae_final'].std():.4f}")
    print(f"Mean Base MAPE : {df_res['mape_base'].mean():.4f} +/- {df_res['mape_base'].std():.4f}")
    print(f"Mean Final MAPE: {df_res['mape_final'].mean():.4f} +/- {df_res['mape_final'].std():.4f}")
    gain = df_res['r2_final'].mean() - df_res['r2_base'].mean()
    print(f"Average Gain from Intercept Prediction: {gain:+.4f}")
    print(f"Lambda used per fold: {df_res['lambda_used'].tolist()}")
    return df_res

if __name__ == "__main__":
    main()
