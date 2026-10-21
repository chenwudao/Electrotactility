"""模型脚本。

文件：catboost_intercept_tuning.py
统一使用项目内的相对路径。
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import catboost as cb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_FILE = os.path.join(
    BASE_DIR,
    r"feature-enginnering/linear-features/experiment1.0-50features/X_结合相关互相关_spe_0.9_rfr_50.xlsx",
)
LABEL_FILE = os.path.join(BASE_DIR, r"feature-enginnering/linear-features/experiment1.0-50features/Y.xlsx")
SUBJECT_COL = "受试者编号"

RF_CONFIG_CROSS = {
    "criterion": "friedman_mse",
    "max_depth": 5,
    "max_features": 3,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 7,
    "random_state": 1435,
}

LAMBDA_SHRINK = 0.4
LAMBDA_TUNE_METHOD = "loo"
LAMBDA_CANDIDATES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

ORIGINAL_12_FEATURES = [
    "v1-v2", "LOG_R2", "指围-指长", "45. FFM% of Right Arm", 
    "199. Measured Circumference of Neck", "206. Measured Circumference of Left Thigh",
    "GLCM_t_0_d_2_SVD15_465", "STHOS_r_2_a_45_LRHGLE_22",
    "LBP_r3_pt_17_342", "LBP_r3_pt_18_343", "ZER_r3_pt_4_80", "ZER_r3_pt_19_95",
]
INITIAL_LOG_FEATURES = ["45. FFM% of Right Arm", "199. Measured Circumference of Neck"]
TOP3_ADDITIONAL = ["ZER_r3_pt_15_91", "GLCM_t_0_d_4_SVD14_524", "STHOS_r_2_a_135_LRLGLE_43"]
NEW_LOG_FEATURES = ["206. Measured Circumference of Left Thigh", "STHOS_r_2_a_45_LRHGLE_22", "LOG_R2"]

STATIC_FEATURE_NAMES = [
    "199. Measured Circumference of Neck",
    "206. Measured Circumference of Left Thigh",
    "45. FFM% of Right Arm",
    "指围-指长",
]

KF_RANDOM_STATE_CROSS = 427
N_SPLITS = 5

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
    
    X_12 = df_feat[ORIGINAL_12_FEATURES].to_numpy()
    X_init = np.column_stack([safe_log(df_feat[f].values) for f in INITIAL_LOG_FEATURES])
    X_top3 = df_feat[TOP3_ADDITIONAL].to_numpy()
    X_new = np.column_stack([safe_log(df_feat[f].values) for f in NEW_LOG_FEATURES])
    X_dynamic = np.hstack([X_12, X_init, X_top3, X_new])
    
    X_static = df_feat[STATIC_FEATURE_NAMES].to_numpy()
    
    return X_dynamic, X_static, y, subjects

def get_subject_static_features(X_static, subjects):
    sub_static = {}
    unique_subs = np.unique(subjects)
    for s in unique_subs:
        mask = subjects == s
        mean_static = np.mean(X_static[mask], axis=0)
        sub_static[s] = mean_static
    return sub_static

def train_intercept_model(X_train_static_agg, alpha_train, **catboost_params):
    model = cb.CatBoostRegressor(
        verbose=False,
        allow_writing_files=False,
        random_state=42,
        **catboost_params
    )
    model.fit(X_train_static_agg, alpha_train)
    return model

def main():
    X_dynamic, X_static, y, subjects = load_data()

    scaler_dyn = MinMaxScaler()
    X_dynamic_scaled = scaler_dyn.fit_transform(X_dynamic)

    unique_subjects = np.unique(subjects)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=KF_RANDOM_STATE_CROSS)

    catboost_param_grid = {
        'iterations': [300, 500, 700],
        'depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1.0, 3.0, 5.0, 10.0],
    }

    param_combinations = list(ParameterGrid(catboost_param_grid))
    print(f"CatBoost 参数组合数: {len(param_combinations)}")

    best_params = None
    best_mean_r2 = -np.inf
    results_summary = []

    print("\n" + "=" * 60)
    print("CatBoost 超参数调优")
    print("=" * 60)

    for param_idx, params in enumerate(param_combinations):
        fold_results = []
        
        for fold_idx, (train_sub_idx, test_sub_idx) in enumerate(kf.split(unique_subjects)):
            train_subs = unique_subjects[train_sub_idx]
            test_subs = unique_subjects[test_sub_idx]
            train_mask = np.isin(subjects, train_subs)
            test_mask = np.isin(subjects, test_subs)

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

            intercept_model = train_intercept_model(
                train_sub_static_feats, train_sub_alphas, **params
            )

            alpha_pred_LOO = {}
            for j, s in enumerate(train_subs):
                feats_loo = np.array([train_sub_static_feats[i] for i in range(len(train_subs)) if i != j])
                alphas_loo = np.array([train_sub_alphas[i] for i in range(len(train_subs)) if i != j])
                if len(alphas_loo) < 2:
                    alpha_pred_LOO[s] = 0.0
                    continue
                g_loo = train_intercept_model(feats_loo, alphas_loo, **params)
                alpha_pred_LOO[s] = g_loo.predict(train_sub_static_feats[j].reshape(1, -1))[0]
            alpha_pred_train = np.array([alpha_pred_LOO[s] for s in subs_train_fold], dtype=float)

            alpha_pred_per_sample = np.zeros(len(y_test), dtype=float)
            for i, s in enumerate(subs_test_fold):
                alpha_pred_per_sample[i] = intercept_model.predict(sub_static_dict[s].reshape(1, -1))[0]

            best_lam = LAMBDA_SHRINK
            best_mse = np.inf
            for lam in LAMBDA_CANDIDATES:
                y_fit = y_pred_train + lam * alpha_pred_train
                mse = mean_squared_error(y_train, y_fit)
                if mse < best_mse:
                    best_mse = mse
                    best_lam = lam

            y_pred_final = y_pred_base_test + best_lam * alpha_pred_per_sample

            r2_base = r2_score(y_test, y_pred_base_test)
            r2_final = r2_score(y_test, y_pred_final)
            
            fold_results.append(r2_final)

        mean_r2 = np.mean(fold_results)
        std_r2 = np.std(fold_results)
        
        results_summary.append({
            'param_idx': param_idx,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            **params
        })
        
        if mean_r2 > best_mean_r2:
            best_mean_r2 = mean_r2
            best_params = params
        
        if (param_idx + 1) % 10 == 0:
            print(f"进度: {param_idx + 1}/{len(param_combinations)} | 当前最佳 R2: {best_mean_r2:.4f}")

    print("\n" + "=" * 60)
    print("调优结果")
    print("=" * 60)
    print(f"\n最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\n最佳 R2: {best_mean_r2:.4f}")

    df_results = pd.DataFrame(results_summary)
    df_results = df_results.sort_values('mean_r2', ascending=False)
    
    print("\nTop 10 参数组合:")
    print(df_results.head(10)[['mean_r2', 'std_r2', 'iterations', 'depth', 'learning_rate', 'l2_leaf_reg']].to_string(index=False))

    output_file = os.path.join(os.path.dirname(__file__), "catboost_tuning_results.xlsx")
    df_results.to_excel(output_file, index=False)
    print(f"\n结果已保存: {output_file}")

if __name__ == "__main__":
    main()
