"""模型脚本。

文件：catboost_intercept_model.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import catboost_intercept_tuning as base

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 图片中的结论参数
FIXED_CATBOOST_PARAMS = {
    "iterations": 700,
    "depth": 3,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
}

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "catboost_intercept_reproduce_best_results.xlsx")


def mean_absolute_percentage_error(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def run_fixed_config(X_dynamic, X_static, y, subjects):
    """按照固定参数复现结果。"""
    scaler_dyn = MinMaxScaler()
    X_dynamic_scaled = scaler_dyn.fit_transform(X_dynamic)

    unique_subjects = np.unique(subjects)
    kf = KFold(n_splits=base.N_SPLITS, shuffle=True, random_state=base.KF_RANDOM_STATE_CROSS)
    fold_records = []

    print("=" * 80)
    print("CatBoost Intercept Prediction - Fixed Best Parameters")
    print("=" * 80)
    print("Fixed parameters:")
    for key, value in FIXED_CATBOOST_PARAMS.items():
        print(f"  {key}: {value}")
    print(f"  random_state: 42")
    print(f"Lambda tune: {base.LAMBDA_TUNE_METHOD} | candidates: {base.LAMBDA_CANDIDATES}")
    print("-" * 80)

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
        sub_static_dict = base.get_subject_static_features(X_static_scaled_full, subjects)

        X_d_train = X_dynamic_scaled[train_mask]
        X_d_test = X_dynamic_scaled[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        subs_train_fold = subjects[train_mask]
        subs_test_fold = subjects[test_mask]

        base_model = base.RandomForestRegressor(**base.RF_CONFIG_CROSS)
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

        intercept_model = base.train_intercept_model(
            train_sub_static_feats,
            train_sub_alphas,
            **FIXED_CATBOOST_PARAMS,
        )

        alpha_pred_loo = {}
        for j, s in enumerate(train_subs):
            feats_loo = np.array([train_sub_static_feats[i] for i in range(len(train_subs)) if i != j])
            alphas_loo = np.array([train_sub_alphas[i] for i in range(len(train_subs)) if i != j])
            if len(alphas_loo) < 2:
                alpha_pred_loo[s] = 0.0
                continue
            g_loo = base.train_intercept_model(
                feats_loo,
                alphas_loo,
                **FIXED_CATBOOST_PARAMS,
            )
            alpha_pred_loo[s] = g_loo.predict(train_sub_static_feats[j].reshape(1, -1))[0]
        alpha_pred_train = np.array([alpha_pred_loo[s] for s in subs_train_fold], dtype=float)

        alpha_pred_per_sample = np.zeros(len(y_test), dtype=float)
        for i, s in enumerate(subs_test_fold):
            alpha_pred_per_sample[i] = intercept_model.predict(sub_static_dict[s].reshape(1, -1))[0]

        best_lam = base.LAMBDA_SHRINK
        best_mse = np.inf
        for lam in base.LAMBDA_CANDIDATES:
            y_fit = y_pred_train + lam * alpha_pred_train
            mse = mean_squared_error(y_train, y_fit)
            if mse < best_mse:
                best_mse = mse
                best_lam = lam

        y_pred_final = y_pred_base_test + best_lam * alpha_pred_per_sample

        r2_base = r2_score(y_test, y_pred_base_test)
        r2_final = r2_score(y_test, y_pred_final)
        mae_base = mean_absolute_error(y_test, y_pred_base_test)
        mae_final = mean_absolute_error(y_test, y_pred_final)
        mape_base = mean_absolute_percentage_error(y_test, y_pred_base_test)
        mape_final = mean_absolute_percentage_error(y_test, y_pred_final)

        print(
            f"Fold {fold_idx}: Base R2={r2_base:.4f} -> Final R2={r2_final:.4f} "
            f"(Diff: {r2_final - r2_base:+.4f}) lambda={best_lam} | "
            f"MAE: {mae_base:.4f} -> {mae_final:.4f} | "
            f"MAPE: {mape_base:.4f} -> {mape_final:.4f}"
        )

        fold_records.append(
            {
                "fold": fold_idx,
                "r2_base": r2_base,
                "r2_final": r2_final,
                "mae_base": mae_base,
                "mae_final": mae_final,
                "mape_base": mape_base,
                "mape_final": mape_final,
                "lambda_used": best_lam,
            }
        )

    df_folds = pd.DataFrame(fold_records)
    summary = {
        "mean_r2": float(df_folds["r2_final"].mean()),
        "std_r2": float(df_folds["r2_final"].std()),
        "mean_r2_base": float(df_folds["r2_base"].mean()),
        "std_r2_base": float(df_folds["r2_base"].std()),
        "mean_mae": float(df_folds["mae_final"].mean()),
        "mean_mae_base": float(df_folds["mae_base"].mean()),
        "mean_mape": float(df_folds["mape_final"].mean()),
        "mean_mape_base": float(df_folds["mape_base"].mean()),
        "mean_gain": float(df_folds["r2_final"].mean() - df_folds["r2_base"].mean()),
        "lambda_used": str(df_folds["lambda_used"].tolist()),
        **FIXED_CATBOOST_PARAMS,
    }
    return summary, df_folds


def main():
    X_dynamic, X_static, y, subjects = base.load_data()

    summary, df_folds = run_fixed_config(X_dynamic, X_static, y, subjects)
    df_summary = pd.DataFrame([summary])

    print("-" * 80)
    print("Reproduced Summary")
    print("-" * 80)
    print(df_summary[["mean_r2", "std_r2", "iterations", "depth", "learning_rate", "l2_leaf_reg", "mean_gain"]].to_string(index=False))

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_folds.to_excel(writer, sheet_name="folds", index=False)

    print()
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
