"""模型脚本。

文件：intercept_model_comparison.py
统一使用项目内的相对路径。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge, BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import catboost as cb
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_FILE = os.path.join(
    BASE_DIR,
    r"feature-enginnering\linear-features\experiment1.0-50features\X_结合相关互相关_spe_0.9_rfr_50.xlsx",
)
LABEL_FILE = os.path.join(BASE_DIR, r"feature-enginnering\linear-features\experiment1.0-50features\Y.xlsx")
SUBJECT_COL = "受试者编号"

N_SPLITS = 5
SHUFFLE = True
KF_RANDOM_STATE_WITHIN = 53

RF_CONFIG_WITHIN = {
    "criterion": "poisson",
    "max_depth": 7,
    "max_features": 2,
    "min_samples_leaf": 1,
    "min_samples_split": 3,
    "n_estimators": 22,
    "random_state": 5532311,
}

INTERCEPT_MODEL_TYPES = ["ridge", "huber", "catboost", "xgboost", "bayesian"]

RIDGE_ALPHA_SUBJECT = 1.0
RIDGE_ALPHA_CANDIDATES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
HUBER_EPSILON = 1.35
CATBOOST_ITERATIONS = 500
CATBOOST_DEPTH = 3
CATBOOST_LR = 0.05
CATBOOST_L2 = 1.0
XGB_ITERATIONS = 500
XGB_DEPTH = 3
XGB_LR = 0.05
XGB_LAMBDA = 1.0

TARGET_MODE_WITHIN = "winsor"
WINSOR_PCT = (0.02, 0.98)

LAMBDA_SHRINK = 1.0
LAMBDA_TUNE_METHOD = "fixed"
LAMBDA_CANDIDATES = [1.0]

SCALER_MODE = "auto"

OUTPUT_DIR = SCRIPT_DIR
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "intercept_model_comparison_within.xlsx")

ORIGINAL_12_FEATURES = [
    "v1-v2",
    "LOG_R2",
    "指围-指长",
    "45. FFM% of Right Arm",
    "199. Measured Circumference of Neck",
    "206. Measured Circumference of Left Thigh",
    "GLCM_t_0_d_2_SVD15_465",
    "STHOS_r_2_a_45_LRHGLE_22",
    "LBP_r3_pt_17_342",
    "LBP_r3_pt_18_343",
    "ZER_r3_pt_4_80",
    "ZER_r3_pt_19_95",
]
INITIAL_LOG_FEATURES = [
    "45. FFM% of Right Arm",
    "199. Measured Circumference of Neck",
]
TOP3_ADDITIONAL = [
    "ZER_r3_pt_15_91",
    "GLCM_t_0_d_4_SVD14_524",
    "STHOS_r_2_a_135_LRLGLE_43",
]
NEW_LOG_FEATURES = [
    "206. Measured Circumference of Left Thigh",
    "STHOS_r_2_a_45_LRHGLE_22",
    "LOG_R2",
]

def safe_log(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    m = np.min(v)
    if m <= 0:
        return np.log(v - m + 1.0)
    return np.log(v)

def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8
) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def winsorize_target(y: np.ndarray, pct_low: float, pct_high: float) -> np.ndarray:
    low, high = np.quantile(y, [pct_low, pct_high])
    return np.clip(y, low, high)

def make_base_model(rf_config: Dict[str, Any]) -> Any:
    return RandomForestRegressor(**rf_config)

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    df_feat = pd.read_excel(FEAT_FILE)
    tf = pd.read_excel(LABEL_FILE, index_col=0)
    if "阈值" in tf.columns:
        y = tf["阈值"].to_numpy(dtype=float)
    else:
        y = tf.iloc[:, 3].to_numpy(dtype=float)
    subjects = tf[SUBJECT_COL].to_numpy()

    X_12 = df_feat[ORIGINAL_12_FEATURES].to_numpy()
    X_init = np.column_stack([safe_log(np.asarray(df_feat[f].values, dtype=float)) for f in INITIAL_LOG_FEATURES])
    X_top3 = df_feat[TOP3_ADDITIONAL].to_numpy()
    X_new = np.column_stack([safe_log(np.asarray(df_feat[f].values, dtype=float)) for f in NEW_LOG_FEATURES])
    X_20 = np.hstack([X_12, X_init, X_top3, X_new])
    return X_20, y, subjects, tf.index

def build_within_splits(n: int, n_splits: int, shuffle: bool, random_state: int):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(kf.split(np.arange(n)))

def fit_subject_offsets(
    train_subject_ids: np.ndarray,
    residuals: np.ndarray,
    model_type: str = "ridge",
    ridge_alpha: float = RIDGE_ALPHA_SUBJECT,
) -> Tuple[Any, Dict[Any, float]]:
    uniq = np.unique(train_subject_ids)
    n_train = len(train_subject_ids)
    Z = np.zeros((n_train, len(uniq)), dtype=float)
    sub_to_col = {s: i for i, s in enumerate(uniq)}
    for i, s in enumerate(train_subject_ids):
        Z[i, sub_to_col[s]] = 1.0

    if model_type == "ridge":
        model = Ridge(alpha=ridge_alpha, fit_intercept=True, random_state=42)
    elif model_type == "huber":
        model = HuberRegressor(epsilon=HUBER_EPSILON, alpha=0.0001, max_iter=1000, tol=1e-4)
    elif model_type == "catboost":
        model = cb.CatBoostRegressor(
            iterations=CATBOOST_ITERATIONS,
            depth=CATBOOST_DEPTH,
            learning_rate=CATBOOST_LR,
            l2_leaf_reg=CATBOOST_L2,
            verbose=False,
            allow_writing_files=False,
            random_state=42,
        )
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=XGB_ITERATIONS,
            max_depth=XGB_DEPTH,
            learning_rate=XGB_LR,
            reg_lambda=XGB_LAMBDA,
            random_state=42,
            verbosity=0,
        )
    elif model_type == "bayesian":
        model = BayesianRidge(
            max_iter=1000,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            compute_score=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(Z, residuals)

    alpha_by_subject = {}
    if model_type in ["ridge", "huber", "bayesian"]:
        for s, k in sub_to_col.items():
            alpha_by_subject[s] = float(model.intercept_ + model.coef_[k])
    else:
        for s, k in sub_to_col.items():
            one_hot = np.zeros(len(uniq))
            one_hot[k] = 1.0
            alpha_by_subject[s] = float(model.predict(one_hot.reshape(1, -1))[0])

    return model, alpha_by_subject

def evaluate_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sub_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sub_test: np.ndarray,
    base_model: Any,
    intercept_model_type: str,
    lambda_candidates: List[float],
    lambda_tune_method: str = "loo",
    ridge_alpha: float = RIDGE_ALPHA_SUBJECT,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, np.ndarray]]:
    base_model.fit(X_train, y_train)
    y_pred_global_train = base_model.predict(X_train)
    y_pred_global_test = base_model.predict(X_test)

    residuals = y_train - y_pred_global_train
    _, alpha_by_sub = fit_subject_offsets(sub_train, residuals, model_type=intercept_model_type, ridge_alpha=ridge_alpha)

    offset_test = np.zeros(len(y_test), dtype=float)
    for j, s in enumerate(sub_test):
        offset_test[j] = alpha_by_sub.get(s, 0.0)

    offset_train = np.zeros(len(y_train), dtype=float)
    for i, s in enumerate(sub_train):
        offset_train[i] = alpha_by_sub.get(s, 0.0)

    best_lambda = LAMBDA_SHRINK
    best_mse = np.inf
    lambda_results = {}
    
    for lam in lambda_candidates:
        y_fit = y_pred_global_train + lam * offset_train
        mse = mean_squared_error(y_train, y_fit)
        lambda_results[lam] = y_pred_global_test + lam * offset_test
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam

    return y_pred_global_test, lambda_results[best_lambda], best_lambda, lambda_results

def run_evaluation(
    X_scaled: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    base_model_factory,
    mode_name: str,
    intercept_model_type: str,
    lambda_candidates: List[float],
    lambda_tune_method: str = "loo",
    ridge_alpha: float = RIDGE_ALPHA_SUBJECT,
) -> pd.DataFrame:
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train_s = X_scaled[train_idx]
        X_test_s = X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sub_train = subjects[train_idx]
        sub_test = subjects[test_idx]

        base_baseline = base_model_factory()
        pred_b, pred_p, best_lambda, _ = evaluate_one_fold(
            X_train_s, y_train, sub_train,
            X_test_s, y_test, sub_test,
            base_baseline,
            intercept_model_type,
            lambda_candidates,
            lambda_tune_method,
            ridge_alpha,
        )

        r2_b = r2_score(y_test, pred_b)
        mae_b = mean_absolute_error(y_test, pred_b)
        mape_b = mean_absolute_percentage_error(y_test, pred_b)

        r2_p = r2_score(y_test, pred_p)
        mae_p = mean_absolute_error(y_test, pred_p)
        mape_p = mean_absolute_percentage_error(y_test, pred_p)

        rows.append({
            "mode": mode_name,
            "intercept_model": intercept_model_type,
            "fold": fold,
            "method": "baseline",
            "r2": r2_b,
            "mae": mae_b,
            "mape": mape_b,
        })
        rows.append({
            "mode": mode_name,
            "intercept_model": intercept_model_type,
            "fold": fold,
            "method": "structured",
            "r2": r2_p,
            "mae": mae_p,
            "mape": mape_p,
            "lambda": best_lambda,
        })

    df = pd.DataFrame(rows)
    return df

def aggregate_and_print(df: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    agg = df.groupby(["intercept_model", "method"]).agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        mape_mean=("mape", "mean"),
        mape_std=("mape", "std"),
        lambda_mean=("lambda", "mean"),
        lambda_std=("lambda", "std"),
    ).reset_index()
    print(f"\n{mode_name}")
    print(agg.to_string(index=False))
    return agg

def main() -> None:
    X, y, subjects, tf_index = load_data()
    n = len(y)

    y_raw = y.copy()
    if TARGET_MODE_WITHIN == "winsor":
        y_within = winsorize_target(y, WINSOR_PCT[0], WINSOR_PCT[1])
    elif TARGET_MODE_WITHIN == "raw":
        y_within = y.copy()
    else:
        raise ValueError(f"Unknown TARGET_MODE_WITHIN: {TARGET_MODE_WITHIN}")

    if TARGET_MODE_WITHIN != "raw":
        changed = int(np.sum(~np.isclose(y_within, y_raw)))
        print(f"\n被试内去噪模式: {TARGET_MODE_WITHIN} | 变化样本数: {changed}/{len(y)}")

    if SCALER_MODE == "auto":
        use_standard = False
    elif SCALER_MODE == "standard":
        use_standard = True
    elif SCALER_MODE == "minmax":
        use_standard = False
    else:
        raise ValueError(f"Unknown SCALER_MODE: {SCALER_MODE}")

    scaler = StandardScaler() if use_standard else MinMaxScaler()
    X_scaled_20 = scaler.fit_transform(X)

    all_results = []

    print("=" * 80)
    print("多模型截距拟合对比实验（被试内）")
    print("=" * 80)
    print(f"截距模型类型: {INTERCEPT_MODEL_TYPES}")
    print("-" * 80)

    for intercept_model_type in INTERCEPT_MODEL_TYPES:
        print(f"\n正在测试截距模型: {intercept_model_type.upper()}")
        print("-" * 80)

        within_splits = build_within_splits(n, N_SPLITS, SHUFFLE, KF_RANDOM_STATE_WITHIN)
        df_within = run_evaluation(
            X_scaled_20, y_within, subjects,
            within_splits,
            lambda: make_base_model(RF_CONFIG_WITHIN),
            "within_subject",
            intercept_model_type,
            LAMBDA_CANDIDATES,
            LAMBDA_TUNE_METHOD,
        )
        agg_within = aggregate_and_print(df_within, f"被试内 - {intercept_model_type.upper()}")
        all_results.append(df_within)

    print("\n" + "=" * 80)
    print("Ridge Alpha 扫描（仅针对 Ridge 截距模型）")
    print("=" * 80)
    sweep_rows = []
    for ra in RIDGE_ALPHA_CANDIDATES:
        df_ra = run_evaluation(
            X_scaled_20, y_within, subjects,
            within_splits,
            lambda: make_base_model(RF_CONFIG_WITHIN),
            "within_subject",
            "ridge",
            LAMBDA_CANDIDATES,
            LAMBDA_TUNE_METHOD,
            ridge_alpha=ra,
        )
        r2_struct = df_ra[
            (df_ra["method"] == "structured") & (df_ra["intercept_model"] == "ridge")
        ]["r2"].mean()
        sweep_rows.append({"ridge_alpha": ra, "r2_mean_structured": r2_struct})
    df_sweep = pd.DataFrame(sweep_rows)
    best_ra = df_sweep.loc[df_sweep["r2_mean_structured"].idxmax(), "ridge_alpha"]
    best_r2 = df_sweep["r2_mean_structured"].max()
    print(f"\n被试内 RIDGE_ALPHA 扫描: 最佳 alpha={best_ra}, structured R2_mean={best_r2:.4f}")
    print(df_sweep.to_string(index=False))

    df_all = pd.concat(all_results, ignore_index=True)

    overall_agg = df_all.groupby(["mode", "intercept_model", "method"]).agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        lambda_mean=("lambda", "mean"),
        lambda_std=("lambda", "std"),
    ).reset_index()

    print("\n" + "=" * 80)
    print("总体对比汇总")
    print("=" * 80)
    print(overall_agg.to_string(index=False))

    baseline_within = overall_agg[
        (overall_agg["mode"] == "within_subject") & 
        (overall_agg["method"] == "baseline")
    ]["r2_mean"].values[0]
    structured_within = overall_agg[
        (overall_agg["mode"] == "within_subject") & 
        (overall_agg["method"] == "structured")
    ].sort_values("r2_mean", ascending=False).iloc[0]

    print("\n" + "=" * 80)
    print("最佳模型推荐")
    print("=" * 80)
    print(f"被试内最佳截距模型: {structured_within['intercept_model'].upper()}")
    print(f"  R²: {structured_within['r2_mean']:.4f} ± {structured_within['r2_std']:.4f}")
    print(f"  Lambda: {structured_within['lambda_mean']:.3f} ± {structured_within['lambda_std']:.3f}")
    print(f"  增益: +{structured_within['r2_mean'] - baseline_within:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as w:
        df_all.to_excel(w, sheet_name="all_results", index=False)
        overall_agg.to_excel(w, sheet_name="summary", index=False)
        df_sweep.to_excel(w, sheet_name="ridge_alpha_sweep", index=False)

    print(f"\n结果已保存: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
