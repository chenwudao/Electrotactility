"""模型脚本。

文件：regression_structured_prior_model.py
统一使用项目内的相对路径。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, QuantileRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_FILE = os.path.join(
    BASE_DIR,
    r"feature-enginnering/linear-features/experiment1.0-50features/X_结合相关互相关_spe_0.9_rfr_50.xlsx",
)
LABEL_FILE = os.path.join(BASE_DIR, r"feature-enginnering/linear-features/experiment1.0-50features/Y.xlsx")
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

RIDGE_ALPHA_SUBJECT = 50.0
RIDGE_ALPHA_CANDIDATES_WITHIN = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 10.0]
USE_TEMPORAL_SMOOTH_WITHIN = False
TEMPORAL_SMOOTH_WINDOW = 3

TARGET_MODE_WITHIN = "winsor"
WINSOR_PCT = (0.02, 0.98)
SUBJECT_MAD_K = 3.5
EXPORT_TARGET_DIAG = True

BASE_MODEL_TYPE = "rf"
HUBER_CONFIG = {
    "epsilon": 1.35,
    "alpha": 0.0001,
    "max_iter": 1000,
    "tol": 1e-4,
}
QUANTILE_CONFIG = {
    "quantile": 0.5,
    "alpha": 0.0001,
    "solver": "highs",
}

SCALER_MODE = "auto"

OUTPUT_DIR = SCRIPT_DIR
OUTPUT_WITHIN = os.path.join(OUTPUT_DIR, "regression_structured_prior_within.xlsx")
TARGET_DIAG_FILE = os.path.join(OUTPUT_DIR, "target_stability_summary.csv")

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

def smooth_predictions_by_subject(
    y_pred: np.ndarray, subjects: np.ndarray, window: int = 3
) -> np.ndarray:
    out = np.copy(y_pred)
    for s in np.unique(subjects):
        mask = subjects == s
        idx = np.where(mask)[0]
        vals = y_pred[mask].astype(float)
        n = len(vals)
        if n < 2:
            continue
        w = min(window, n)
        kernel = np.ones(w) / w
        smoothed = np.convolve(vals, kernel, mode="same")
        out[mask] = smoothed
    return out

def summarize_target_stability(y: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"subject": subjects, "y": y})
    agg = df.groupby("subject")["y"].agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    ).reset_index()
    agg["iqr"] = agg["q3"] - agg["q1"]
    return agg

def winsorize_target(y: np.ndarray, pct_low: float, pct_high: float) -> np.ndarray:
    low, high = np.quantile(y, [pct_low, pct_high])
    return np.clip(y, low, high)

def subject_mad_clip(y: np.ndarray, subjects: np.ndarray, k: float) -> np.ndarray:
    out = y.astype(float).copy()
    for s in np.unique(subjects):
        mask = subjects == s
        vals = out[mask]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad <= 0:
            continue
        lower = med - k * mad
        upper = med + k * mad
        out[mask] = np.clip(vals, lower, upper)
    return out

def make_base_model(model_type: str, rf_config: Dict[str, Any]) -> Any:
    if model_type == "rf":
        return RandomForestRegressor(**rf_config)
    if model_type == "huber":
        return HuberRegressor(**HUBER_CONFIG)
    if model_type == "quantile":
        return QuantileRegressor(**QUANTILE_CONFIG)
    raise ValueError(f"Unknown BASE_MODEL_TYPE: {model_type}")

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

def fit_subject_offsets_ridge(
    train_subject_ids: np.ndarray,
    residuals: np.ndarray,
    alpha: float = RIDGE_ALPHA_SUBJECT,
) -> Tuple[Ridge, Dict[Any, float]]:
    uniq = np.unique(train_subject_ids)
    n_train = len(train_subject_ids)
    Z = np.zeros((n_train, len(uniq)), dtype=float)
    sub_to_col = {s: i for i, s in enumerate(uniq)}
    for i, s in enumerate(train_subject_ids):
        Z[i, sub_to_col[s]] = 1.0

    ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    ridge.fit(Z, residuals)
    alpha_by_subject = {}
    for s, k in sub_to_col.items():
        alpha_by_subject[s] = float(ridge.intercept_ + ridge.coef_[k])
    return ridge, alpha_by_subject

def evaluate_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sub_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sub_test: np.ndarray,
    base_model: Any,
    use_partial_pooling: bool,
    ridge_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    base_model.fit(X_train, y_train)
    y_pred_global_train = base_model.predict(X_train)
    y_pred_global_test = base_model.predict(X_test)

    if not use_partial_pooling:
        return y_pred_global_test, y_pred_global_test

    residuals = y_train - y_pred_global_train
    _, alpha_by_sub = fit_subject_offsets_ridge(sub_train, residuals, alpha=ridge_alpha)

    offset_test = np.zeros(len(y_test), dtype=float)
    for j, s in enumerate(sub_test):
        offset_test[j] = alpha_by_sub.get(s, 0.0)
    y_pred_pooling = y_pred_global_test + offset_test
    return y_pred_global_test, y_pred_pooling

def run_evaluation(
    X_scaled: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    base_model_factory,
    mode_name: str,
    feature_tag: str,
    ridge_alpha: float = RIDGE_ALPHA_SUBJECT,
    use_temporal_smooth: bool = False,
    temporal_window: int = TEMPORAL_SMOOTH_WINDOW,
) -> pd.DataFrame:
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train_s = X_scaled[train_idx]
        X_test_s = X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sub_train = subjects[train_idx]
        sub_test = subjects[test_idx]

        base_baseline = base_model_factory()
        pred_b, pred_p = evaluate_one_fold(
            X_train_s, y_train, sub_train,
            X_test_s, y_test, sub_test,
            base_baseline,
            use_partial_pooling=False,
            ridge_alpha=ridge_alpha,
        )
        if use_temporal_smooth:
            pred_b = smooth_predictions_by_subject(pred_b, sub_test, window=temporal_window)
        r2_b = r2_score(y_test, pred_b)
        mae_b = mean_absolute_error(y_test, pred_b)
        mape_b = mean_absolute_percentage_error(y_test, pred_b)

        base_pool = base_model_factory()
        pred_b2, pred_p2 = evaluate_one_fold(
            X_train_s, y_train, sub_train,
            X_test_s, y_test, sub_test,
            base_pool,
            use_partial_pooling=True,
            ridge_alpha=ridge_alpha,
        )
        if use_temporal_smooth:
            pred_p2 = smooth_predictions_by_subject(pred_p2, sub_test, window=temporal_window)
        r2_p = r2_score(y_test, pred_p2)
        mae_p = mean_absolute_error(y_test, pred_p2)
        mape_p = mean_absolute_percentage_error(y_test, pred_p2)

        rows.append({
            "mode": mode_name,
            "feature_set": feature_tag,
            "fold": fold,
            "method": "baseline",
            "r2": r2_b,
            "mae": mae_b,
            "mape": mape_b,
        })
        rows.append({
            "mode": mode_name,
            "feature_set": feature_tag,
            "fold": fold,
            "method": "structured_partial_pooling",
            "r2": r2_p,
            "mae": mae_p,
            "mape": mape_p,
        })

    df = pd.DataFrame(rows)
    return df

def aggregate_and_print(df: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    agg = df.groupby(["feature_set", "method"]).agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        mape_mean=("mape", "mean"),
        mape_std=("mape", "std"),
    ).reset_index()
    print(f"\n{mode_name}")
    print(agg.to_string(index=False))
    return agg

def main() -> None:
    X, y, subjects, tf_index = load_data()
    n = len(y)

    if EXPORT_TARGET_DIAG:
        diag = summarize_target_stability(y, subjects)
        diag.to_csv(TARGET_DIAG_FILE, index=False, encoding="utf-8-sig")
        print(f"\n目标值稳定性统计已保存: {TARGET_DIAG_FILE}")

    y_raw = y.copy()
    if TARGET_MODE_WITHIN == "winsor":
        y_within = winsorize_target(y, WINSOR_PCT[0], WINSOR_PCT[1])
    elif TARGET_MODE_WITHIN == "subject_mad":
        y_within = subject_mad_clip(y, subjects, k=SUBJECT_MAD_K)
    elif TARGET_MODE_WITHIN == "raw":
        y_within = y.copy()
    else:
        raise ValueError(f"Unknown TARGET_MODE_WITHIN: {TARGET_MODE_WITHIN}")

    if TARGET_MODE_WITHIN != "raw":
        changed = int(np.sum(~np.isclose(y_within, y_raw)))
        print(f"\n被试内去噪模式: {TARGET_MODE_WITHIN} | 变化样本数: {changed}/{len(y)}")

    if SCALER_MODE == "auto":
        use_standard = BASE_MODEL_TYPE in {"huber", "quantile"}
    elif SCALER_MODE == "standard":
        use_standard = True
    elif SCALER_MODE == "minmax":
        use_standard = False
    else:
        raise ValueError(f"Unknown SCALER_MODE: {SCALER_MODE}")

    scaler = StandardScaler() if use_standard else MinMaxScaler()
    X_scaled_20 = scaler.fit_transform(X)

    within_splits = build_within_splits(
        n, N_SPLITS, SHUFFLE, KF_RANDOM_STATE_WITHIN
    )
    df_within = run_evaluation(
        X_scaled_20, y_within, subjects,
        within_splits,
        lambda: make_base_model(BASE_MODEL_TYPE, RF_CONFIG_WITHIN),
        "within_subject",
        feature_tag="20",
        ridge_alpha=RIDGE_ALPHA_SUBJECT,
        use_temporal_smooth=USE_TEMPORAL_SMOOTH_WITHIN,
        temporal_window=TEMPORAL_SMOOTH_WINDOW,
    )
    agg_within = aggregate_and_print(df_within, "=== 被试内 (within-subject) ===")

    sweep_rows = []
    for ra in RIDGE_ALPHA_CANDIDATES_WITHIN:
        df_ra = run_evaluation(
            X_scaled_20, y_within, subjects,
            within_splits,
            lambda: make_base_model(BASE_MODEL_TYPE, RF_CONFIG_WITHIN),
            "within_subject",
            feature_tag="20",
            ridge_alpha=ra,
            use_temporal_smooth=USE_TEMPORAL_SMOOTH_WITHIN,
            temporal_window=TEMPORAL_SMOOTH_WINDOW,
        )
        df_struct = df_ra[
            (df_ra["method"] == "structured_partial_pooling") & (df_ra["feature_set"] == "20")
        ]
        r2_struct = df_struct["r2"].mean()
        mae_struct = df_struct["mae"].mean()
        mape_struct = df_struct["mape"].mean()
        sweep_rows.append({
            "ridge_alpha": ra,
            "r2_mean_structured": r2_struct,
            "mae_mean_structured": mae_struct,
            "mape_mean_structured": mape_struct,
        })
    df_sweep = pd.DataFrame(sweep_rows)
    best_ra = df_sweep.loc[df_sweep["r2_mean_structured"].idxmax(), "ridge_alpha"]
    best_r2 = df_sweep["r2_mean_structured"].max()
    print(f"\n被试内 RIDGE_ALPHA 扫描: 最佳 alpha={best_ra}, structured R2_mean={best_r2:.4f}")
    print(df_sweep.to_string(index=False))
    alpha10_rows = df_sweep[np.isclose(df_sweep["ridge_alpha"], 10.0)]
    if not alpha10_rows.empty:
        row10 = alpha10_rows.iloc[0]
        print(
            f"\n被试内 alpha=10.0 指标: "
            f"R2={row10['r2_mean_structured']:.4f}, "
            f"MAE={row10['mae_mean_structured']:.4f}, "
            f"MAPE={row10['mape_mean_structured']:.4f}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_WITHIN, engine="openpyxl") as w:
        df_within.to_excel(w, sheet_name="folds", index=False)
        agg_within.to_excel(w, sheet_name="summary", index=False)
        df_sweep.to_excel(w, sheet_name="ridge_alpha_sweep", index=False)

    print("\n结果已保存:")
    print(f"  被试内: {OUTPUT_WITHIN}")
    print("\n说明: structured_partial_pooling = 全局模型 + 强正则被试截距；")
    print("  被试内测试时使用 α_i。")
    print(f"\n模型类型: {BASE_MODEL_TYPE} | 被试内目标模式: {TARGET_MODE_WITHIN}")

if __name__ == "__main__":
    main()
