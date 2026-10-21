"""模型脚本。

文件：model_comparison_regression_cross.py
统一使用项目内的相对路径。
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "model", "regression", "results")
FEAT_FILE = os.path.join(
    BASE_DIR,
    r"feature-enginnering/linear-features/experiment1.0-50features/X_结合相关互相关_spe_0.9_rfr_50.xlsx",
)
LABEL_FILE = os.path.join(BASE_DIR, r"feature-enginnering/linear-features/experiment1.0-50features/Y.xlsx")

SUBJECT_COL = "受试者编号"
FINGER_COL_CANDIDATES = ["手指（优食-0 非优中-1 非优尾-2 非优食-3）", "手指"]
DROP_SUB = []

N_SPLITS = 5
SHUFFLE = True
KF_RANDOM_STATE = 427

RF_CONFIG = {
    "criterion": "friedman_mse",
    "max_depth": 5,
    "max_features": 3,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 7,
    "random_state": 1435,
}

OUTPUT_FILE = os.path.join(RESULTS_DIR, "model_comparison_regression_cross.xlsx")

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

def safe_log(values):
    min_val = values.min()
    if min_val <= 0:
        return np.log(values - min_val + 1)
    return np.log(values)

def mean_absolute_percentage_error(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def load_data():
    df_feat = pd.read_excel(FEAT_FILE)
    tf = pd.read_excel(LABEL_FILE, index_col=0)

    if "阈值" in tf.columns:
        y_cont = tf["阈值"].to_numpy()
    else:
        y_cont = tf.iloc[:, 3].to_numpy()

    X_12 = df_feat[ORIGINAL_12_FEATURES].to_numpy()
    initial_log_array = [safe_log(df_feat[f].values) for f in INITIAL_LOG_FEATURES]
    X_initial_log = np.column_stack(initial_log_array)
    X_top3 = df_feat[TOP3_ADDITIONAL].to_numpy()
    new_log_array = [safe_log(df_feat[f].values) for f in NEW_LOG_FEATURES]
    X_new_log = np.column_stack(new_log_array)

    X_20 = np.hstack([X_12, X_initial_log, X_top3, X_new_log])

    subjects = tf[SUBJECT_COL].to_numpy()
    return X_20, y_cont, subjects

def build_cross_splits(subjects, n_splits, shuffle, random_state):
    subs = np.array(list(set(subjects)))
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = []
    for train_sub_idx, test_sub_idx in kf.split(subs):
        test_subs = subs.take(test_sub_idx).tolist()
        train_mask = ~np.isin(subjects, test_subs)
        test_mask = np.isin(subjects, test_subs)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        splits.append((train_idx, test_idx))
    return splits

def evaluate_models(X_scaled, y_cont, splits):
    models = {
        "RandomForest": RandomForestRegressor(**RF_CONFIG),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        ) if CatBoostRegressor else None,
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "SVR_RBF": SVR(kernel="rbf", C=1.0, gamma="scale"),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.001, random_state=42, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000),
    }

    models = {k: v for k, v in models.items() if v is not None}

    rows = []
    for name, model in models.items():
        r2_list = []
        mae_list = []
        mape_list = []

        for train_idx, test_idx in splits:
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_cont[train_idx], y_cont[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2_list.append(r2_score(y_test, y_pred))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            mape_list.append(mean_absolute_percentage_error(y_test, y_pred))

        rows.append(
            {
                "model": name,
                "r2_mean": float(np.mean(r2_list)),
                "r2_std": float(np.std(r2_list)),
                "mae_mean": float(np.mean(mae_list)),
                "mae_std": float(np.std(mae_list)),
                "mape_mean": float(np.mean(mape_list)),
                "mape_std": float(np.std(mape_list)),
            }
        )

    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False)

def main():
    X, y_cont, subjects = load_data()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    splits = build_cross_splits(subjects, N_SPLITS, SHUFFLE, KF_RANDOM_STATE)

    df_results = evaluate_models(X_scaled, y_cont, splits)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="model_comparison", index=False)

    print("=" * 60)
    print("Model comparison (regression, cross-subject)")
    print("=" * 60)
    print(f"KF_RANDOM_STATE = {KF_RANDOM_STATE}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(df_results)

if __name__ == "__main__":
    main()
