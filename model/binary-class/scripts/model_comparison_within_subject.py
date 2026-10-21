"""模型脚本。

文件：model_comparison_within_subject.py
统一使用项目内的相对路径。
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "model", "binary-class", "results")
FEAT_FILE = os.path.join(
    BASE_DIR,
    r"feature-enginnering/linear-features/experiment1.0-50features/X_结合相关互相关_spe_0.9_rfr_50.xlsx",
)
LABEL_FILE = os.path.join(BASE_DIR, r"feature-enginnering/linear-features/experiment1.0-50features/Y.xlsx")

SUBJECT_COL = "受试者编号"
FINGER_COL_CANDIDATES = ["手指（优食-0 非优中-1 非优尾-2 非优食-3）", "手指"]
DROP_SUB = []

# KFold 参数（最佳参数）
N_SPLITS = 5
SHUFFLE = True
KF_RANDOM_STATE = 3477

# RF 参数（最佳参数）
RF_CONFIG = {
    "n_estimators": 100,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 1612,
}

OUTPUT_FILE = os.path.join(RESULTS_DIR, "model_comparison_within_subject_results.xlsx")

MARGIN_MA = 15

# 被试内最佳特征
BEST_FEATURES = [
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

class my_KFold_Binary:
    """被试内 KFold。"""

    def __init__(self, tf, subject_col, finger_col, n_splits=5, shuffle=True, random_state=42, drop_sub=None):
        self.tf = tf
        self.subject_col = subject_col
        self.finger_col = finger_col
        self.n_splits = n_splits
        self.drop_sub = set(drop_sub or [])
        self.res = []

        tf_filtered = self.tf[~self.tf[self.subject_col].isin(self.drop_sub)]
        data = tf_filtered.index
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_i, test_i in kf.split(data):
            train_idx = data[train_i].tolist()
            test_idx = data[test_i].tolist()
            self.res.append((train_idx, test_idx))

    def split(self, X=None, y=None, groups=None):
        for train_idx, test_idx in self.res:
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def load_data():
    df_feat = pd.read_excel(FEAT_FILE)
    tf = pd.read_excel(LABEL_FILE, index_col=0)

    finger_col = None
    for col in FINGER_COL_CANDIDATES:
        if col in tf.columns:
            finger_col = col
            break

    if "阈值" in tf.columns:
        y_cont = tf["阈值"].to_numpy()
    else:
        y_cont = tf.iloc[:, 3].to_numpy()

    meta_cols = {"数据编号", SUBJECT_COL, "手指", FINGER_COL_CANDIDATES[0]}
    all_feature_cols = [c for c in df_feat.columns if c not in meta_cols]
    X_all = df_feat[all_feature_cols].to_numpy()
    X_best = df_feat[BEST_FEATURES].to_numpy()

    threshold = np.median(y_cont)
    y_binary = np.array([1 if val > threshold else 0 for val in y_cont])

    return X_all, X_best, y_binary, y_cont, tf, finger_col, all_feature_cols, threshold

def evaluate_models(X_scaled, y_binary, y_cont, threshold, cv):
    models = {
        "RandomForest": RandomForestClassifier(**RF_CONFIG),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            iterations=300,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ) if XGBClassifier else None,
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ) if LGBMClassifier else None,
        "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }

    models = {k: v for k, v in models.items() if v is not None}

    rows = []
    def split_xy(X, indices):
        if hasattr(X, "iloc"):
            return X.iloc[indices]
        return X[indices]

    for name, model in models.items():
        acc_list = []
        acc_margin_list = []
        f1_list = []
        for train_idx, test_idx in cv.split():
            X_train, X_test = split_xy(X_scaled, train_idx), split_xy(X_scaled, test_idx)
            y_train, y_test = y_binary[train_idx], y_binary[test_idx]
            y_cont_test = y_cont[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_list.append(accuracy_score(y_test, y_pred))
            margin_correct = np.where(
                np.abs(y_cont_test - threshold) <= MARGIN_MA,
                True,
                y_pred == y_test,
            )
            acc_margin_list.append(float(np.mean(margin_correct)))
            f1_list.append(f1_score(y_test, y_pred, zero_division=0))

        rows.append(
            {
                "model": name,
                "acc_mean": float(np.mean(acc_list)),
                "acc_std": float(np.std(acc_list)),
                "acc_margin_mean": float(np.mean(acc_margin_list)),
                "acc_margin_std": float(np.std(acc_margin_list)),
                "f1_mean": float(np.mean(f1_list)),
                "f1_std": float(np.std(f1_list)),
            }
        )

    return pd.DataFrame(rows).sort_values("acc_margin_mean", ascending=False)

def main():
    X_all, X_best, y_binary, y_cont, tf, finger_col, all_feature_cols, threshold = load_data()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_scaled_sub = X_scaled[:, [all_feature_cols.index(f) for f in BEST_FEATURES]]
    X_scaled_sub = pd.DataFrame(X_scaled_sub, columns=BEST_FEATURES)

    cv = my_KFold_Binary(
        tf,
        SUBJECT_COL,
        finger_col,
        n_splits=N_SPLITS,
        shuffle=SHUFFLE,
        random_state=KF_RANDOM_STATE,
        drop_sub=DROP_SUB,
    )

    df_results = evaluate_models(X_scaled_sub, y_binary, y_cont, threshold, cv)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="model_comparison", index=False)

    print("=" * 60)
    print("Model comparison (within-subject)")
    print("=" * 60)
    print(f"Features: {len(BEST_FEATURES)}")
    print(f"Threshold (median): {threshold:.6f}")
    print(f"Margin: {MARGIN_MA} mA")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(df_results)

if __name__ == "__main__":
    main()
