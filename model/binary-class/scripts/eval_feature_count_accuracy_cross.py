"""模型脚本。

文件：eval_feature_count_accuracy_cross.py
统一使用项目内的相对路径。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import rf_binary_11features_cross as base

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_XLSX = os.path.join(PARENT_DIR, "results", "feature_count_accuracy_cross.xlsx")
OUTPUT_CSV = os.path.join(PARENT_DIR, "results", "feature_count_accuracy_cross.csv")

def get_rf_importance_order(X, y, feature_cols):
    rf = RandomForestClassifier(**base.RF_CONFIG)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    order = importances.sort_values(ascending=False).index.tolist()
    return order, importances

def build_selected_features(k, base_features, all_order, importances):
    if k == 0:
        return []

    base_ranked = sorted(base_features, key=lambda f: importances.get(f, 0), reverse=True)
    if k <= len(base_ranked):
        keep = set(base_ranked[:k])
        return [f for f in base_features if f in keep]

    remaining = [f for f in all_order if f not in base_features]
    need = k - len(base_features)
    return base_features + remaining[:need]

def build_cross_splits(subjects, n_splits, shuffle, random_state):
    subs = np.array(list(set(subjects)))
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = []
    for train_sub_idx, test_sub_idx in kf.split(subs):
        test_subs = subs.take(test_sub_idx).tolist()
        train_mask = ~np.isin(subjects, test_subs)
        test_mask = np.isin(subjects, test_subs)
        splits.append((train_mask, test_mask))
    return splits

def majority_acc(y_train, y_test):
    majority = 1 if np.mean(y_train) >= 0.5 else 0
    y_pred = np.full_like(y_test, fill_value=majority)
    return float(np.mean(y_pred == y_test))

def main():
    X_all, y_binary, y_cont, subjects, threshold, feature_cols = base.load_data()

    scaler = MinMaxScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    cross_splits = build_cross_splits(subjects, base.N_SPLITS, base.SHUFFLE, base.KF_RANDOM_STATE)
    all_order, importances = get_rf_importance_order(X_all_scaled, y_binary, feature_cols)

    records = []
    for k in range(0, len(feature_cols) + 1):
        if k == 0:
            fold_acc = []
            for train_mask, test_mask in cross_splits:
                y_train, y_test = y_binary[train_mask], y_binary[test_mask]
                fold_acc.append(majority_acc(y_train, y_test))
            acc_mean = float(np.mean(fold_acc))
        else:
            selected = build_selected_features(k, base.SELECTED_12_FEATURES, all_order, importances)
            idxs = [feature_cols.index(f) for f in selected]
            X_scaled_sub = X_all_scaled[:, idxs]
            result = base.cross_subject_validation(
                X_scaled_sub,
                y_binary,
                y_cont,
                subjects,
                threshold,
                margin=0,
            )
            acc_mean = result["acc_mean"]

        records.append({"k": k, "acc": acc_mean})
        print(f"k={k:2d} -> acc={acc_mean:.6f}")

    df = pd.DataFrame.from_records(records)
    df.to_excel(OUTPUT_XLSX, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUTPUT_XLSX}")
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
