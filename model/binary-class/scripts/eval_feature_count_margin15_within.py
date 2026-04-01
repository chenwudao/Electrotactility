"""模型脚本。

文件：eval_feature_count_margin15_within.py
统一使用项目内的相对路径。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import rf_binary_12features_within as base

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_XLSX = os.path.join(PARENT_DIR, "results", "feature_count_margin15_within.xlsx")
OUTPUT_CSV = os.path.join(PARENT_DIR, "results", "feature_count_margin15_within.csv")

MARGIN = 15

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

def majority_margin_acc(y_train, y_test, y_cont_test, threshold):
    majority = 1 if np.mean(y_train) >= 0.5 else 0
    y_pred = np.full_like(y_test, fill_value=majority)
    acc, _ = base.tolerant_scores(y_test, y_pred, y_cont_test, threshold, MARGIN)
    return acc

def main():
    X_all, y_binary, y_cont, tf, finger_col, feature_cols, threshold = base.load_data()

    scaler = MinMaxScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    cv = base.my_KFold_Binary(
        tf,
        base.SUBJECT_COL,
        finger_col,
        n_splits=base.N_SPLITS,
        shuffle=base.SHUFFLE,
        random_state=base.KF_RANDOM_STATE,
        drop_sub=base.DROP_SUB,
    )
    splits = list(cv.split())

    all_order, importances = get_rf_importance_order(X_all_scaled, y_binary, feature_cols)

    records = []
    for k in range(0, len(feature_cols) + 1):
        if k == 0:
            fold_acc = []
            for train_idx, test_idx in splits:
                y_train, y_test = y_binary[train_idx], y_binary[test_idx]
                y_cont_test = y_cont[test_idx]
                fold_acc.append(majority_margin_acc(y_train, y_test, y_cont_test, threshold))
            acc_mean = float(np.mean(fold_acc))
        else:
            selected = build_selected_features(k, base.SELECTED_12_FEATURES, all_order, importances)
            idxs = [feature_cols.index(f) for f in selected]
            X_scaled_sub = X_all_scaled[:, idxs]
            results = base.evaluate_model(X_scaled_sub, y_binary, y_cont, tf, finger_col, threshold, cv)
            acc_mean = next(r["acc_mean"] for r in results if r["margin"] == MARGIN)

        records.append({"k": k, "acc": acc_mean})
        print(f"k={k:2d} -> acc={acc_mean:.6f}")

    df = pd.DataFrame.from_records(records)
    df.to_excel(OUTPUT_XLSX, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUTPUT_XLSX}")
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
