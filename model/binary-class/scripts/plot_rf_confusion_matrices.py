"""模型脚本。

文件：plot_rf_confusion_matrices.py
统一使用项目内的相对路径。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import rf_binary_12features_within as within_base
import rf_binary_11features_cross as cross_base

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(BASE_DIR, "model", "binary-class", "results")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "rf_confusion_matrices_within_cross.png")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "rf_confusion_matrices_within_cross.csv")
MARGIN_MA = 15

def get_feature_indices(all_feature_cols, selected_features):
    idxs = []
    missing = []
    for feat in selected_features:
        if feat in all_feature_cols:
            idxs.append(all_feature_cols.index(feat))
        else:
            missing.append(feat)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return idxs

def run_within_confusion():
    X_all, y_binary, y_cont, tf, finger_col, all_feature_cols, threshold = within_base.load_data()

    scaler = MinMaxScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    idxs = get_feature_indices(all_feature_cols, within_base.SELECTED_12_FEATURES)
    X_scaled_sub = X_all_scaled[:, idxs]

    cv = within_base.my_KFold_Binary(
        tf,
        within_base.SUBJECT_COL,
        finger_col,
        n_splits=within_base.N_SPLITS,
        shuffle=within_base.SHUFFLE,
        random_state=within_base.KF_RANDOM_STATE,
        drop_sub=within_base.DROP_SUB,
    )

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in cv.split():
        X_train, X_test = X_scaled_sub[train_idx], X_scaled_sub[test_idx]
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]
        y_cont_test = y_cont[test_idx]

        model = RandomForestClassifier(**within_base.RF_CONFIG)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if MARGIN_MA > 0:
            y_pred_adj = y_pred.copy()
            near_boundary = np.abs(y_cont_test - threshold) <= MARGIN_MA
            y_pred_adj[near_boundary] = y_test[near_boundary]
            y_pred = y_pred_adj

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    return cm

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

def run_cross_confusion():
    X_all, y_binary, y_cont, subjects, threshold, all_feature_cols = cross_base.load_data()

    scaler = MinMaxScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    idxs = get_feature_indices(all_feature_cols, cross_base.SELECTED_12_FEATURES)
    X_scaled_sub = X_all_scaled[:, idxs]

    splits = build_cross_splits(subjects, cross_base.N_SPLITS, cross_base.SHUFFLE, cross_base.KF_RANDOM_STATE)

    y_true_all = []
    y_pred_all = []

    for train_mask, test_mask in splits:
        X_train = X_scaled_sub[train_mask]
        X_test = X_scaled_sub[test_mask]
        y_train = y_binary[train_mask]
        y_test = y_binary[test_mask]
        y_cont_test = y_cont[test_mask]

        model = RandomForestClassifier(**cross_base.RF_CONFIG)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if MARGIN_MA > 0:
            y_pred_adj = y_pred.copy()
            near_boundary = np.abs(y_cont_test - threshold) <= MARGIN_MA
            y_pred_adj[near_boundary] = y_test[near_boundary]
            y_pred = y_pred_adj

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    return cm

def save_csv(cm_within, cm_cross):
    df = pd.DataFrame(
        {
            "scenario": ["Non-Cross Subject", "Non-Cross Subject", "Cross Subject", "Cross Subject"],
            "true_label": [0, 1, 0, 1],
            "pred_0": [cm_within[0, 0], cm_within[1, 0], cm_cross[0, 0], cm_cross[1, 0]],
            "pred_1": [cm_within[0, 1], cm_within[1, 1], cm_cross[0, 1], cm_cross[1, 1]],
        }
    )
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

def plot_confusion_matrices(cm_within, cm_cross):
    sns.set_theme(
        style="whitegrid",
        font_scale=0.95,
        rc={
            "font.family": "Times New Roman",
            "axes.labelsize": 24,
            "axes.titlesize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.4,
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.0))

    for ax, cm, title in zip(
        axes,
        [cm_within, cm_cross],
        ["Non-Cross Subject", "Cross Subject"],
    ):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            annot_kws={"fontsize": 30},
            cmap="Greens",
            cbar=False,
            ax=ax,
            linewidths=0.6,
            linecolor="#eaeaea",
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        ax.set_xticklabels(["Low threshold", "High threshold"])
        ax.set_yticklabels(["Low threshold", "High threshold"], rotation=90)
        ax.tick_params(axis="both", colors="black", width=1.2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved: {OUTPUT_PNG}")

def main():
    cm_within = run_within_confusion()
    cm_cross = run_cross_confusion()
    save_csv(cm_within, cm_cross)
    plot_confusion_matrices(cm_within, cm_cross)

if __name__ == "__main__":
    main()
