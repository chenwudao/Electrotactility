"""模型脚本。

文件：eval_feature_combo_accuracy.py
统一使用项目内的相对路径。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import rf_binary_12features_within as within_base
import rf_binary_11features_cross as cross_base

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ADD_REMOVE_CSV = os.path.join(PARENT_DIR, "results", "rf_combo_add_remove_margin15.csv")
OUTPUT_CSV = os.path.join(PARENT_DIR, "results", "rf_accuracy_by_feature_combo.csv")
OUTPUT_PNG = os.path.join(PARENT_DIR, "results", "rf_accuracy_by_feature_combo.png")

COMBOS = [
    ("Overall", "解剖"),
    ("Local", "图像"),
    ("Bioelectrical", "电学"),
    ("Overall+Local", "解剖+图像"),
    ("Overall+Bioelectrical", "电学+解剖"),
    ("Local+Bioelectrical", "电学+图像"),
    ("All Three", "电学+解剖+图像"),
]

def apply_y_axis_ruler_ticks(ax):
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", direction="in", length=4, width=1.8, left=True, right=False)
    ax.tick_params(axis="y", which="minor", direction="in", length=2, width=1.6, left=True, right=False)

def load_best_rows(path):
    if not os.path.exists(path):
        if os.path.exists(OUTPUT_CSV):
            df_best = pd.read_csv(OUTPUT_CSV)
            if "All Three" not in df_best["combo"].values:
                df_best = add_all_three_row(df_best)
            return df_best
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if "combo" not in df.columns:
        raise ValueError("Expected column 'combo' in add/remove CSV")

    best_rows = []
    for combo, gdf in df.groupby("combo"):
        best_row = gdf.loc[gdf["within_acc_margin15"].idxmax()]
        best_rows.append(best_row)

    df_best = pd.DataFrame(best_rows)
    if "All Three" not in df_best["combo"].values:
        df_best = add_all_three_row(df_best)

    return df_best

def add_all_three_row(df_best):
    X_all, y_binary, y_cont, tf, finger_col, all_feature_cols, threshold = within_base.load_data()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all)

    within_idxs = [all_feature_cols.index(f) for f in within_base.SELECTED_12_FEATURES]
    X_within = X_scaled[:, within_idxs]

    cv = within_base.my_KFold_Binary(
        tf,
        within_base.SUBJECT_COL,
        finger_col,
        n_splits=within_base.N_SPLITS,
        shuffle=within_base.SHUFFLE,
        random_state=within_base.KF_RANDOM_STATE,
        drop_sub=within_base.DROP_SUB,
    )
    results = within_base.evaluate_model(X_within, y_binary, y_cont, tf, finger_col, threshold, cv)
    within_acc = next(r["acc_mean"] for r in results if r["margin"] == 15)

    X_all_c, y_binary_c, y_cont_c, subjects, threshold_c, all_feature_cols_c = cross_base.load_data()
    scaler_c = MinMaxScaler()
    X_scaled_c = scaler_c.fit_transform(X_all_c)
    cross_idxs = [all_feature_cols_c.index(f) for f in cross_base.SELECTED_11_FEATURES]
    X_cross = X_scaled_c[:, cross_idxs]
    cross_result = cross_base.cross_subject_validation(
        X_cross,
        y_binary_c,
        y_cont_c,
        subjects,
        threshold_c,
        margin=15,
    )
    cross_acc = cross_result["acc_mean"]

    extra = {
        "combo": "All Three",
        "group_key": "电学+解剖+图像",
        "k": len(within_base.SELECTED_12_FEATURES),
        "within_acc_margin15": within_acc,
        "cross_acc_margin15": cross_acc,
        "within_n_features": len(within_base.SELECTED_12_FEATURES),
        "cross_n_features": len(cross_base.SELECTED_11_FEATURES),
    }
    return pd.concat([df_best, pd.DataFrame([extra])], ignore_index=True)

def main():
    df = load_best_rows(ADD_REMOVE_CSV)
    df = df.sort_values("combo", key=lambda s: s.map({name: i for i, (name, _) in enumerate(COMBOS)}))
    df["combo_display"] = df["combo"].replace({"All Three": "All"})
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    sns.set_theme(
        style="white",
        font_scale=0.95,
        rc={
            "font.family": "Times New Roman",
            "axes.labelsize": 24,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.4,
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
        },
    )
    fig, ax = plt.subplots(1, 1, figsize=(15.0, 6.0))

    x = np.arange(len(df)) * 1.35
    width = 0.36

    ax.bar(
        x - width / 2,
        df["within_acc_margin15"] * 100,
        width,
        label="Non-Cross Subject",
        color="#f7c58a",
    )
    ax.bar(
        x + width / 2,
        df["cross_acc_margin15"] * 100,
        width,
        label="Cross Subject",
        color="#9ad49b",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["combo_display"], rotation=0)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(x[0] - 0.8, x[-1] + 0.8)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(False)
    ax.tick_params(axis="both", colors="black", width=1.2)
    apply_y_axis_ruler_ticks(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.4)

    for i, (wm, cm) in enumerate(zip(df["within_acc_margin15"], df["cross_acc_margin15"])):
        ax.text(x[i] - width / 2, wm * 100 + 1.0, f"{wm * 100:.2f}", ha="center", va="bottom", fontsize=11)
        ax.text(x[i] + width / 2, cm * 100 + 1.0, f"{cm * 100:.2f}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
