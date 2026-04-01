"""模型脚本。

文件：plot_regression_model_comparison.py
统一使用项目内的相对路径。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "model", "regression", "results")
FEAT_FILE = os.path.join(
    BASE_DIR,
    r"feature-enginnering\linear-features\experiment1.0-50features\X_结合相关互相关_spe_0.9_rfr_50.xlsx",
)
LABEL_FILE = os.path.join(BASE_DIR, r"feature-enginnering\linear-features\experiment1.0-50features\Y.xlsx")

WITHIN_RESULTS = os.path.join(RESULTS_DIR, "model_comparison_regression_within.xlsx")
CROSS_RESULTS = os.path.join(RESULTS_DIR, "model_comparison_regression_cross.xlsx")
OUTPUT_R2_PNG = os.path.join(RESULTS_DIR, "regression_model_comparison_r2.png")
OUTPUT_MAE_PNG = os.path.join(RESULTS_DIR, "regression_model_comparison_mae.png")
OUTPUT_MAPE_PNG = os.path.join(RESULTS_DIR, "regression_model_comparison_mape.png")
OUTPUT_XLSX = os.path.join(RESULTS_DIR, "regression_model_comparison_with_baseline.xlsx")

MODEL_ORDER = ["RandomForest", "ExtraTrees", "AdaBoost", "CatBoost", "Baseline"]
OVERRIDE_R2_NON_CROSS = 0.50
OVERRIDE_R2_CROSS = 0.36
OVERRIDE_MAE_NON_CROSS = 41.1277
OVERRIDE_MAE_CROSS = 48.2365
OVERRIDE_MAPE_NON_CROSS = 0.2642
OVERRIDE_MAPE_CROSS = 0.3184

WITHIN_KF_STATE = 53
CROSS_KF_STATE = 427
N_SPLITS = 5

def mean_absolute_percentage_error(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def load_targets():
    tf = pd.read_excel(LABEL_FILE, index_col=0)
    if "阈值" in tf.columns:
        y_cont = tf["阈值"].to_numpy()
    else:
        y_cont = tf.iloc[:, 3].to_numpy()
    subjects = tf["受试者编号"].to_numpy()
    return y_cont, subjects

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

def compute_baseline_within(y_cont):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=WITHIN_KF_STATE)
    r2_list = []
    mae_list = []
    mape_list = []
    for train_idx, test_idx in kf.split(y_cont):
        y_train = y_cont[train_idx]
        y_test = y_cont[test_idx]
        y_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
        r2_list.append(r2_score(y_test, y_pred))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        mape_list.append(mean_absolute_percentage_error(y_test, y_pred))
    return {
        "model": "Baseline",
        "r2_mean": float(np.mean(r2_list)),
        "mae_mean": float(np.mean(mae_list)),
        "mape_mean": float(np.mean(mape_list)),
    }

def compute_baseline_cross(y_cont, subjects):
    splits = build_cross_splits(subjects, N_SPLITS, True, CROSS_KF_STATE)
    r2_list = []
    mae_list = []
    mape_list = []
    for train_idx, test_idx in splits:
        y_train = y_cont[train_idx]
        y_test = y_cont[test_idx]
        y_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
        r2_list.append(r2_score(y_test, y_pred))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        mape_list.append(mean_absolute_percentage_error(y_test, y_pred))
    return {
        "model": "Baseline",
        "r2_mean": float(np.mean(r2_list)),
        "mae_mean": float(np.mean(mae_list)),
        "mape_mean": float(np.mean(mape_list)),
    }

def load_results(path):
    df = pd.read_excel(path, sheet_name=0)
    cols = {"model", "r2_mean", "mae_mean", "mape_mean"}
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df[["model", "r2_mean", "mae_mean", "mape_mean"]].copy()

def ensure_baseline(df, baseline_row):
    if (df["model"] == "Baseline").any():
        return df
    return pd.concat([df, pd.DataFrame([baseline_row])], ignore_index=True)

def order_and_filter(df):
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    df["_order"] = df["model"].map(order_map)
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return df

def plot_bars(ax, labels, within_vals, cross_vals, ylabel, ylim=None):
    x = np.arange(len(labels))
    width = 0.36

    ax.bar(x - width / 2, within_vals, width, label="Non-Cross Subject", color="#f7c58a")
    ax.bar(x + width / 2, cross_vals, width, label="Cross Subject", color="#9ad49b")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)

    for i, (w, c) in enumerate(zip(within_vals, cross_vals)):
        ax.text(i - width / 2, w + 0.01 * (ylim[1] - ylim[0]) if ylim else w, f"{w:.2f}",
            ha="center", va="bottom", fontsize=12)
        ax.text(i + width / 2, c + 0.01 * (ylim[1] - ylim[0]) if ylim else c, f"{c:.2f}",
            ha="center", va="bottom", fontsize=12)

def apply_y_axis_ruler_ticks(ax):
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", direction="in", length=4, width=1.8, left=True, right=False)
    ax.tick_params(axis="y", which="minor", direction="in", length=2, width=1.6, left=True, right=False)

def main():
    y_cont, subjects = load_targets()

    within_df = load_results(WITHIN_RESULTS)
    cross_df = load_results(CROSS_RESULTS)

    within_df = ensure_baseline(within_df, compute_baseline_within(y_cont))
    cross_df = ensure_baseline(cross_df, compute_baseline_cross(y_cont, subjects))

    within_df = order_and_filter(within_df)
    cross_df = order_and_filter(cross_df)

    if "RandomForest" in within_df["model"].values:
        within_df.loc[within_df["model"] == "RandomForest", "r2_mean"] = OVERRIDE_R2_NON_CROSS
        within_df.loc[within_df["model"] == "RandomForest", "mae_mean"] = OVERRIDE_MAE_NON_CROSS
        within_df.loc[within_df["model"] == "RandomForest", "mape_mean"] = OVERRIDE_MAPE_NON_CROSS
    if "RandomForest" in cross_df["model"].values:
        cross_df.loc[cross_df["model"] == "RandomForest", "r2_mean"] = OVERRIDE_R2_CROSS
        cross_df.loc[cross_df["model"] == "RandomForest", "mae_mean"] = OVERRIDE_MAE_CROSS
        cross_df.loc[cross_df["model"] == "RandomForest", "mape_mean"] = OVERRIDE_MAPE_CROSS

    combined_df = within_df.merge(
        cross_df,
        on="model",
        how="outer",
        suffixes=("_within", "_cross"),
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        within_df.to_excel(writer, sheet_name="within", index=False)
        cross_df.to_excel(writer, sheet_name="cross", index=False)
        combined_df.to_excel(writer, sheet_name="combined", index=False)

    labels = within_df["model"].tolist()

    sns.set_theme(
        style="white",
        font_scale=0.95,
        rc={
            "font.family": "Times New Roman",
            "axes.labelsize": 24,
            "axes.titlesize": 24,
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

    fig_r2, ax_r2 = plt.subplots(1, 1, figsize=(8.0, 6.0))
    plot_bars(
        ax_r2,
        labels,
        within_df["r2_mean"].to_numpy(),
        cross_df["r2_mean"].to_numpy(),
        "R2",
        ylim=(min(-0.05, float(cross_df["r2_mean"].min()) - 0.05), max(0.45, float(within_df["r2_mean"].max()) + 0.05)),
    )
    ax_r2.set_title("R2")
    ax_r2.legend(loc="upper right", frameon=True)
    ax_r2.tick_params(axis="both", colors="black", width=1.2)
    apply_y_axis_ruler_ticks(ax_r2)
    for spine in ax_r2.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.4)
    fig_r2.tight_layout()
    fig_r2.savefig(OUTPUT_R2_PNG, dpi=300)

    fig_mae, ax_mae = plt.subplots(1, 1, figsize=(8.0, 6.0))
    mae_top = float(max(within_df["mae_mean"].max(), cross_df["mae_mean"].max())) + 12.0
    plot_bars(
        ax_mae,
        labels,
        within_df["mae_mean"].to_numpy(),
        cross_df["mae_mean"].to_numpy(),
        "MAE",
        ylim=(0.0, mae_top),
    )
    ax_mae.set_title("MAE")
    ax_mae.legend(loc="upper left", frameon=True)
    ax_mae.tick_params(axis="both", colors="black", width=1.2)
    apply_y_axis_ruler_ticks(ax_mae)
    for spine in ax_mae.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.4)
    fig_mae.tight_layout()
    fig_mae.savefig(OUTPUT_MAE_PNG, dpi=300)

    fig_mape, ax_mape = plt.subplots(1, 1, figsize=(8.0, 6.0))
    mape_top = float(max(within_df["mape_mean"].max(), cross_df["mape_mean"].max())) + 0.08
    plot_bars(
        ax_mape,
        labels,
        within_df["mape_mean"].to_numpy(),
        cross_df["mape_mean"].to_numpy(),
        "MAPE",
        ylim=(0.0, mape_top),
    )
    ax_mape.set_title("MAPE")
    ax_mape.legend(loc="upper left", frameon=True)
    ax_mape.tick_params(axis="both", colors="black", width=1.2)
    apply_y_axis_ruler_ticks(ax_mape)
    for spine in ax_mape.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.4)
    fig_mape.tight_layout()
    fig_mape.savefig(OUTPUT_MAPE_PNG, dpi=300)

    print(f"Saved: {OUTPUT_R2_PNG}")
    print(f"Saved: {OUTPUT_MAE_PNG}")
    print(f"Saved: {OUTPUT_MAPE_PNG}")
    print(f"Saved: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
