"""模型脚本。

文件：plot_feature_count_trend.py
统一使用项目内的相对路径。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "model", "binary-class", "results")
WITHIN_CSV = os.path.join(RESULTS_DIR, "feature_count_margin15_within.csv")
CROSS_CSV = os.path.join(RESULTS_DIR, "feature_count_margin15_cross.csv")
WITHIN_ACC_CSV = os.path.join(RESULTS_DIR, "feature_count_accuracy_within.csv")
CROSS_ACC_CSV = os.path.join(RESULTS_DIR, "feature_count_accuracy_cross.csv")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "feature_count_trend.png")

def load_data():
    within_df = pd.read_csv(WITHIN_CSV)
    cross_df = pd.read_csv(CROSS_CSV)
    within_acc_df = pd.read_csv(WITHIN_ACC_CSV)
    cross_acc_df = pd.read_csv(CROSS_ACC_CSV)
    return within_df, cross_df, within_acc_df, cross_acc_df

def main():
    within_df, cross_df, within_acc_df, cross_acc_df = load_data()

    sns.set_theme(
        style="whitegrid",
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
    fig, ax = plt.subplots(1, 1, figsize=(12.0, 7.0))

    def acc_pct_at_k(df, k):
        hit = df[df["k"] == k]
        if hit.empty:
            idx = int(np.argmin(np.abs(df["k"].to_numpy() - k)))
            return float(df.iloc[idx]["acc"] * 100.0)
        return float(hit.iloc[-1]["acc"] * 100.0)

    ax.plot(
        within_df["k"],
        within_df["acc"] * 100.0,
        label="Non-Cross Subject (Margin=15)",
        color="#f2b179",
        linewidth=2.2,
    )
    ax.plot(
        cross_df["k"],
        cross_df["acc"] * 100.0,
        label="Cross Subject (Margin=15)",
        color="#7ecb8b",
        linewidth=2.2,
    )
    ax.plot(
        within_acc_df["k"],
        within_acc_df["acc"] * 100.0,
        label="Non-Cross Subject (Margin=0)",
        color="#f2b179",
        linewidth=1.8,
        linestyle="--",
    )
    ax.plot(
        cross_acc_df["k"],
        cross_acc_df["acc"] * 100.0,
        label="Cross Subject (Margin=0)",
        color="#7ecb8b",
        linewidth=1.8,
        linestyle="--",
    )

    ax.axvline(12, color="#f2b179", linestyle="--", linewidth=1)
    ax.axvline(11, color="#7ecb8b", linestyle="--", linewidth=1)

    all_pct = np.concatenate([
        within_df["acc"].to_numpy() * 100.0,
        cross_df["acc"].to_numpy() * 100.0,
        within_acc_df["acc"].to_numpy() * 100.0,
        cross_acc_df["acc"].to_numpy() * 100.0,
    ])
    y_min = 40
    y_max = 100

    k_non_cross = 12
    k_cross = 11
    y_non_cross = acc_pct_at_k(within_df, k_non_cross)
    y_cross = acc_pct_at_k(cross_df, k_cross)
    ax.scatter([k_non_cross], [y_non_cross], color="#f2b179", s=28, zorder=5)
    ax.scatter([k_cross], [y_cross], color="#7ecb8b", s=28, zorder=5)
    ax.annotate(f"{y_non_cross:.2f}", (k_non_cross, y_non_cross), textcoords="offset points", xytext=(4, -1), fontsize=12)
    ax.annotate(f"{y_cross:.2f}", (k_cross, y_cross), textcoords="offset points", xytext=(-30, 0), fontsize=12)

    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 5, 10, 11, 12, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.legend(loc="upper right", frameon=True)
    ax.tick_params(axis="both", colors="black", width=1.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
