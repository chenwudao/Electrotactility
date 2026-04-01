"""模型脚本。

文件：plot_model_comparison_bar.py
统一使用项目内的相对路径。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "model", "binary-class", "results")
WITHIN_FILE = os.path.join(RESULTS_DIR, "model_comparison_within_subject_results.xlsx")
CROSS_FILE = os.path.join(RESULTS_DIR, "model_comparison_cross_subject_results.xlsx")

OUTPUT_PNG = os.path.join(RESULTS_DIR, "model_comparison_within_vs_cross.png")

MODEL_ORDER = [
    "RandomForest",
    "SVM_RBF",
    "GradientBoosting",
    "CatBoost",
    "Baseline",
]
DISPLAY_MODEL = {
    "SVM_RBF": "SVM",
}

BASELINE_ACC = 0.50

def apply_y_axis_ruler_ticks(ax):
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", direction="in", length=4, width=1.8, left=True, right=False)
    ax.tick_params(axis="y", which="minor", direction="in", length=2, width=1.6, left=True, right=False)

def load_results(path):
    df = pd.read_excel(path)
    df["model"] = df["model"].astype(str)
    if "acc_margin_mean" in df.columns:
        metric_col = "acc_margin_mean"
    else:
        metric_col = "acc_mean"
    return df.set_index("model"), metric_col

def main():
    within_df, within_metric = load_results(WITHIN_FILE)
    cross_df, cross_metric = load_results(CROSS_FILE)

    within_vals = []
    cross_vals = []
    for name in MODEL_ORDER:
        if name == "Baseline":
            within_vals.append(BASELINE_ACC)
            cross_vals.append(BASELINE_ACC)
            continue
        within_vals.append(float(within_df.loc[name, within_metric]))
        cross_vals.append(float(cross_df.loc[name, cross_metric]))

    within_vals = np.array(within_vals) * 100
    cross_vals = np.array(cross_vals) * 100

    plot_df = pd.DataFrame(
        {
            "Model": [DISPLAY_MODEL.get(m, m) for m in MODEL_ORDER] * 2,
            "Scenario": ["Non-Cross Subject"] * len(MODEL_ORDER) + ["Cross Subject"] * len(MODEL_ORDER),
            "Accuracy": np.concatenate([within_vals, cross_vals]),
        }
    )

    sns.set_theme(
        style="white",
        font_scale=1.0,
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
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    palette = {"Non-Cross Subject": "#f7c58a", "Cross Subject": "#9ad49b"}

    sns.barplot(
        data=plot_df,
        x="Model",
        y="Accuracy",
        hue="Scenario",
        palette=palette,
        ax=ax,
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(False)
    ax.tick_params(axis="both", colors="black", width=1.2)
    apply_y_axis_ruler_ticks(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.4)

    for p in ax.patches:
        height = p.get_height()
        if height < 1:
            continue
        ax.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
