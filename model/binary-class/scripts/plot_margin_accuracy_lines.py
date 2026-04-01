"""模型脚本。

文件：plot_margin_accuracy_lines.py
统一使用项目内的相对路径。
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WITHIN_FILE = os.path.join(SCRIPT_DIR, "分类非跨试验结果")
CROSS_FILE = os.path.join(SCRIPT_DIR, "分类跨试验结果.xlsx")

OUTPUT_PNG = os.path.join(SCRIPT_DIR, "margin_accuracy_within_vs_cross.png")

PALETTE = {"Non-Cross Subject": "#f2b179", "Cross Subject": "#7ecb8b"}

def load_within_text(path):
    rows = []
    pattern = re.compile(r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            margin = int(m.group(1))
            acc = float(m.group(2))
            rows.append({"Margin": margin, "Accuracy": acc, "Scenario": "Non-Cross Subject"})
    return pd.DataFrame(rows)

def load_cross_excel(path):
    df = pd.read_excel(path, sheet_name="Summary")
    df = df.rename(columns={"Margin (mA)": "Margin", "Accuracy": "Accuracy"})
    df = df[["Margin", "Accuracy"]].copy()
    df["Scenario"] = "Cross Subject"
    return df

def main():
    within_df = load_within_text(WITHIN_FILE)
    cross_df = load_cross_excel(CROSS_FILE)

    plot_df = pd.concat([within_df, cross_df], ignore_index=True)
    plot_df["Accuracy"] = plot_df["Accuracy"] * 100

    sns.set_theme(
        style="whitegrid",
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

    sns.lineplot(
        data=plot_df,
        x="Margin",
        y="Accuracy",
        hue="Scenario",
        palette=PALETTE,
        marker="o",
        ax=ax,
    )

    ax.set_xlabel("Margin")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(75, 86)
    ax.set_xticks([0, 5, 10, 15])
    ax.legend(loc="upper left", frameon=True)
    ax.tick_params(axis="both", colors="black", width=1.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.4)

    for _, row in plot_df.iterrows():
        ax.annotate(
            f"{row['Accuracy']:.2f}",
            (row["Margin"], row["Accuracy"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
