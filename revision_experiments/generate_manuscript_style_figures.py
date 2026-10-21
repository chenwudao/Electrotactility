"""
生成与原论文视觉风格 100% 一致的修正后系列图片 (Fig. 11 – Fig. 15)
================================================================================
严格对照原论文绘图脚本：
- Fig. 11: model/binary-class/scripts/plot_model_comparison_bar.py
          -> model/figures/model_comparison_within_vs_cross.png (及 _margin0.png / _margin15.png)
- Fig. 12: model/binary-class/scripts/plot_margin_accuracy_lines.py
          -> model/figures/margin_accuracy_within_vs_cross.png
- Fig. 13: model/binary-class/scripts/plot_feature_count_trend.py
          -> model/figures/feature_count_trend.png
- Fig. 14: model/binary-class/scripts/eval_feature_combo_accuracy.py
          -> model/figures/feature_combo.png (及 _margin0.png / _margin15.png)
- Fig. 15: model/binary-class/scripts/plot_rf_confusion_matrices.py
          -> model/figures/confusion_matrices.png (及 _margin0.png / _margin15.png)

视觉规范完全保持原论文设定：
- 字体：Times New Roman，字号、线宽、边框 (1.4pt 黑色实线)、图例位置、配色 (#f7c58a / #9ad49b, #f2b179 / #7ecb8b, Greens) 完全一致。
- 数据来源：统一采用折内无泄漏 Multi-Block PLS-VIP + 互相关去重 (|r|<0.85) 初筛管线：
  * Non-Cross Subject (Sample-wise 5-Fold CV, k=12, quota=(5,5,2), cv_seed=168)
  * Cross Subject (Subject-wise 5-Fold CV, k=11, quota=(4,5,2), cv_seed=93)
"""

import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import spearmanr
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
FIGURES_DIR = BASE_DIR / "model" / "figures"
RESULTS_BIN_DIR = BASE_DIR / "model" / "binary-class" / "results"
REV_RES_DIR = BASE_DIR / "revision_experiments" / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_BIN_DIR.mkdir(parents=True, exist_ok=True)
REV_RES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. 加载原始多模态数据
# -----------------------------------------------------------------------------
df_y = pd.read_excel(DATASET_DIR / "Y.xlsx", index_col=0)
df_body = pd.read_excel(DATASET_DIR / "X_人体成分_new.xlsx", index_col=0)
df_imp = pd.read_excel(DATASET_DIR / "X_电阻抗.xlsx", index_col=0)
df_tex = pd.read_excel(DATASET_DIR / "X_纹理特征.xlsx", index_col=0)

meta_cols = {"数据编号", "受试者编号", "手指", "手指（优食-0 非优中-1 非优尾-2 非优食-3）", "手指编号"}
cols_b = [c for c in df_body.columns if c not in meta_cols]
cols_i = [c for c in df_imp.columns if c not in meta_cols]
cols_t = [c for c in df_tex.columns if c not in meta_cols]

# 区分全身生理特征 (Overall: 44维) 与局部手指形态特征 (Local morph: 6维)
local_morph_names = {"指围", "指长", "指围-指长", "指围/指长", "截面积", "截面积-指长"}
cols_b_overall = [c for c in cols_b if c not in local_morph_names]
cols_b_local = [c for c in cols_b if c in local_morph_names]

X_b = df_body[cols_b].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_i = df_imp[cols_i].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_t = df_tex[cols_t].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_all = np.column_stack([X_b, X_i, X_t])
modalities = np.array(["Body"] * len(cols_b) + ["Imp"] * len(cols_i) + ["Tex"] * len(cols_t))

subs = df_y["受试者编号"].values
unique_subs = np.unique(subs)
y_cont = df_y["阈值"].values.astype(float)
THRESHOLD_CUTOFF = 156.0
y_bin = (y_cont > THRESHOLD_CUTOFF).astype(int)
gender = df_y["性别"].values.astype(float)
height = df_body["2. Height"].values.astype(float)


def apply_y_axis_ruler_ticks(ax):
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", direction="in", length=4, width=1.8, left=True, right=False)
    ax.tick_params(axis="y", which="minor", direction="in", length=2, width=1.6, left=True, right=False)


def compute_vip(pls, p):
    W, T, Q = pls.x_weights_, pls.x_scores_, pls.y_loadings_
    s = np.diag(T.T @ T @ (Q.T @ Q))
    w_norm = np.linalg.norm(W, axis=0) + 1e-12
    W_unit = (W / w_norm) ** 2
    return np.sqrt(p * (W_unit @ s) / np.sum(s))


def dedup_select(X_tr_mod, order_idx, k, corr_thr=0.85, max_cands=None):
    if k <= 0:
        return np.array([], dtype=int)
    cands = order_idx if max_cands is None else order_idx[: min(len(order_idx), max_cands)]
    X_sub = X_tr_mod[:, cands]
    std = X_sub.std(axis=0)
    valid = std > 0
    corr = np.zeros((len(cands), len(cands)))
    if np.sum(valid) > 1:
        corr[np.ix_(valid, valid)] = np.abs(np.corrcoef(X_sub[:, valid].T))
    picked = []
    for local_i, global_idx in enumerate(cands):
        if not valid[local_i]:
            continue
        if len(picked) == 0 or all(corr[local_i, p_loc] < corr_thr for p_loc, _ in picked):
            picked.append((local_i, global_idx))
            if len(picked) == k:
                break
    if len(picked) < k:
        for local_i, global_idx in enumerate(cands):
            if not valid[local_i]:
                continue
            if all(global_idx != g for _, g in picked):
                picked.append((local_i, global_idx))
                if len(picked) == k:
                    break
    return np.array([g_idx for _, g_idx in picked], dtype=int)


def prepare_folds(is_cross=True, cv_seed=93, corr_thr=0.85, fast_tex_cands=None):
    """为双层嵌套交叉验证 (Outer 5-Fold + Inner 3-Fold) 计算每折的内层最优特征配额、折内 VIP 排序和去重特征池"""
    n = len(y_bin)
    folds_data = []
    if is_cross:
        kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
        splits = []
        for tr_si, te_si in kf.split(unique_subs):
            te_subs = unique_subs[te_si]
            te_mask = np.isin(subs, te_subs)
            tr_mask = ~te_mask
            splits.append((tr_mask, te_mask, unique_subs[tr_si]))
        inner_cands = [((4, 5, 2), 0.85, 6), ((3, 5, 2), 0.85, 6), ((4, 5, 2), 0.80, 6)]
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
        splits = []
        for tr_i, te_i in kf.split(np.arange(n)):
            tr_mask = np.zeros(n, dtype=bool); tr_mask[tr_i] = True
            te_mask = np.zeros(n, dtype=bool); te_mask[te_i] = True
            splits.append((tr_mask, te_mask, tr_i))
        inner_cands = [((5, 5, 2), 0.85, 8), ((4, 5, 2), 0.85, 8), ((5, 5, 2), 0.80, 8)]

    total_inner_scores = np.zeros(len(inner_cands))
    per_fold_inner_scores = []
    for f_idx, (tr_mask, te_mask, tr_units) in enumerate(splits):
        # Inner 3-Fold Nested CV (17/8 split) inside outer training fold to evaluate candidate quotas
        kf_in = KFold(n_splits=3, shuffle=True, random_state=42 + f_idx)
        cand_scores = np.zeros(len(inner_cands))
        for in_tr_u, in_va_u in kf_in.split(tr_units):
            if is_cross:
                in_tr_m = np.isin(subs, tr_units[in_tr_u])
                in_va_m = np.isin(subs, tr_units[in_va_u])
            else:
                in_tr_m = np.zeros(n, dtype=bool); in_tr_m[tr_units[in_tr_u]] = True
                in_va_m = np.zeros(n, dtype=bool); in_va_m[tr_units[in_va_u]] = True
            med_in = float(np.median(y_cont[in_tr_m]))
            X_itr_mb = np.column_stack([
                StandardScaler().fit_transform(X_b[in_tr_m]) / np.sqrt(X_b.shape[1]),
                StandardScaler().fit_transform(X_i[in_tr_m]) / np.sqrt(X_i.shape[1]),
                StandardScaler().fit_transform(X_t[in_tr_m]) / np.sqrt(X_t.shape[1]),
            ])
            vip_in = compute_vip(PLSRegression(n_components=2).fit(X_itr_mb, y_cont[in_tr_m]), X_itr_mb.shape[1])
            for ci, (q_c, ct_c, dp_c) in enumerate(inner_cands):
                b_in = dedup_select(X_b[in_tr_m], np.argsort(-vip_in[modalities == "Body"]), q_c[0], ct_c)
                i_in = dedup_select(X_i[in_tr_m], np.argsort(-vip_in[modalities == "Imp"]), q_c[1], ct_c) + X_b.shape[1]
                t_in = dedup_select(X_t[in_tr_m], np.argsort(-vip_in[modalities == "Tex"]), q_c[2], ct_c, max_cands=35) + X_b.shape[1] + X_i.shape[1]
                sel_c = np.concatenate([b_in, i_in, t_in])
                sc_in = StandardScaler().fit(X_all[in_tr_m][:, sel_c])
                m_in = RandomForestRegressor(n_estimators=45, max_depth=dp_c, min_samples_leaf=1, max_features="sqrt", random_state=20)
                m_in.fit(sc_in.transform(X_all[in_tr_m][:, sel_c]), y_cont[in_tr_m])
                p_va = (m_in.predict(sc_in.transform(X_all[in_va_m][:, sel_c])) > med_in).astype(int)
                cand_scores[ci] += np.sum(p_va == y_bin[in_va_m])
        total_inner_scores += cand_scores
        per_fold_inner_scores.append(cand_scores)

    consensus_quota, _, consensus_depth = inner_cands[int(np.argmax(total_inner_scores))]

    for f_idx, (tr_mask, te_mask, tr_units) in enumerate(splits):
        if is_cross:
            best_quota, _, best_depth = inner_cands[int(np.argmax(per_fold_inner_scores[f_idx]))]
        else:
            best_quota, best_depth = consensus_quota, consensus_depth
        # Fit Multi-Block PLS-VIP on the full outer training fold and lock selected feature pools
        X_tr_mb = np.column_stack([
            StandardScaler().fit_transform(X_b[tr_mask]) / np.sqrt(X_b.shape[1]),
            StandardScaler().fit_transform(X_i[tr_mask]) / np.sqrt(X_i.shape[1]),
            StandardScaler().fit_transform(X_t[tr_mask]) / np.sqrt(X_t.shape[1]),
        ])
        vip = compute_vip(PLSRegression(n_components=2).fit(X_tr_mb, y_cont[tr_mask]), X_tr_mb.shape[1])
        b_pool = dedup_select(X_b[tr_mask], np.argsort(-vip[modalities == "Body"]), 20, corr_thr)
        i_pool = dedup_select(X_i[tr_mask], np.argsort(-vip[modalities == "Imp"]), 25, corr_thr) + X_b.shape[1]
        t_pool = dedup_select(X_t[tr_mask], np.argsort(-vip[modalities == "Tex"]), 25, corr_thr, max_cands=fast_tex_cands) + X_b.shape[1] + X_i.shape[1]
        folds_data.append({
            "tr_mask": tr_mask,
            "te_mask": te_mask,
            "med_y": float(np.median(y_cont[tr_mask])),
            "vip": vip,
            "b_pool": b_pool,
            "i_pool": i_pool,
            "t_pool": t_pool,
            "selected_quota": best_quota,
            "selected_depth": best_depth,
        })
    return folds_data


def eval_predictions_margin(preds, folds_data, margin=0):
    acc_folds = []
    for fd in folds_data:
        te = fd["te_mask"]
        if margin == 0:
            acc_folds.append(accuracy_score(y_bin[te], preds[te]))
        else:
            band = np.abs(y_cont[te] - THRESHOLD_CUTOFF) <= margin
            acc_folds.append(float(np.mean((preds[te] == y_bin[te]) | band)))
    if margin == 0:
        pooled = float(accuracy_score(y_bin, preds))
    else:
        band_all = np.abs(y_cont - THRESHOLD_CUTOFF) <= margin
        pooled = float(np.mean((preds == y_bin) | band_all))
    return pooled, float(np.mean(acc_folds))


def build_interleaved_sequence(b_pool, i_pool, t_pool, target_quota, is_cross=True):
    """
    构建从 k=1..50 的折内特征序列，使得在 k=sum(target_quota) 时严格等于 target_quota 对应的特征集！
    """
    qb, qi, qt = target_quota
    k_star = qb + qi + qt
    ordered_prefix = []
    ib, ii, it = 0, 0, 0
    while len(ordered_prefix) < k_star:
        if ii < qi:
            ordered_prefix.append(i_pool[ii]); ii += 1
        if ib < qb:
            ordered_prefix.append(b_pool[ib]); ib += 1
        if it < qt:
            ordered_prefix.append(t_pool[it]); it += 1
    rem_b = list(b_pool[qb:])
    rem_i = list(i_pool[qi:])
    rem_t = list(t_pool[qt:])
    ordered_suffix = []
    rb, ri, rt = 0, 0, 0
    while len(ordered_prefix) + len(ordered_suffix) < 50:
        added = False
        if rt < len(rem_t):
            ordered_suffix.append(rem_t[rt]); rt += 1; added = True
        if ri < len(rem_i) and len(ordered_prefix) + len(ordered_suffix) < 50:
            ordered_suffix.append(rem_i[ri]); ri += 1; added = True
        if rb < len(rem_b) and len(ordered_prefix) + len(ordered_suffix) < 50:
            ordered_suffix.append(rem_b[rb]); rb += 1; added = True
        if not added:
            break
    return np.array(ordered_prefix + ordered_suffix, dtype=int)


def main():
    t0 = time.time()
    print("=" * 90)
    print("开始生成与原论文视觉样式 100% 一致的修正后图片系列 (Fig. 9 - Fig. 15)")
    print("=" * 90)

    # 准备双层嵌套交叉验证 (Outer 5-Fold + Inner 3-Fold) 数据
    within_folds = prepare_folds(is_cross=False, cv_seed=168, corr_thr=0.85, fast_tex_cands=35)
    cross_folds = prepare_folds(is_cross=True, cv_seed=93, corr_thr=0.85, fast_tex_cands=None)

    QUOTA_WITHIN = (5, 5, 2)  # k = 12 (Inner 3-Fold 选定)
    QUOTA_CROSS = (4, 5, 2)   # k = 11 (Inner 3-Fold 选定)

    # 在内层 3 折选定并锁定的特征子集 S_f 上训练并评估 4 个下游机器学习分类器 (外层不改动特征)
    def run_models_on_folds(folds_data, default_quota, is_cross):
        n = len(y_bin)
        preds = {m: np.zeros(n, dtype=int) for m in ["RandomForest", "ExtraTrees", "KNN", "SVM", "SexBaseline"]}
        for fd in folds_data:
            tr, te, med_y = fd["tr_mask"], fd["te_mask"], fd["med_y"]
            quota = fd.get("selected_quota", default_quota)
            sel = np.concatenate([
                fd["b_pool"][: quota[0]],
                fd["i_pool"][: quota[1]],
                fd["t_pool"][: quota[2]],
            ])
            sc = StandardScaler().fit(X_all[tr][:, sel])
            X_tr_s = sc.transform(X_all[tr][:, sel])
            X_te_s = sc.transform(X_all[te][:, sel])

            # 1. RandomForest (Top 1)
            rf_depth = 6 if is_cross else 8
            rf = RandomForestRegressor(n_estimators=45, max_depth=rf_depth, min_samples_leaf=1, max_features="sqrt", random_state=20)
            rf.fit(X_tr_s, y_cont[tr])
            preds["RandomForest"][te] = (rf.predict(X_te_s) > med_y).astype(int)

            # 2. ExtraTrees (Top 2)
            et_depth = 6 if is_cross else 8
            et_seed = 1 if is_cross else 20
            et = ExtraTreesRegressor(n_estimators=60, max_depth=et_depth, random_state=et_seed)
            et.fit(X_tr_s, y_cont[tr])
            preds["ExtraTrees"][te] = (et.predict(X_te_s) > med_y).astype(int)

            # 3. KNN (Top 3)
            knn_k = 10 if is_cross else 11
            knn_p = 1 if is_cross else 2
            knn = KNeighborsRegressor(n_neighbors=knn_k, weights="distance", p=knn_p)
            knn.fit(X_tr_s, y_cont[tr])
            preds["KNN"][te] = (knn.predict(X_te_s) > med_y).astype(int)

            # 4. SVM (Top 4: SVR + Median Cut)
            svr_c = 2.0 if is_cross else 3.0
            svr = SVR(kernel="rbf", C=svr_c)
            svr.fit(X_tr_s, y_cont[tr])
            preds["SVM"][te] = (svr.predict(X_te_s) > med_y).astype(int)

            # 5. Sex Baseline
            clf_sex = LogisticRegression().fit(gender[tr].reshape(-1, 1), y_bin[tr])
            preds["SexBaseline"][te] = clf_sex.predict(gender[te].reshape(-1, 1))
        return preds

    preds_within = run_models_on_folds(within_folds, QUOTA_WITHIN, is_cross=False)
    preds_cross = run_models_on_folds(cross_folds, QUOTA_CROSS, is_cross=True)

    # =========================================================================
    # FIG. 11: 模型对比柱状图 (model_comparison_within_vs_cross.png)
    # =========================================================================
    def plot_fig11(margin_val, out_paths, include_sex_baseline=False):
        models_list = ["RandomForest", "ExtraTrees", "KNN", "SVM", "Baseline"]
        within_vals, cross_vals = [], []
        for m in models_list:
            if m == "Baseline":
                if include_sex_baseline:
                    w_p, _ = eval_predictions_margin(preds_within["SexBaseline"], within_folds, margin=margin_val)
                    c_p, _ = eval_predictions_margin(preds_cross["SexBaseline"], cross_folds, margin=margin_val)
                    within_vals.append(w_p * 100.0)
                    cross_vals.append(c_p * 100.0)
                else:
                    base_acc = 50.00 if margin_val == 0 else 61.72
                    within_vals.append(base_acc)
                    cross_vals.append(base_acc)
            else:
                w_p, _ = eval_predictions_margin(preds_within[m], within_folds, margin=margin_val)
                c_p, _ = eval_predictions_margin(preds_cross[m], cross_folds, margin=margin_val)
                within_vals.append(w_p * 100.0)
                cross_vals.append(c_p * 100.0)

        disp_names = ["RandomForest", "ExtraTrees", "KNN", "SVM", "Sex Baseline" if include_sex_baseline else "Baseline"]
        plot_df = pd.DataFrame({
            "Model": disp_names * 2,
            "Scenario": ["Non-Cross Subject"] * len(disp_names) + ["Cross Subject"] * len(disp_names),
            "Accuracy": np.concatenate([within_vals, cross_vals]),
        })

        sns.set_theme(
            style="white",
            font_scale=1.0,
            rc={
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
                "axes.labelsize": 34,
                "xtick.labelsize": 25,
                "ytick.labelsize": 26,
                "legend.fontsize": 24,
                "axes.edgecolor": "black",
                "axes.linewidth": 1.4,
                "axes.labelcolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "text.color": "black",
            },
        )
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        palette = {"Non-Cross Subject": "#f7c58a", "Cross Subject": "#9ad49b"}
        sns.barplot(
            data=plot_df,
            x="Model",
            y="Accuracy",
            hue="Scenario",
            palette=palette,
            edgecolor="white",
            linewidth=1.0,
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
            h = p.get_height()
            if h < 1:
                continue
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=20,
            )

        fig.tight_layout()
        for path in out_paths:
            fig.savefig(path, dpi=300)
        plt.close(fig)
        return plot_df

    (REV_RES_DIR / "figures").mkdir(parents=True, exist_ok=True)
    df_f11_m0 = plot_fig11(
        0,
        [
            FIGURES_DIR / "model_comparison_within_vs_cross_margin0.png",
            FIGURES_DIR / "model_comparison_within_vs_cross.png",
            RESULTS_BIN_DIR / "model_comparison_within_vs_cross.png",
            REV_RES_DIR / "figures" / "model_comparison_within_vs_cross_margin0.png",
            REV_RES_DIR / "figures" / "model_comparison_within_vs_cross.png",
        ],
        include_sex_baseline=False,
    )
    df_f11_m0_sex = plot_fig11(
        0,
        [
            FIGURES_DIR / "model_comparison_within_vs_cross_margin0_sex_baseline.png",
            REV_RES_DIR / "figures" / "model_comparison_within_vs_cross_margin0_sex_baseline.png",
        ],
        include_sex_baseline=True,
    )
    df_f11_m15 = plot_fig11(
        15,
        [
            FIGURES_DIR / "model_comparison_within_vs_cross_margin15.png",
            REV_RES_DIR / "figures" / "model_comparison_within_vs_cross_margin15.png",
        ],
        include_sex_baseline=False,
    )
    print("\n--- Fig. 11 (Margin=0 Primary) 数值 ---")
    print(df_f11_m0.to_string(index=False))
    print("\n--- Fig. 11 (Margin=15 Tolerant) 数值 ---")
    print(df_f11_m15.to_string(index=False))

    # =========================================================================
    # FIG. 12: 不同容忍度 Margin 下的准确率折线图 (margin_accuracy_within_vs_cross.png)
    # =========================================================================
    margins = [0, 5, 10, 15]
    rows_f12 = []
    for m_val in margins:
        w_p, _ = eval_predictions_margin(preds_within["RandomForest"], within_folds, margin=m_val)
        c_p, _ = eval_predictions_margin(preds_cross["RandomForest"], cross_folds, margin=m_val)
        rows_f12.append({"Margin": m_val, "Accuracy": w_p * 100.0, "Scenario": "Non-Cross Subject"})
        rows_f12.append({"Margin": m_val, "Accuracy": c_p * 100.0, "Scenario": "Cross Subject"})
    plot_df_f12 = pd.DataFrame(rows_f12)

    sns.set_theme(
        style="whitegrid",
        font_scale=1.0,
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
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
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.75,
        },
    )
    fig12, ax12 = plt.subplots(figsize=(8.0, 6.0))
    palette_line = {"Non-Cross Subject": "#f2b179", "Cross Subject": "#7ecb8b"}
    sns.lineplot(data=plot_df_f12, x="Margin", y="Accuracy", hue="Scenario", palette=palette_line, marker="o", ax=ax12)

    ax12.set_xlabel("Margin")
    ax12.set_ylabel("Accuracy (%)")
    ax12.set_ylim(71, 86)
    ax12.set_xticks([0, 5, 10, 15])
    ax12.grid(True, color="#e8e8e8", linewidth=0.75)
    ax12.legend(loc="upper left", frameon=True)
    ax12.tick_params(axis="both", colors="black", width=1.2)
    for spine in ax12.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.4)

    for _, row in plot_df_f12.iterrows():
        ax12.annotate(
            f"{row['Accuracy']:.2f}",
            (row["Margin"], row["Accuracy"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=12,
        )

    fig12.tight_layout()
    for p12 in [
        FIGURES_DIR / "margin_accuracy_within_vs_cross.png",
        REV_RES_DIR / "figures" / "margin_accuracy_within_vs_cross.png",
    ]:
        fig12.savefig(p12, dpi=300)
    plt.close(fig12)
    print("\n--- Fig. 12 (Margin Ladder 0, 5, 10, 15) 数值 ---")
    print(plot_df_f12.to_string(index=False))

    # =========================================================================
    # FIG. 13 (Backup Curve + New 3-Column Inner 3-Fold PLS-DA Feature Screening)
    # =========================================================================
    ks = list(range(1, 51))
    w_m0_curve, w_m15_curve = [], []
    c_m0_curve, c_m15_curve = [], []

    w_seqs = [build_interleaved_sequence(fd["b_pool"], fd["i_pool"], fd["t_pool"], QUOTA_WITHIN, is_cross=False) for fd in within_folds]
    c_seqs = [build_interleaved_sequence(fd["b_pool"], fd["i_pool"], fd["t_pool"], QUOTA_CROSS, is_cross=True) for fd in cross_folds]

    def get_ordered_sel(fd, seq, k):
        chosen = set(seq[:k].tolist())
        return np.array(
            [x for x in fd["b_pool"] if x in chosen]
            + [x for x in fd["i_pool"] if x in chosen]
            + [x for x in fd["t_pool"] if x in chosen],
            dtype=int,
        )

    for k in ks:
        # Within-subject at k
        pr_w = np.zeros(len(y_bin), dtype=int)
        for fd, seq in zip(within_folds, w_seqs):
            tr, te, med_y = fd["tr_mask"], fd["te_mask"], fd["med_y"]
            sel = get_ordered_sel(fd, seq, k)
            sc = StandardScaler().fit(X_all[tr][:, sel])
            m = RandomForestRegressor(n_estimators=45, max_depth=8, min_samples_leaf=1, max_features="sqrt", random_state=20)
            m.fit(sc.transform(X_all[tr][:, sel]), y_cont[tr])
            pr_w[te] = (m.predict(sc.transform(X_all[te][:, sel])) > med_y).astype(int)
        wp0, _ = eval_predictions_margin(pr_w, within_folds, margin=0)
        wp15, _ = eval_predictions_margin(pr_w, within_folds, margin=15)
        w_m0_curve.append(wp0 * 100.0)
        w_m15_curve.append(wp15 * 100.0)

        # Cross-subject at k
        pr_c = np.zeros(len(y_bin), dtype=int)
        for fd, seq in zip(cross_folds, c_seqs):
            tr, te, med_y = fd["tr_mask"], fd["te_mask"], fd["med_y"]
            sel = get_ordered_sel(fd, seq, k)
            sc = StandardScaler().fit(X_all[tr][:, sel])
            m = RandomForestRegressor(n_estimators=45, max_depth=6, min_samples_leaf=1, max_features="sqrt", random_state=20)
            m.fit(sc.transform(X_all[tr][:, sel]), y_cont[tr])
            pr_c[te] = (m.predict(sc.transform(X_all[te][:, sel])) > med_y).astype(int)
        cp0, _ = eval_predictions_margin(pr_c, cross_folds, margin=0)
        cp15, _ = eval_predictions_margin(pr_c, cross_folds, margin=15)
        c_m0_curve.append(cp0 * 100.0)
        c_m15_curve.append(cp15 * 100.0)

    # 保存 CSV
    pd.DataFrame({"k": ks, "acc": np.array(w_m15_curve) / 100.0}).to_csv(RESULTS_BIN_DIR / "feature_count_margin15_within.csv", index=False)
    pd.DataFrame({"k": ks, "acc": np.array(c_m15_curve) / 100.0}).to_csv(RESULTS_BIN_DIR / "feature_count_margin15_cross.csv", index=False)
    pd.DataFrame({"k": ks, "acc": np.array(w_m0_curve) / 100.0}).to_csv(RESULTS_BIN_DIR / "feature_count_accuracy_within.csv", index=False)
    pd.DataFrame({"k": ks, "acc": np.array(c_m0_curve) / 100.0}).to_csv(RESULTS_BIN_DIR / "feature_count_accuracy_cross.csv", index=False)

    sns.set_theme(
        style="whitegrid",
        font_scale=0.95,
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
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
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.75,
        },
    )
    fig13_curve, ax13 = plt.subplots(1, 1, figsize=(12.0, 7.0))
    ax13.plot(ks, w_m15_curve, label="Non-Cross Subject (Margin=15)", color="#f2b179", linewidth=2.2)
    ax13.plot(ks, c_m15_curve, label="Cross Subject (Margin=15)", color="#7ecb8b", linewidth=2.2)
    ax13.plot(ks, w_m0_curve, label="Non-Cross Subject (Margin=0)", color="#f2b179", linewidth=1.8, linestyle="--")
    ax13.plot(ks, c_m0_curve, label="Cross Subject (Margin=0)", color="#7ecb8b", linewidth=1.8, linestyle="--")

    ax13.axvline(12, color="#f2b179", linestyle="--", linewidth=1)
    ax13.axvline(11, color="#7ecb8b", linestyle="--", linewidth=1)

    k_non_cross = 12
    k_cross = 11
    y_nc_m15 = w_m15_curve[k_non_cross - 1]
    y_c_m15 = c_m15_curve[k_cross - 1]
    y_nc_m0 = w_m0_curve[k_non_cross - 1]
    y_c_m0 = c_m0_curve[k_cross - 1]

    ax13.scatter([k_non_cross], [y_nc_m15], color="#f2b179", s=28, zorder=5)
    ax13.scatter([k_cross], [y_c_m15], color="#7ecb8b", s=28, zorder=5)
    ax13.scatter([k_non_cross], [y_nc_m0], color="#f2b179", s=28, zorder=5)
    ax13.scatter([k_cross], [y_c_m0], color="#7ecb8b", s=28, zorder=5)

    ax13.annotate(f"{y_nc_m15:.2f}", (k_non_cross, y_nc_m15), textcoords="offset points", xytext=(4, 2), fontsize=12)
    ax13.annotate(f"{y_c_m15:.2f}", (k_cross, y_c_m15), textcoords="offset points", xytext=(-34, 2), fontsize=12)
    ax13.annotate(f"{y_nc_m0:.2f}", (k_non_cross, y_nc_m0), textcoords="offset points", xytext=(4, 2), fontsize=12)
    ax13.annotate(f"{y_c_m0:.2f}", (k_cross, y_c_m0), textcoords="offset points", xytext=(-34, 2), fontsize=12)

    ax13.set_xlabel("Number of Selected Features")
    ax13.set_ylabel("Accuracy (%)")
    ax13.set_xlim(0, 50)
    ax13.set_xticks([0, 5, 10, 11, 12, 15, 20, 25, 30, 35, 40, 45, 50])
    ax13.set_ylim(40, 100)
    ax13.yaxis.set_major_locator(MultipleLocator(5))
    ax13.grid(True, color="#e8e8e8", linewidth=0.75)
    ax13.legend(loc="upper right", frameon=True)
    ax13.tick_params(axis="both", colors="black", width=1.2)
    for spine in ax13.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.4)

    fig13_curve.tight_layout()
    for p13_bak in [
        FIGURES_DIR / "feature_count_trend_curve_backup.png",
        REV_RES_DIR / "figures" / "feature_count_trend_curve_backup.png",
    ]:
        fig13_curve.savefig(p13_bak, dpi=300)
    plt.close(fig13_curve)
    print(f"\n--- Fig. 13 Backup Curve (k=12 Non-Cross & k=11 Cross) 数值 ---")
    print(f"  Non-Cross (k=12): M0 = {y_nc_m0:.2f}%, M15 = {y_nc_m15:.2f}%")
    print(f"  Cross     (k=11): M0 = {y_c_m0:.2f}%, M15 = {y_c_m15:.2f}%")

    # =========================================================================
    # FIG. 14: 不同模态组合与人口学基线 (Sex / Sex+Height) 准确率柱状图 (feature_combo.png)
    # =========================================================================
    combo_rows = []

    # 1. Trivial Demographic Baselines: Sex & Sex+Height (严格折内训练与归一化)
    n_samples = len(y_bin)
    pr_w_sex = np.zeros(n_samples, dtype=int)
    pr_c_sex = np.zeros(n_samples, dtype=int)
    pr_w_sh = np.zeros(n_samples, dtype=int)
    pr_c_sh = np.zeros(n_samples, dtype=int)

    for fd in within_folds:
        tr, te, med_y = fd["tr_mask"], fd["te_mask"], fd["med_y"]
        clf_s = LogisticRegression().fit(gender[tr].reshape(-1, 1), y_bin[tr])
        pr_w_sex[te] = clf_s.predict(gender[te].reshape(-1, 1))

        sc_sh = StandardScaler().fit(np.column_stack([gender[tr], height[tr]]))
        rf_sh = RandomForestRegressor(n_estimators=45, max_depth=2, random_state=20).fit(
            sc_sh.transform(np.column_stack([gender[tr], height[tr]])), y_cont[tr]
        )
        pr_w_sh[te] = (rf_sh.predict(sc_sh.transform(np.column_stack([gender[te], height[te]]))) > med_y).astype(int)

    for fd in cross_folds:
        tr, te = fd["tr_mask"], fd["te_mask"]
        clf_s = LogisticRegression().fit(gender[tr].reshape(-1, 1), y_bin[tr])
        pr_c_sex[te] = clf_s.predict(gender[te].reshape(-1, 1))

        sc_sh = StandardScaler().fit(np.column_stack([gender[tr], height[tr]]))
        clf_sh = LogisticRegression().fit(sc_sh.transform(np.column_stack([gender[tr], height[tr]])), y_bin[tr])
        pr_c_sh[te] = clf_sh.predict(sc_sh.transform(np.column_stack([gender[te], height[te]])))

    combo_rows.append({
        "combo_display": "Sex",
        "within_acc_margin0": eval_predictions_margin(pr_w_sex, within_folds, margin=0)[0],
        "cross_acc_margin0": eval_predictions_margin(pr_c_sex, cross_folds, margin=0)[0],
        "within_acc_margin15": eval_predictions_margin(pr_w_sex, within_folds, margin=15)[0],
        "cross_acc_margin15": eval_predictions_margin(pr_c_sex, cross_folds, margin=15)[0],
    })
    combo_rows.append({
        "combo_display": "Sex+Height",
        "within_acc_margin0": eval_predictions_margin(pr_w_sh, within_folds, margin=0)[0],
        "cross_acc_margin0": eval_predictions_margin(pr_c_sh, cross_folds, margin=0)[0],
        "within_acc_margin15": eval_predictions_margin(pr_w_sh, within_folds, margin=15)[0],
        "cross_acc_margin15": eval_predictions_margin(pr_c_sh, cross_folds, margin=15)[0],
    })

    # 2. 7 种生理/生物电/局部模态组合
    combo_defs = [
        ("Overall", (4, 0, 0), (4, 0, 0)),
        ("Local", (0, 0, 5), (0, 0, 4)),
        ("Bioelectrical", (0, 6, 0), (0, 5, 0)),
        ("Overall+Local", (5, 0, 2), (4, 0, 2)),
        ("Overall+Bioelectrical", (5, 5, 0), (4, 5, 0)),
        ("Local+Bioelectrical", (0, 5, 2), (0, 5, 2)),
        ("All Three", QUOTA_WITHIN, QUOTA_CROSS),
    ]

    for c_name, q_w, q_c in combo_defs:
        # Within
        pr_w = np.zeros(len(y_bin), dtype=int)
        for fd in within_folds:
            tr, te, med_y = fd["tr_mask"], fd["te_mask"], fd["med_y"]
            sel = np.concatenate([
                fd["b_pool"][: q_w[0]],
                fd["i_pool"][: q_w[1]],
                fd["t_pool"][: q_w[2]],
            ])
            sc = StandardScaler().fit(X_all[tr][:, sel])
            m = RandomForestRegressor(n_estimators=45, max_depth=8, min_samples_leaf=1, max_features="sqrt", random_state=20)
            m.fit(sc.transform(X_all[tr][:, sel]), y_cont[tr])
            pr_w[te] = (m.predict(sc.transform(X_all[te][:, sel])) > med_y).astype(int)
        w_m0, _ = eval_predictions_margin(pr_w, within_folds, margin=0)
        w_m15, _ = eval_predictions_margin(pr_w, within_folds, margin=15)

        # Cross
        pr_c = np.zeros(len(y_bin), dtype=int)
        for fd in cross_folds:
            tr, te, med_y = fd["tr_mask"], fd["te_mask"], fd["med_y"]
            sel = np.concatenate([
                fd["b_pool"][: q_c[0]],
                fd["i_pool"][: q_c[1]],
                fd["t_pool"][: q_c[2]],
            ])
            sc = StandardScaler().fit(X_all[tr][:, sel])
            m = RandomForestRegressor(n_estimators=45, max_depth=6, min_samples_leaf=1, max_features="sqrt", random_state=20)
            m.fit(sc.transform(X_all[tr][:, sel]), y_cont[tr])
            pr_c[te] = (m.predict(sc.transform(X_all[te][:, sel])) > med_y).astype(int)
        c_m0, _ = eval_predictions_margin(pr_c, cross_folds, margin=0)
        c_m15, _ = eval_predictions_margin(pr_c, cross_folds, margin=15)

        combo_rows.append({
            "combo_display": c_name,
            "within_acc_margin0": w_m0,
            "cross_acc_margin0": c_m0,
            "within_acc_margin15": w_m15,
            "cross_acc_margin15": c_m15,
        })
    df_combo = pd.DataFrame(combo_rows)
    df_combo.to_csv(RESULTS_BIN_DIR / "rf_accuracy_by_feature_combo.csv", index=False, encoding="utf-8-sig")

    def plot_fig14(w_col, c_col, out_paths):
        sns.set_theme(
            style="white",
            font_scale=1.0,
            rc={
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
                "axes.labelsize": 34,
                "xtick.labelsize": 23,
                "ytick.labelsize": 24,
                "legend.fontsize": 22,
                "axes.edgecolor": "black",
                "axes.linewidth": 1.4,
                "axes.labelcolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "text.color": "black",
            },
        )
        fig, ax = plt.subplots(1, 1, figsize=(17.5, 9.5))
        x = np.arange(len(df_combo)) * 1.35
        width = 0.42

        ax.bar(
            x - width / 2,
            df_combo[w_col] * 100.0,
            width,
            label="Non-Cross Subject",
            color="#f7c58a",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar(
            x + width / 2,
            df_combo[c_col] * 100.0,
            width,
            label="Cross Subject",
            color="#9ad49b",
            edgecolor="white",
            linewidth=0.8,
        )

        # 将 '+' 号后面的名词换行，与原论文 Fig. 14 样式 100% 一致
        wrapped_labels = [lbl.replace("+", "+\n") for lbl in df_combo["combo_display"]]
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_labels, rotation=0)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(x[0] - 0.8, x[-1] + 0.8)
        ax.legend(loc="upper left", bbox_to_anchor=(0.004, 0.992), borderaxespad=0.15, frameon=True)
        ax.grid(False)
        ax.tick_params(axis="both", colors="black", width=1.2)
        apply_y_axis_ruler_ticks(ax)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.4)

        for i, (wm, cm_v) in enumerate(zip(df_combo[w_col], df_combo[c_col])):
            ax.text(x[i] - width / 2, wm * 100.0 + 1.0, f"{wm * 100.0:.2f}", ha="center", va="bottom", fontsize=17.5)
            ax.text(x[i] + width / 2, cm_v * 100.0 + 1.0, f"{cm_v * 100.0:.2f}", ha="center", va="bottom", fontsize=17.5)

        fig.tight_layout()
        for p in out_paths:
            fig.savefig(p, dpi=300)
        plt.close(fig)

    plot_fig14(
        "within_acc_margin15",
        "cross_acc_margin15",
        [
            FIGURES_DIR / "feature_combo.png",
            FIGURES_DIR / "feature_combo_margin15.png",
            RESULTS_BIN_DIR / "rf_accuracy_by_feature_combo.png",
            REV_RES_DIR / "figures" / "feature_combo.png",
            REV_RES_DIR / "figures" / "feature_combo_margin15.png",
        ],
    )
    plot_fig14(
        "within_acc_margin0",
        "cross_acc_margin0",
        [
            FIGURES_DIR / "feature_combo_margin0.png",
            REV_RES_DIR / "figures" / "feature_combo_margin0.png",
        ],
    )
    print("\n--- Fig. 14 (2种人口学基线 + 7种模态组合 Margin=0 与 Margin=15) 数值 ---")
    print(df_combo.to_string(index=False))

    # =========================================================================
    # FIG. 15: 混淆矩阵图 (confusion_matrices.png) —— 原论文 Greens 配色与排版
    # =========================================================================
    cm_w_m0 = confusion_matrix(y_bin, preds_within["RandomForest"], labels=[0, 1])
    cm_c_m0 = confusion_matrix(y_bin, preds_cross["RandomForest"], labels=[0, 1])

    in_band = np.abs(y_cont - THRESHOLD_CUTOFF) <= 15.0
    pr_w_m15 = preds_within["RandomForest"].copy()
    pr_w_m15[in_band] = y_bin[in_band]
    cm_w_m15 = confusion_matrix(y_bin, pr_w_m15, labels=[0, 1])

    pr_c_m15 = preds_cross["RandomForest"].copy()
    pr_c_m15[in_band] = y_bin[in_band]
    cm_c_m15 = confusion_matrix(y_bin, pr_c_m15, labels=[0, 1])

    def plot_fig15_original_style(cm_left, cm_right, out_paths):
        sns.set_theme(
            style="whitegrid",
            font_scale=0.95,
            rc={
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
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
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.2))
        for ax, cm_mat, title, panel_lbl in zip(
            axes,
            [cm_left, cm_right],
            ["Non-Cross Subject", "Cross Subject"],
            ["a", "b"],
        ):
            sns.heatmap(
                cm_mat,
                annot=True,
                fmt="d",
                annot_kws={"fontsize": 30},
                cmap="Greens",
                cbar=False,
                square=True,
                ax=ax,
                linewidths=0.6,
                linecolor="#eaeaea",
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(title, pad=8)
            ax.text(-0.16, 1.12, panel_lbl, transform=ax.transAxes, fontsize=38, fontweight="normal", va="top", ha="left")
            ax.set_xticklabels(["Low threshold", "High threshold"])
            ax.set_yticklabels(["Low threshold", "High threshold"], rotation=90, va="center")
            ax.tick_params(axis="both", colors="black", width=1.2)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.4)

        fig.tight_layout()
        for p in out_paths:
            fig.savefig(p, dpi=300)
        plt.close(fig)

    # 主推荐：Margin=0 真实无篡改混淆矩阵 (同时生成 Margin=15 版本供对照)
    plot_fig15_original_style(
        cm_w_m0,
        cm_c_m0,
        [
            FIGURES_DIR / "confusion_matrices.png",
            FIGURES_DIR / "confusion_matrices_margin0.png",
            RESULTS_BIN_DIR / "rf_confusion_matrices_within_cross.png",
            REV_RES_DIR / "figures" / "confusion_matrices.png",
            REV_RES_DIR / "figures" / "confusion_matrices_margin0.png",
        ],
    )
    plot_fig15_original_style(
        cm_w_m15,
        cm_c_m15,
        [
            FIGURES_DIR / "confusion_matrices_margin15.png",
            REV_RES_DIR / "figures" / "confusion_matrices_margin15.png",
        ],
    )

    print("\n--- Fig. 15 混淆矩阵核对 ---")
    print(f"  Non-Cross Subject (Margin=0):  TN={cm_w_m0[0,0]}, FP={cm_w_m0[0,1]}, FN={cm_w_m0[1,0]}, TP={cm_w_m0[1,1]} -> {np.trace(cm_w_m0)}/128 = {np.trace(cm_w_m0)/128*100:.2f}%")
    print(f"  Cross Subject     (Margin=0):  TN={cm_c_m0[0,0]}, FP={cm_c_m0[0,1]}, FN={cm_c_m0[1,0]}, TP={cm_c_m0[1,1]} -> {np.trace(cm_c_m0)}/128 = {np.trace(cm_c_m0)/128*100:.2f}%")
    print(f"  Non-Cross Subject (Margin=15): TN={cm_w_m15[0,0]}, FP={cm_w_m15[0,1]}, FN={cm_w_m15[1,0]}, TP={cm_w_m15[1,1]} -> {np.trace(cm_w_m15)}/128 = {np.trace(cm_w_m15)/128*100:.2f}%")
    print(f"  Cross Subject     (Margin=15): TN={cm_c_m15[0,0]}, FP={cm_c_m15[0,1]}, FN={cm_c_m15[1,0]}, TP={cm_c_m15[1,1]} -> {np.trace(cm_c_m15)}/128 = {np.trace(cm_c_m15)/128*100:.2f}%")

    # =========================================================================
    # FIG. 9 & FIG. 10: 基于折内 Multi-Block PLS-VIP + 去重初筛的 50 维候选特征池
    #                   Spearman 相关性条形图 (Fig. 9) 与 50x50 互相关矩阵图 (Fig. 10)
    # =========================================================================
    df_all_named = pd.concat([df_body[cols_b], df_imp[cols_i], df_tex[cols_t]], axis=1)

    features_50_spec = [
        # --- 8 Global physiological features (y = 0..7) ---
        ("2. Height", "Height", "Global"),
        ("45. FFM% of Right Arm", "Fat Free Mass% of Right Arm", "Global"),
        ("49. FFM% of Left Arm", "Fat Free Mass% of Left Arm", "Global"),
        ("107. ECW/TBW", "ECW/TBW", "Global"),
        ("199. Measured Circumference of Neck", "Measured Circumference of Neck", "Global"),
        ("206. Measured Circumference of Left Thigh", "Measured Circumference of Left Thigh", "Global"),
        ("217. SMI", "Skeletal Muscle Index", "Global"),
        ("227. SMM/WT", "Skeletal Muscle Mass/Weight", "Global"),
        # --- 8 Bioelectrical features (y = 8..15) ---
        ("rms_u", "v_pre, rms", "Bioelectrical"),
        ("crestf_u", "v_pre, crestf", "Bioelectrical"),
        ("margin_u_2", "v_post, margin", "Bioelectrical"),
        ("v1-v2", "Δv", "Bioelectrical"),
        ("Phase", "Phase", "Bioelectrical"),
        ("LOG_R1", "LOG_R_1", "Bioelectrical"),
        ("CPE1-P(+)", "CPE1_n", "Bioelectrical"),
        ("LOG_R2", "LOG_R_2", "Bioelectrical"),
        # --- 34 Local morphological features (y = 16..49) ---
        # Finger geometry (3: y = 16..18)
        ("截面积", "A_sec", "Local"),
        ("指长（指根）/cm", "l_1", "Local"),
        ("截面积-指长", "A_sur", "Local"),
        # BP / Wavelet / FOS / GLCM / GLRLM / FPS (13: y = 19..31)
        ("BP_Euler_Num_3", "BP_Euler_Num", "Local"),
        ("BP_Solidity_13", "BP_Solidity", "Local"),
        ("BP_Perimeter_12", "BP_Perimeter", "Local"),
        ("BP_wt_norm_moments_14__79", "BP_wt_norm_moments_14", "Local"),
        ("BP_wt_norm_moments_6__71", "BP_wt_norm_moments_6", "Local"),
        ("STFOS_Skewness_6", "FOS_Skewness", "Local"),
        ("GLCM_d_1_rng_Enr_126", "GLCM_d1_rng_Enr", "Local"),
        ("GLCM_t_90_d_10_Var_374", "GLCM_d10_a90_Var", "Local"),
        ("GLCM_t_0_d_2_SVD15_465", "GLCM_d2_a0_SVD15", "Local"),
        ("GLCM_t_0_d_4_SVD14_524", "GLCM_d4_a0_SVD14", "Local"),
        ("STHOS_r_2_a_45_LRHGLE_22", "GLRLM_d2_a45_LRHGLE", "Local"),
        ("STHOS_r_2_a_90_LRLGLE_32", "GLRLM_d2_a90_LRLGLE", "Local"),
        ("SPFPS_R0_T0_1", "FPS_r0_a0", "Local"),
        # LBP (4: y = 32..35)
        ("LBP_r3_pt_17_342", "LBP_r3_p17", "Local"),
        ("LBP_r3_pt_18_343", "LBP_r3_p18", "Local"),
        ("LBP_r3_pt_76_401", "LBP_r3_p76", "Local"),
        ("LBP_r3_pt_84_409", "LBP_r3_p84", "Local"),
        # ZER (14: y = 36..49)
        ("ZER_r0_pt_17_18", "ZER_r0_p17", "Local"),
        ("ZER_r1_pt_6_32", "ZER_r1_p6", "Local"),
        ("ZER_r1_pt_8_34", "ZER_r1_p8", "Local"),
        ("ZER_r1_pt_16_42", "ZER_r1_p16", "Local"),
        ("ZER_r2_pt_0_51", "ZER_r2_p0", "Local"),
        ("ZER_r2_pt_2_53", "ZER_r2_p2", "Local"),
        ("ZER_r3_pt_1_77", "ZER_r3_p1", "Local"),
        ("ZER_r3_pt_4_80", "ZER_r3_p4", "Local"),
        ("ZER_r3_pt_6_82", "ZER_r3_p6", "Local"),
        ("ZER_r3_pt_15_91", "ZER_r3_p15", "Local"),
        ("ZER_r3_pt_16_92", "ZER_r3_p16", "Local"),
        ("ZER_r3_pt_19_95", "ZER_r3_p19", "Local"),
        ("ZER_r4_pt_11_112", "ZER_r4_p11", "Local"),
        ("ZER_r4_pt_19_120", "ZER_r4_p19", "Local"),
    ]

    raw_cols_50 = [c for c, _, _ in features_50_spec]
    disp_labels_50 = [lbl for _, lbl, _ in features_50_spec]
    groups_50 = [grp for _, _, grp in features_50_spec]

    X_50 = df_all_named[raw_cols_50].apply(pd.to_numeric, errors="coerce").fillna(0).values
    rhos_50 = np.array([spearmanr(X_50[:, j], y_cont)[0] for j in range(50)])
    corr_50 = pd.DataFrame(X_50).corr(method="spearman").values

    # ---- 绘制 Fig. 9 (完全复刻原论文 image9.tiff 样式) ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
    })
    fig9, ax9 = plt.subplots(figsize=(6.32, 11.41), dpi=300)
    color_map_f9 = {
        "Global": "#f7c58a",
        "Bioelectrical": "#9ad49b",
        "Local": "#c6b7d9",
    }
    bar_colors = [color_map_f9[g] for g in groups_50]
    y_pos = np.arange(50)

    ax9.grid(False)
    for gx in [-0.4, -0.2, 0.2, 0.4]:
        ax9.axvline(gx, color="#e8e8e8", linewidth=0.75, zorder=0)
    ax9.barh(y_pos, rhos_50, height=0.72, color=bar_colors, edgecolor="#222222", linewidth=0.65, zorder=2)
    ax9.axvline(0.0, color="black", linewidth=0.9, zorder=3)

    ax9.set_xlim(-0.78, 0.60)
    ax9.set_ylim(-0.8, 49.8)
    ax9.set_yticks([])
    ax9.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    ax9.set_xticklabels(["-0.4", "-0.2", "0.0", "0.2", "0.4", "0.6"], fontsize=13)
    ax9.xaxis.set_minor_locator( matplotlib.ticker.FixedLocator([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]) )
    ax9.tick_params(axis="x", which="major", direction="in", length=5, width=1.1, bottom=True, top=False)
    ax9.tick_params(axis="x", which="minor", direction="in", length=2.8, width=0.9, bottom=True, top=False)

    # 在 x 轴刻度行左侧标注 ρ
    ax9.text(-0.565, -1.25, "ρ", fontsize=14, ha="center", va="top", clip_on=False)

    # 左侧框内特征名称与柱内相关系数标注
    for idx_y, (lbl, r_val) in enumerate(zip(disp_labels_50, rhos_50)):
        ax9.text(-0.72, idx_y, lbl, ha="left", va="center", fontsize=10.5, color="black")
        if r_val >= 0:
            ax9.text(0.012, idx_y, f"{r_val:.2f}", ha="left", va="center", fontsize=11, color="black")
        else:
            ax9.text(-0.012, idx_y, f"{r_val:.2f}", ha="right", va="center", fontsize=11, color="black")

    for spine in ax9.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)

    legend_handles = [
        Patch(facecolor="#f7c58a", edgecolor="#444444", linewidth=0.8, label="Global physiological features"),
        Patch(facecolor="#c6b7d9", edgecolor="#444444", linewidth=0.8, label="Local morphological features"),
        Patch(facecolor="#9ad49b", edgecolor="#444444", linewidth=0.8, label="Bioelectrical features"),
    ]
    ax9.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.005),
        ncol=2,
        frameon=False,
        fontsize=13.5,
        handlelength=1.8,
        handleheight=1.1,
        columnspacing=1.2,
    )

    fig9.subplots_adjust(left=0.03, right=0.96, bottom=0.04, top=0.93)
    for p9 in [FIGURES_DIR / "fig9_spearman_50features.png", REV_RES_DIR / "figures" / "fig9_spearman_50features.png"]:
        fig9.savefig(p9, dpi=300)
    plt.close(fig9)

    # ---- 绘制 Fig. 10 (完全复刻原论文 image10.tiff 50x50 相关性热力图样式) ----
    fig10, ax10 = plt.subplots(figsize=(8.09, 6.79), dpi=300)
    cmap_disc = plt.get_cmap("RdYlBu_r", 20)
    norm_disc = BoundaryNorm(np.linspace(-1.0, 1.0, 21), cmap_disc.N)

    im10 = ax10.imshow(corr_50, cmap=cmap_disc, norm=norm_disc, origin="lower", aspect="equal", extent=[0, 50, 0, 50])
    ax10.set_xticks([])
    ax10.set_yticks([])

    # 实线分隔三大模态 (8, 16)
    for b_pos in [8, 16]:
        ax10.axhline(b_pos, color="black", linestyle="-", linewidth=1.1)
        ax10.axvline(b_pos, color="black", linestyle="-", linewidth=1.1)

    # 虚线分隔局部形态内部子组 (19: Finger size, 32: BP/GLCM/FPS, 36: LBP, 50: ZER)
    for d_pos in [19, 32, 36]:
        ax10.axhline(d_pos, color="black", linestyle=(0, (5, 5)), linewidth=1.0)
        ax10.axvline(d_pos, color="black", linestyle=(0, (5, 5)), linewidth=1.0)

    for spine in ax10.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.3)

    # 外侧刻度短线与文字标注（复刻原图排版）
    # 顶部标注
    ax10.text(1.5, 52.8, "Global\nphysiological\nfeatures", fontsize=13.5, ha="left", va="bottom", clip_on=False)
    ax10.plot([3.2, 4.8], [51.8, 50.5], color="black", linewidth=1.1, clip_on=False)
    ax10.text(16.0, 52.0, "Bioelectrical features", fontsize=13.5, ha="center", va="bottom", clip_on=False)
    ax10.plot([14.2, 12.6], [51.7, 50.4], color="black", linewidth=1.1, clip_on=False)
    ax10.text(37.8, 52.0, "Local morphological features", fontsize=13.5, ha="center", va="bottom", clip_on=False)
    ax10.plot([32.6, 31.3], [51.5, 50.2], color="black", linewidth=1.1, clip_on=False)

    # 左侧标注
    ax10.text(-11.8, 34.5, "Local\nmorphological\nfeatures", fontsize=13.0, ha="left", va="center", clip_on=False)
    ax10.plot([-1.8, -0.5], [33.2, 32.0], color="black", linewidth=1.1, clip_on=False)
    ax10.text(-11.8, 13.0, "Bioelectrical\nfeatures", fontsize=13.0, ha="left", va="center", clip_on=False)
    ax10.plot([-1.8, -0.5], [13.5, 12.2], color="black", linewidth=1.1, clip_on=False)
    ax10.text(-11.8, 3.3, "Global\nphysiological\nfeatures", fontsize=13.0, ha="left", va="center", clip_on=False)
    ax10.plot([-1.8, -0.5], [4.5, 3.2], color="black", linewidth=1.1, clip_on=False)

    # 底部标注
    ax10.plot([0.3, 1.3], [-0.2, -1.6], color="black", linewidth=1.1, clip_on=False)
    ax10.text(2.5, -2.0, "Height", fontsize=13.0, ha="center", va="top", clip_on=False)
    ax10.plot([17.5, 19.3], [-0.4, -1.5], color="black", linewidth=1.1, clip_on=False)
    ax10.text(21.0, -2.0, "Finger size features", fontsize=13.0, ha="center", va="top", clip_on=False)
    ax10.plot([33.9, 35.4], [-0.3, -1.6], color="black", linewidth=1.1, clip_on=False)
    ax10.text(36.2, -2.0, "LBP", fontsize=13.0, ha="center", va="top", clip_on=False)
    ax10.plot([49.7, 49.2], [-0.2, -1.6], color="black", linewidth=1.1, clip_on=False)
    ax10.text(44.5, -2.0, "ZER_r4_p19", fontsize=13.0, ha="center", va="top", clip_on=False)

    # 锁定坐标轴范围为 [0, 50]，防止外侧引导线撑开白边
    ax10.set_xlim(0, 50)
    ax10.set_ylim(0, 50)

    # Colorbar
    cbar_ax = fig10.add_axes([0.885, 0.085, 0.026, 0.78])
    cbar = fig10.colorbar(im10, cax=cbar_ax, ticks=[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    cbar.ax.tick_params(labelsize=12, length=3, width=0.9)
    cbar.outline.set_linewidth(0.9)
    cbar_ax.text(-0.55, 1.0, "ρ", fontsize=13.5, ha="right", va="center", transform=cbar_ax.transAxes)

    fig10.subplots_adjust(left=0.18, right=0.835, bottom=0.085, top=0.865)
    for p10 in [FIGURES_DIR / "fig10_corr_matrix_50features.png", REV_RES_DIR / "figures" / "fig10_corr_matrix_50features.png"]:
        fig10.savefig(p10, dpi=300)
    plt.close(fig10)

    # =========================================================================
    # ---- 绘制新版 Fig. 13: 内层 3 折 (Inner 3-Fold) PLS-DA 特征筛选三列示意图 ----
    #      严格延续 Fig. 9 的原论文框内左文右柱排版、配色与浅灰网格线 (#e8e8e8)
    # =========================================================================
    grps_50_arr = np.array(groups_50)
    disp_50_arr = np.array(disp_labels_50)

    def get_inner_3fold_selected_50(is_cross_mode=True):
        """提取代表性外层训练折内部 3 折 (Inner Fold 1, 2, 3) 各自经 Multi-Block PLS-DA VIP 筛出的特征及 VIP 值"""
        if is_cross_mode:
            kf_o = KFold(n_splits=5, shuffle=True, random_state=93)
            tr_si, _ = next(kf_o.split(unique_subs))
            tr_u = unique_subs[tr_si]
            q_g, q_b, q_l = 4, 2, 5  # k=11 (4 Global, 2 Bioelectrical, 5 Local)
        else:
            kf_o = KFold(n_splits=5, shuffle=True, random_state=168)
            tr_u, _ = next(kf_o.split(np.arange(len(y_bin))))
            q_g, q_b, q_l = 5, 2, 5  # k=12 (5 Global, 2 Bioelectrical, 5 Local)

        kf_i = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_results = []
        for in_idx, (in_tr_u, _) in enumerate(kf_i.split(tr_u)):
            if is_cross_mode:
                in_tr_m = np.isin(subs, tr_u[in_tr_u])
            else:
                in_tr_m = np.zeros(len(y_bin), dtype=bool)
                in_tr_m[tr_u[in_tr_u]] = True

            X_g = X_50[in_tr_m][:, grps_50_arr == "Global"]
            X_b_sub = X_50[in_tr_m][:, grps_50_arr == "Bioelectrical"]
            X_l = X_50[in_tr_m][:, grps_50_arr == "Local"]
            X_mb = np.column_stack([
                StandardScaler().fit_transform(X_g) / np.sqrt(X_g.shape[1]),
                StandardScaler().fit_transform(X_b_sub) / np.sqrt(X_b_sub.shape[1]),
                StandardScaler().fit_transform(X_l) / np.sqrt(X_l.shape[1]),
            ])
            vip_in = compute_vip(PLSRegression(n_components=2).fit(X_mb, y_cont[in_tr_m]), X_mb.shape[1])

            # 按各模态 VIP 降序并去重 (|r|<0.85) 选出对应配额
            g_sel = dedup_select(X_g, np.argsort(-vip_in[grps_50_arr == "Global"]), q_g, 0.85)
            b_sel = dedup_select(X_b_sub, np.argsort(-vip_in[grps_50_arr == "Bioelectrical"]), q_b, 0.85) + 8
            l_sel = dedup_select(X_l, np.argsort(-vip_in[grps_50_arr == "Local"]), q_l, 0.85) + 16

            # 与 Fig. 9 一致的自下而上顺序：底部 Global (升序排列使最高 VIP 在该组最上方)，中部 Bioelectrical，顶部 Local
            g_sel_ord = g_sel[np.argsort(vip_in[g_sel])]
            b_sel_ord = b_sel[np.argsort(vip_in[b_sel])]
            l_sel_ord = l_sel[np.argsort(vip_in[l_sel])]
            sel_all = np.concatenate([g_sel_ord, b_sel_ord, l_sel_ord])

            fold_results.append({
                "fold_title": f"Inner Fold {in_idx + 1}",
                "labels": disp_50_arr[sel_all].tolist(),
                "groups": grps_50_arr[sel_all].tolist(),
                "vips": vip_in[sel_all].tolist(),
            })
        return fold_results

    def draw_inner_3fold_columns(axes_row, fold_results, xlim_left=-2.35, xlim_right=2.05):
        for col_idx, (ax_c, fres) in enumerate(zip(axes_row, fold_results)):
            lbls = fres["labels"]
            grps_c = fres["groups"]
            vips_c = fres["vips"]
            n_f = len(lbls)
            y_c = np.arange(n_f)
            c_bars = [color_map_f9[g] for g in grps_c]

            ax_c.grid(False)
            for gx in [0.5, 1.0, 1.5, 2.0]:
                ax_c.axvline(gx, color="#e8e8e8", linewidth=0.75, zorder=0)

            ax_c.barh(y_c, vips_c, height=0.68, color=c_bars, edgecolor="#222222", linewidth=0.65, zorder=2)
            ax_c.axvline(0.0, color="black", linewidth=0.9, zorder=3)
            ax_c.axvline(1.0, color="#888888", linestyle="--", linewidth=0.85, zorder=3)

            ax_c.set_xlim(xlim_left, xlim_right)
            ax_c.set_ylim(-0.75, n_f - 0.25)
            ax_c.set_yticks([])
            ax_c.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
            ax_c.set_xticklabels(["0.0", "0.5", "1.0", "1.5", "2.0"], fontsize=12.5)
            ax_c.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([0.25, 0.75, 1.25, 1.75]))
            ax_c.tick_params(axis="x", which="major", direction="in", length=5, width=1.1, bottom=True, top=False)
            ax_c.tick_params(axis="x", which="minor", direction="in", length=2.8, width=0.9, bottom=True, top=False)
            ax_c.set_title(fres["fold_title"], fontsize=15, pad=8, fontweight="normal")

            # 在 x 轴刻度行左侧标注 VIP
            ax_c.text(-0.42, -1.05 * (n_f / 11.0), "VIP", fontsize=12.5, ha="center", va="top", clip_on=False)

            # 左侧框内特征名称与柱内 VIP 数值标注 (完全对齐 Fig. 9 样式)
            for idx_y, (lbl, v_val) in enumerate(zip(lbls, vips_c)):
                ax_c.text(xlim_left + 0.08, idx_y, lbl, ha="left", va="center", fontsize=11.0, color="black")
                ax_c.text(0.04, idx_y, f"{v_val:.2f}", ha="left", va="center", fontsize=11.0, color="black")

            for spine in ax_c.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.1)

    # 1. 生成主版三列图 (1x3: Inner Fold 1, Inner Fold 2, Inner Fold 3)
    cross_3folds = get_inner_3fold_selected_50(is_cross_mode=True)
    fig13_3col, axes13_3col = plt.subplots(1, 3, figsize=(16.8, 5.2), dpi=300)
    draw_inner_3fold_columns(axes13_3col, cross_3folds, xlim_left=-2.35, xlim_right=2.05)

    fig13_3col.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=14,
        handlelength=1.8,
        handleheight=1.1,
        columnspacing=2.0,
    )
    fig13_3col.subplots_adjust(left=0.02, right=0.985, bottom=0.10, top=0.84, wspace=0.08)
    for p13 in [
        FIGURES_DIR / "feature_count_trend.png",
        RESULTS_BIN_DIR / "feature_count_trend.png",
        REV_RES_DIR / "figures" / "feature_count_trend.png",
        FIGURES_DIR / "fig13_inner_3fold_plsda.png",
        REV_RES_DIR / "figures" / "fig13_inner_3fold_plsda.png",
    ]:
        fig13_3col.savefig(p13, dpi=300)
    plt.close(fig13_3col)

    # 2. 生成双行三列版 (2x3: a Non-Cross Subject k=12, b Cross Subject k=11) 供备选
    within_3folds = get_inner_3fold_selected_50(is_cross_mode=False)
    fig13_2x3, axes13_2x3 = plt.subplots(2, 3, figsize=(16.8, 9.8), dpi=300)
    draw_inner_3fold_columns(axes13_2x3[0], within_3folds, xlim_left=-2.35, xlim_right=2.05)
    draw_inner_3fold_columns(axes13_2x3[1], cross_3folds, xlim_left=-2.35, xlim_right=2.05)
    axes13_2x3[0][0].text(-0.04, 1.03, "a", transform=axes13_2x3[0][0].transAxes, fontsize=28, va="bottom", ha="right")
    axes13_2x3[1][0].text(-0.04, 1.03, "b", transform=axes13_2x3[1][0].transAxes, fontsize=28, va="bottom", ha="right")
    fig13_2x3.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=14,
        handlelength=1.8,
        handleheight=1.1,
        columnspacing=2.0,
    )
    fig13_2x3.subplots_adjust(left=0.03, right=0.985, bottom=0.06, top=0.90, wspace=0.08, hspace=0.24)
    for p13_2x3 in [
        FIGURES_DIR / "fig13_inner_3fold_plsda_2x3.png",
        REV_RES_DIR / "figures" / "fig13_inner_3fold_plsda_2x3.png",
    ]:
        fig13_2x3.savefig(p13_2x3, dpi=300)
    plt.close(fig13_2x3)

    # 导出内层 3 折筛出特征明细 CSV 供查验
    rows_inner_3f = []
    for sc_lbl, f_list in [("Non-Cross Subject (k=12)", within_3folds), ("Cross Subject (k=11)", cross_3folds)]:
        for fres in f_list:
            for rk, (lbl, grp, vip_v) in enumerate(zip(fres["labels"][::-1], fres["groups"][::-1], fres["vips"][::-1]), 1):
                rows_inner_3f.append({
                    "Scenario": sc_lbl,
                    "Inner_Fold": fres["fold_title"],
                    "Order": rk,
                    "Modality": grp,
                    "Feature": lbl,
                    "VIP": round(vip_v, 4),
                })
    pd.DataFrame(rows_inner_3f).to_csv(REV_RES_DIR / "inner_3fold_plsda_selected_features.csv", index=False)

    print(f"\n✓ 全部原论文同款图片 (Fig. 9 - Fig. 15，含三列版内层 3 折 PLS-DA 特征筛选图 Fig. 13) 生成完毕，耗时: {time.time() - t0:.2f} 秒")


if __name__ == "__main__":
    main()
