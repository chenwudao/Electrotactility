"""Reproduce historical classification predictions without ablation or association analyses.

The label cut-off is 156 us. Regressors use each training partition's PT median
as their prediction cut-off. Seeds and fixed model settings reproduce the
historical development configuration; they are not an independent validation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
df_y = pd.read_excel(DATASET_DIR / "Y.xlsx", index_col=0)
df_body = pd.read_excel(DATASET_DIR / "X_人体成分_new.xlsx", index_col=0)
df_imp = pd.read_excel(DATASET_DIR / "X_电阻抗.xlsx", index_col=0)
df_tex = pd.read_excel(DATASET_DIR / "X_纹理特征.xlsx", index_col=0)
if not all(df_y.index.equals(frame.index) for frame in (df_body, df_imp, df_tex)):
    raise ValueError("Feature and target row indices must match exactly.")
meta_cols = {"数据编号", "受试者编号", "手指", "手指（优食-0 非优中-1 非优尾-2 非优食-3）", "手指编号"}
cols_b = [c for c in df_body.columns if c not in meta_cols]
cols_i = [c for c in df_imp.columns if c not in meta_cols]
cols_t = [c for c in df_tex.columns if c not in meta_cols]
X_b = df_body[cols_b].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_i = df_imp[cols_i].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_t = df_tex[cols_t].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_all = np.column_stack([X_b, X_i, X_t])
modalities = np.array(["Body"] * len(cols_b) + ["Imp"] * len(cols_i) + ["Tex"] * len(cols_t))
all_feature_names = list(cols_b) + list(cols_i) + list(cols_t)
geometry_names = {"指围（cm）", "截面积", "指长（指根）/cm", "指围-指长", "截面积-指长"}
subs = df_y["受试者编号"].values
unique_subs = np.unique(subs)
y_cont = df_y["阈值"].values.astype(float)
THRESHOLD_CUTOFF = 156.0
y_bin = (y_cont > THRESHOLD_CUTOFF).astype(int)
gender = df_y["性别"].values.astype(float)

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
    # Historical fallback: refill to quota even if this relaxes correlation filtering.
    if len(picked) < k:
        for local_i, global_idx in enumerate(cands):
            if not valid[local_i]:
                continue
            if all(global_idx != g for _, g in picked):
                picked.append((local_i, global_idx))
                if len(picked) == k:
                    break
    return np.array([g_idx for _, g_idx in picked], dtype=int)

def prepare_folds(is_cross=True, cv_seed=93, inner_seed=42, corr_thr=0.85, fast_tex_cands=None):
    """Select each outer fold's quota using only its own inner validation scores."""
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
        inner_cands = [((4, 5, 2), 0.85, 6), ((3, 5, 2), 0.85, 6)]  # correlation threshold fixed at 0.85
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
        splits = []
        for tr_i, te_i in kf.split(np.arange(n)):
            tr_mask = np.zeros(n, dtype=bool); tr_mask[tr_i] = True
            te_mask = np.zeros(n, dtype=bool); te_mask[te_i] = True
            splits.append((tr_mask, te_mask, tr_i))
        inner_cands = [((5, 5, 2), 0.85, 8), ((4, 5, 2), 0.85, 8)]  # correlation threshold fixed at 0.85

    per_fold_inner_scores = []
    for f_idx, (tr_mask, te_mask, tr_units) in enumerate(splits):
        # Split only outer-training units into three inner folds to evaluate quotas.
        kf_in = KFold(n_splits=3, shuffle=True, random_state=inner_seed + f_idx)
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
                StandardScaler().fit_transform(X_b[in_tr_m]),
                StandardScaler().fit_transform(X_i[in_tr_m]),
                StandardScaler().fit_transform(X_t[in_tr_m]),
            ])
            vip_in = compute_vip(PLSRegression(n_components=2, scale=True).fit(X_itr_mb, y_cont[in_tr_m]), X_itr_mb.shape[1])
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
        per_fold_inner_scores.append(cand_scores)

    for f_idx, (tr_mask, te_mask, tr_units) in enumerate(splits):
        # Strict nested CV: the quota/depth for outer fold f is chosen ONLY from the inner-CV scores
        # computed inside outer fold f's own training set (no aggregation across outer folds),
        # for both the cross-subject and the non-cross-subject (sample-level) protocols.
        best_quota, _, best_depth = inner_cands[int(np.argmax(per_fold_inner_scores[f_idx]))]
        # Fit PLS-VIP on the full outer training fold and lock selected feature pools
        X_tr_mb = np.column_stack([
            StandardScaler().fit_transform(X_b[tr_mask]),
            StandardScaler().fit_transform(X_i[tr_mask]),
            StandardScaler().fit_transform(X_t[tr_mask]),
        ])
        vip = compute_vip(PLSRegression(n_components=2, scale=True).fit(X_tr_mb, y_cont[tr_mask]), X_tr_mb.shape[1])
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
            "inner_correct_counts": per_fold_inner_scores[f_idx].tolist(),
        })
    return folds_data


def feature_category(index):
    if modalities[index] == "Imp":
        return "Bioelectrical"
    if modalities[index] == "Tex" or all_feature_names[index] in geometry_names:
        return "Local morphological"
    return "Global physiological"


def downstream_models(is_cross):
    return {
        "RandomForest": RandomForestRegressor(n_estimators=45, max_depth=6 if is_cross else 8,
            min_samples_leaf=1, max_features="sqrt", random_state=20),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=60, max_depth=6 if is_cross else 8,
            random_state=1 if is_cross else 20),
        "KNN": KNeighborsRegressor(n_neighbors=10 if is_cross else 11,
            weights="distance", p=1 if is_cross else 2),
        "SVM_SVR": SVR(kernel="rbf", C=2.0 if is_cross else 3.0),
    }


def plot_results(metrics, predictions, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Non-cross-subject", "Cross-subject"]
    colors = ["#f2b179", "#7ecb8b"]
    model_names = list(downstream_models(True)) + ["SexOnly", "Majority"]
    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for i, scenario in enumerate(labels):
        values = metrics.loc[(metrics.scenario == scenario) & (metrics.margin_us == 0)].set_index("model")
        ax.bar(x + (i - .5) * .36, values.loc[model_names, "accuracy_pooled"] * 100,
               width=.36, label=scenario, color=colors[i])
    ax.axhline(50, color="gray", linestyle="--", label="Expected 50/50 random")
    ax.set(xticks=x, xticklabels=["RF", "ExtraTrees", "KNN", "SVR", "Sex only", "Majority"],
           ylabel="Strict accuracy (%)", ylim=(0, 100))
    ax.legend(fontsize=9)
    fig.savefig(output_dir / "model_comparison.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for scenario, color in zip(labels, colors):
        rows = metrics[(metrics.scenario == scenario) & (metrics.model == "RandomForest")]
        ax.plot(rows.margin_us, rows.accuracy_pooled * 100, "o-", label=scenario, color=color)
    ax.set(xlabel="Tolerance margin (us)", ylabel="Accuracy / tolerance score (%)",
           xticks=[0, 5, 10, 15], ylim=(0, 100))
    ax.legend()
    fig.savefig(output_dir / "margin_sensitivity.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, scenario in zip(axes, labels):
        rows = predictions[predictions.scenario == scenario]
        cm = confusion_matrix(rows.y_true_bin, rows.pred_RandomForest, labels=[0, 1])
        ax.imshow(cm, cmap="Greens", vmin=0, vmax=64)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18)
        ax.set(title=scenario, xlabel="Predicted label", ylabel="True label",
               xticks=[0, 1], yticks=[0, 1])
    fig.savefig(output_dir / "confusion_matrices_strict.png", dpi=200)
    plt.close(fig)
    plot_fold_mean_paper_figures(metrics, predictions, output_dir)


def plot_fold_mean_paper_figures(metrics, predictions, output_dir):
    """Draw paper figures from equal-weight outer-fold means, including random baselines."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import AutoMinorLocator
    for candidate in font_manager.findSystemFonts():
        try:
            if font_manager.FontProperties(fname=candidate).get_name() == "Times New Roman":
                font_manager.fontManager.addfont(candidate)
        except (OSError, RuntimeError, ValueError):
            continue
    scenarios = ["Non-cross-subject", "Cross-subject"]
    names = ["RandomForest", "ExtraTrees", "KNN", "SVM_SVR"]
    labels = ["RandomForest", "ExtraTrees", "KNN", "SVM", "Baseline"]
    colors = ["#e9c498", "#a1cda2"]
    paper_rows = []
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "axes.linewidth": 1.4, "axes.edgecolor": "black"}):
        for margin in (0, 15):
            fig, ax = plt.subplots(figsize=(11.5, 8.5))
            for i, scenario in enumerate(scenarios):
                subset = metrics[(metrics.scenario == scenario) & (metrics.margin_us == margin)].set_index("model")
                values = [float(subset.loc[name, "fold_mean"]) for name in names]
                # Baseline uses the SAME equal-fold weighting as all model bars.
                frame = predictions[predictions.scenario == scenario]
                expected = []
                for _, fold in frame.groupby("fold"):
                    proportion = 0.0 if margin == 0 else float((np.abs(fold.y_true_PT - THRESHOLD_CUTOFF) <= margin).mean())
                    expected.append(.5 + .5 * proportion)
                values.append(float(np.mean(expected)))
                bars = ax.bar(np.arange(5) + (i-.5)*.4, np.array(values)*100, .4,
                              label="Non-Cross Subject" if i == 0 else "Cross Subject", color=colors[i],
                              edgecolor="white", linewidth=1.0)
                for j, (bar, value) in enumerate(zip(bars, values)):
                    ax.annotate(f"{value*100:.2f}",(bar.get_x()+bar.get_width()/2, value*100),
                                ha="center", va="bottom", xytext=(0,3),textcoords="offset points",fontsize=20)
                    paper_rows.append({"scenario":scenario,"model":names[j] if j<4 else "ExpectedUniformRandom",
                        "margin_us":margin,"accuracy_fold_mean":value,"display_percent":f"{value*100:.2f}",
                        "aggregation":"unweighted mean of five outer-fold scores"})
            ax.set_xticks(np.arange(5),labels,fontsize=25)
            ax.set_ylabel("Accuracy (%)",fontsize=34)
            ax.set_ylim(0,100);ax.tick_params(axis="y",labelsize=26,direction="in",length=4,width=1.8)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2));ax.tick_params(axis="y",which="minor",direction="in",length=2,width=1.6)
            ax.legend(loc="upper right",fontsize=24,frameon=True)
            fig.tight_layout();fig.savefig(output_dir/f"paper_model_comparison_fold_mean_margin{margin}.png",dpi=300);plt.close(fig)
        fig, ax = plt.subplots(figsize=(8,6))
        for i, scenario in enumerate(scenarios):
            frame = metrics[(metrics.scenario == scenario) & (metrics.model == "RandomForest")].sort_values("margin_us")
            ax.plot(frame.margin_us,frame.fold_mean*100,"o-",label="Non-Cross Subject" if i==0 else "Cross Subject",
                    color=["#f2b179","#7ecb8b"][i],linewidth=2)
            for _,row in frame.iterrows():
                ax.annotate(f"{row.fold_mean*100:.2f}",(row.margin_us,row.fold_mean*100),
                            textcoords="offset points",xytext=(0,7),ha="center",fontsize=13)
        ax.set_xlabel("Margin",fontsize=24);ax.set_ylabel("Accuracy (%)",fontsize=24)
        ax.set_xticks([0,5,10,15]);ax.tick_params(labelsize=16)
        ax.set_ylim(71,86);ax.grid(True,color="#e8e8e8",linewidth=.75)
        ax.legend(loc="upper left",fontsize=16,frameon=True)
        fig.tight_layout();fig.savefig(output_dir/"paper_margin_accuracy_fold_mean.png",dpi=300);plt.close(fig)
    pd.DataFrame(paper_rows).to_csv(output_dir/"paper_fold_mean_plotted_values.csv",index=False)


def run(output_dir):
    output_dir = Path(output_dir) / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_frames, feature_rows, fold_records = [], [], [], []
    for cross, outer_seed, inner_seed in ((False, 168, 5), (True, 93, 42)):
        scenario = "Cross-subject" if cross else "Non-cross-subject"
        folds = prepare_folds(is_cross=cross, cv_seed=outer_seed, inner_seed=inner_seed,
                              fast_tex_cands=None if cross else 35)
        model_names = list(downstream_models(cross)) + ["SexOnly", "Majority"]
        predictions = {name: np.zeros(len(y_bin), dtype=int) for name in model_names}
        scores = {name: np.zeros(len(y_bin), dtype=float) for name in model_names}
        continuous = {name: np.zeros(len(y_bin), dtype=float) for name in downstream_models(cross)}
        fold_ids = np.full(len(y_bin), -1, dtype=int)
        coverage = np.zeros(len(y_bin), dtype=int)
        for fold_idx, fd in enumerate(folds):
            tr, te = fd["tr_mask"], fd["te_mask"]
            if cross and set(subs[tr]) & set(subs[te]):
                raise ValueError("Cross-subject training/test overlap.")
            coverage[te] += 1
            fold_ids[te] = fold_idx + 1
            quota = fd["selected_quota"]
            selected = np.concatenate([fd["b_pool"][:quota[0]], fd["i_pool"][:quota[1]],
                                       fd["t_pool"][:quota[2]]])
            scaler = StandardScaler().fit(X_all[tr][:, selected])
            x_train = scaler.transform(X_all[tr][:, selected])
            x_test = scaler.transform(X_all[te][:, selected])
            for name, model in downstream_models(cross).items():
                model.fit(x_train, y_cont[tr])
                continuous[name][te] = model.predict(x_test)
                scores[name][te] = continuous[name][te] - fd["med_y"]
                predictions[name][te] = (scores[name][te] > 0).astype(int)
            sex = LogisticRegression().fit(gender[tr].reshape(-1, 1), y_bin[tr])
            predictions["SexOnly"][te] = sex.predict(gender[te].reshape(-1, 1))
            scores["SexOnly"][te] = sex.predict_proba(gender[te].reshape(-1, 1))[:, 1]
            dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((tr.sum(), 1)), y_bin[tr])
            predictions["Majority"][te] = dummy.predict(np.zeros((te.sum(), 1)))
            scores["Majority"][te] = y_bin[tr].mean()
            fold_records.append({"scenario": scenario, "fold": fold_idx + 1,
                "outer_seed": outer_seed, "inner_seed": inner_seed + fold_idx,
                "quota_Body_Imp_Tex": list(quota), "inner_correct_counts": fd["inner_correct_counts"],
                "prediction_cutoff_us": fd["med_y"], "train_indices": np.flatnonzero(tr).tolist(),
                "test_indices": np.flatnonzero(te).tolist()})
            for rank, index in enumerate(selected, 1):
                feature_rows.append({"scenario": scenario, "fold": fold_idx + 1, "rank_in_subset": rank,
                    "feature": all_feature_names[index], "storage_block": modalities[index],
                    "scientific_modality": feature_category(index), "VIP": fd["vip"][index]})
        if not np.all(coverage == 1):
            raise ValueError("Every sample must be tested exactly once.")
        oof = pd.DataFrame({"scenario": scenario, "sample_idx": np.arange(len(y_bin)), "subject_id": subs,
                            "fold": fold_ids, "y_true_PT": y_cont, "y_true_bin": y_bin})
        for name in model_names:
            oof[f"pred_{name}"] = predictions[name]
            oof[f"score_{name}"] = scores[name]
            if name in continuous:
                oof[f"predicted_PT_{name}"] = continuous[name]
            for margin in (0, 5, 10, 15):
                correct = predictions[name] == y_bin
                if margin > 0:
                    correct = correct | (np.abs(y_cont - THRESHOLD_CUTOFF) <= margin)
                fold_values = [float(correct[fold_ids == i].mean()) for i in range(1, 6)]
                metric_rows.append({"scenario": scenario, "model": name, "margin_us": margin,
                    "accuracy_pooled": float(correct.mean()), "correct": int(correct.sum()),
                    "n": len(y_bin), "fold_mean": float(np.mean(fold_values)),
                    "fold_sd_ddof0": float(np.std(fold_values)),
                    "strict_F1_positive": float(f1_score(y_bin, predictions[name])),
                    "strict_ROC_AUC_pooled": float(roc_auc_score(y_bin, scores[name]))})
        prediction_frames.append(oof)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(output_dir / "selected_features.csv", index=False)
    (output_dir / "folds.json").write_text(json.dumps(fold_records, indent=2), encoding="utf-8")
    baseline = []
    for margin in (0, 5, 10, 15):
        count = 0 if margin == 0 else int(np.sum(np.abs(y_cont - THRESHOLD_CUTOFF) <= margin))
        baseline.append({"margin_us": margin, "automatic_credit_samples": count,
                         "expected_random_score": .5 + .5 * count / len(y_bin)})
    pd.DataFrame(baseline).to_csv(output_dir / "random_baseline.csv", index=False)
    plot_results(metrics, predictions, output_dir)
    print("\nClassification (strict pooled accuracy):")
    print(metrics[metrics.margin_us == 0][["scenario", "model", "accuracy_pooled"]].to_string(index=False))
    return {"samples": len(y_bin), "subjects": len(unique_subs), "label_cutoff_us": 156,
        "sample_median_PT_us": float(np.median(y_cont)),
        "outer_seeds": {"noncross": 168, "cross": 93}, "inner_base_seeds": {"noncross": 5, "cross": 42},
        "model_parameters": {"cross": {name: model.get_params() for name, model in downstream_models(True).items()},
                             "noncross": {name: model.get_params() for name, model in downstream_models(False).items()}},
        "selection_blocks": {"Body": len(cols_b), "Imp": len(cols_i), "Tex": len(cols_t)},
        "notes": ["PLSRegression(n_components=2, scale=True) uses continuous PT; no PLS-DA.",
            "Scientific modalities differ from storage blocks: Body includes five finger-geometry measures.",
            "Regression predictions are dichotomised at the corresponding training partition median.",
            "Historical seed selection used outer scores; this release reproduces development results.",
            "Positive tolerance margins give automatic credit to any sample with true PT in the band.",
            "Historical de-correlation fallback can refill a quota without enforcing 0.85."]}
