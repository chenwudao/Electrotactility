# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：overfit_validation_nestedcv_binary.py
统一使用项目内的相对路径。
"""

from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR
import sys
import itertools
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

COL_NUM = 3  # 特征从第 4 列开始
RANDOM_STATE = 83

def load_y_cont(data_dir: str) -> np.ndarray:
    tf = pd.read_excel(resolve_input_file("Y.xlsx"), index_col=0)
    return tf.iloc[:, 3].astype(float).to_numpy().ravel()

def load_X_big(data_dir: str) -> pd.DataFrame:
    """读取大特征集合，包含前 3 列元信息和后续特征列。"""
    path = resolve_input_file("X_结合相关互相关_spe_0.9_new.xlsx")
    return pd.read_excel(path, index_col=0)

def load_original_50_feature_names(data_dir: str) -> list[str]:
    """读取固定 50 特征表，用于和每折重新筛选结果对比。"""
    path = resolve_input_file("X_结合相关互相关_spe_0.9_new_rfr_50.xlsx")
    df = pd.read_excel(path, index_col=0)
    return df.columns.tolist()[COL_NUM:]

def discretize_by_train_median(y_train_cont: np.ndarray, y_cont_all: np.ndarray) -> tuple[np.ndarray, float]:
    thr = float(np.median(y_train_cont))
    y_bin_all = (np.asarray(y_cont_all, dtype=float) >= thr).astype(int)
    return y_bin_all, thr

def select_topk_features_by_rf_importance(X_train: np.ndarray, y_train: np.ndarray, k: int) -> tuple[list[int], np.ndarray]:
    """
    仅在训练折上拟合 RF，按重要性选 Top-K 特征，返回特征列索引（相对 X_train 的列）。
    同时返回 importances，便于做稳定性分析。

    这里的 RF 参数尽量复现你当前的筛选规则（见 features_props_jiehe_rfr.py）：
    RandomForestClassifier(random_state=610, n_estimators=92, max_depth=7, max_features=7, ...)
    """
    fs_model = RandomForestClassifier(
        random_state=610,
        criterion="gini",
        n_estimators=92,
        max_depth=7,
        max_features=7,
        min_samples_leaf=1,
        min_samples_split=2,
        n_jobs=-1,
    )
    fs_model.fit(X_train, y_train)
    importances = np.asarray(fs_model.feature_importances_, dtype=float)
    k = int(min(k, importances.shape[0]))
    top_idx = np.argsort(importances)[::-1][:k]
    return top_idx.tolist(), importances

def inner_tune_params(X_train: np.ndarray, y_train: np.ndarray, inner_splits: int = 3) -> dict:
    """
    在训练折上做一个轻量的内层 CV 调参（仅在已筛选的特征上）。
    以 AUC 为主指标（若某折 AUC 不可算则跳过该折）。
    """
    param_grid = [
        {"n_estimators": 200, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 10, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "max_features": 0.5, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 3},
    ]

    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)

    best_params = None
    best_score = -np.inf

    for params in param_grid:
        aucs = []
        for tr_i, va_i in inner_cv.split(X_train, y_train):
            X_tr, X_va = X_train[tr_i], X_train[va_i]
            y_tr, y_va = y_train[tr_i], y_train[va_i]

            model = RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_split=2,
                class_weight="balanced",
                **params,
            )
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:, 1]
            # 某些情况下验证折只有一个类别会导致 AUC 报错；这种折跳过
            try:
                aucs.append(float(roc_auc_score(y_va, proba)))
            except Exception:
                continue

        score = float(np.mean(aucs)) if len(aucs) > 0 else -np.inf
        if score > best_score:
            best_score = score
            best_params = params

    # 兜底
    return best_params or param_grid[0]

def main():
    print("数据目录:", LINEAR_WORK_DIR)
    y_cont = load_y_cont(LINEAR_WORK_DIR)
    df_X_big = load_X_big(LINEAR_WORK_DIR)
    X_big = df_X_big.iloc[:, COL_NUM:].astype(float).to_numpy()
    feat_names = df_X_big.columns.tolist()[COL_NUM:]
    orig50 = load_original_50_feature_names(LINEAR_WORK_DIR)
    orig50_set = set(orig50)

    n_samples, n_features = X_big.shape
    print("样本数:", n_samples, "大特征数:", n_features)
    print("原始固定50特征数:", len(orig50_set))

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    fold = 0

    # 为了外层 StratifiedKFold，需要先给一个初始二分类标签；这里用全体中位数，仅用于分层，不用于训练/评估阈值
    y_for_strat = (y_cont >= float(np.median(y_cont))).astype(int)

    selected_name_lists: list[list[str]] = []
    # 稳定性：统计每个特征被选中的次数
    select_counts = dict.fromkeys(feat_names, 0)

    for train_idx, test_idx in outer_cv.split(X_big, y_for_strat):
        fold += 1
        X_train_all, X_test_all = X_big[train_idx], X_big[test_idx]
        y_train_cont, y_test_cont = y_cont[train_idx], y_cont[test_idx]

        # 1) 阈值仅由训练折决定，并应用到 train/test
        y_bin_all, thr = discretize_by_train_median(y_train_cont, y_cont)
        y_train = y_bin_all[train_idx]
        y_test = y_bin_all[test_idx]

        pos_ratio_train = float(np.mean(y_train))
        pos_ratio_test = float(np.mean(y_test))

        # 2) 训练折内做特征筛选（Top-50）
        top50_idx, _importances = select_topk_features_by_rf_importance(X_train_all, y_train, k=50)
        X_train = X_train_all[:, top50_idx]
        X_test = X_test_all[:, top50_idx]
        selected_names = [feat_names[i] for i in top50_idx]
        selected_name_lists.append(selected_names)
        for n in selected_names:
            select_counts[n] += 1

        # 与原始固定50的重合情况
        sel_set = set(selected_names)
        inter = len(sel_set & orig50_set)
        union = len(sel_set | orig50_set)
        overlap_ratio = inter / 50.0
        jaccard = inter / union if union > 0 else 0.0

        # 3) 内层 CV 调参（在 X_train 上）
        best_params = inner_tune_params(X_train, y_train, inner_splits=3)

        # 4) 用最优参数在外层训练折训练，外层测试折评估
        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_split=2,
            class_weight="balanced",
            **best_params,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, pred))
        f1 = float(f1_score(y_test, pred))
        try:
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = float("nan")

        rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "thr(train_median)": thr,
                "pos_ratio_train": round(pos_ratio_train, 6),
                "pos_ratio_test": round(pos_ratio_test, 6),
                "acc": round(acc, 6),
                "f1": round(f1, 6),
                "auc": round(auc, 6) if not np.isnan(auc) else None,
                "selected_k": 50,
                "overlap_with_orig50": inter,
                "overlap_ratio_with_orig50": round(overlap_ratio, 6),
                "jaccard_with_orig50": round(jaccard, 6),
                "selected_features": ",".join(selected_names),
                "best_params": str(best_params),
            }
        )

        print(f"fold {fold}: acc={acc:.4f} f1={f1:.4f} auc={auc if not np.isnan(auc) else 'NA'} thr={thr:.3f}")

    df = pd.DataFrame(rows)

    # 5折间的稳定性：两两Jaccard（仅看“每折选出来的50”之间）
    pair_rows = []
    for (i, a), (j, b) in itertools.combinations(list(enumerate(selected_name_lists, start=1)), 2):
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        pair_rows.append(
            {
                "fold_i": i,
                "fold_j": j,
                "intersection": inter,
                "jaccard": inter / union if union > 0 else 0.0,
            }
        )
    df_pairs = pd.DataFrame(pair_rows)

    # 特征被选中频次（稳定性）
    df_freq = (
        pd.DataFrame({"feature": list(select_counts.keys()), "selected_count": list(select_counts.values())})
        .sort_values(["selected_count", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    df_freq["selected_ratio"] = df_freq["selected_count"] / float(df["fold"].nunique())

    summary = {
        "acc_mean": float(np.mean(df["acc"])),
        "acc_std": float(np.std(df["acc"])),
        "f1_mean": float(np.mean(df["f1"])),
        "f1_std": float(np.std(df["f1"])),
        "auc_mean": float(np.nanmean(pd.to_numeric(df["auc"], errors="coerce"))),
        "auc_std": float(np.nanstd(pd.to_numeric(df["auc"], errors="coerce"))),
        "overlap_with_orig50_mean": float(np.mean(df["overlap_with_orig50"])),
        "overlap_with_orig50_std": float(np.std(df["overlap_with_orig50"])),
        "overlap_ratio_with_orig50_mean": float(np.mean(df["overlap_ratio_with_orig50"])),
        "jaccard_with_orig50_mean": float(np.mean(df["jaccard_with_orig50"])),
        "fold_pairwise_jaccard_mean": float(np.mean(df_pairs["jaccard"])) if len(df_pairs) else float("nan"),
        "fold_pairwise_jaccard_std": float(np.std(df_pairs["jaccard"])) if len(df_pairs) else float("nan"),
        "n_samples": n_samples,
        "n_features_big": n_features,
        "k_selected": 50,
        "outer_folds": 5,
    }
    df_sum = pd.DataFrame([summary])

    out_xlsx = resolve_output_file("overfit_nestedcv_binary_result.xlsx", target="result")
    with pd.ExcelWriter(out_xlsx) as w:
        df.to_excel(w, index=False, sheet_name="outer_folds")
        df_sum.to_excel(w, index=False, sheet_name="summary")
        df_pairs.to_excel(w, index=False, sheet_name="stability_pairs")
        df_freq.to_excel(w, index=False, sheet_name="stability_freq")

    out_txt = resolve_output_file("overfit_nestedcv_binary_readme.txt", target="result")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("嵌套交叉验证（二分类）说明\n")
        f.write("=" * 60 + "\n")
        f.write("外层：5 折分层 CV（仅用于最终评估）\n")
        f.write("每个外层训练折内：\n")
        f.write("- 用训练折 y 中位数作为阈值离散化为 0/1（阈值不泄漏到测试折）\n")
        f.write("- 在训练折上做 RF 重要性 Top-50 特征筛选（避免用全数据筛特征造成乐观偏差）\n")
        f.write("- 在训练折上做 3 折内层 CV 轻量调参（在已筛选的 50 特征上）\n\n")
        f.write("输出：overfit_nestedcv_binary_result.xlsx（outer_folds + summary）\n")
        f.write("本版本额外输出：\n")
        f.write("- 每折选出的50与“原始固定50”的重合数/重合率/Jaccard\n")
        f.write("- 5折之间两两Jaccard（stability_pairs）\n")
        f.write("- 每个特征被选中的频次（stability_freq）\n\n")
        f.write("解读：\n")
        f.write("- 若与原始固定50重合率低、或5折之间Jaccard低，说明特征选择不稳定（更可能存在选择偏差/过拟合风险）。\n")
        f.write("- 若重合率高且Jaccard高，说明你那组50更稳定，更有力支持其可泛化。\n")

    print("结果已保存:", out_xlsx)
    print("说明已保存:", out_txt)
    print("summary:", summary)

if __name__ == "__main__":
    main()
