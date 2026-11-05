"""Recompute the historical Fig. 16 regression results from the supplied data.

This is a result-reproduction module, not a claim of unbiased nested evaluation.
The historical full-cohort scaling, within-subject target winsorization and
previously selected parameters are retained explicitly; see metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
WITHIN_SEED = 53
CROSS_SEED = 427
N_SPLITS = 5
WITHIN_MODEL = "RF+partially_pooled_subject_intercept"
CROSS_MODEL = "RF+CatBoost_predicted_subject_intercept"
RF_WITHIN = {
    "criterion": "poisson", "max_depth": 7, "max_features": 2,
    "min_samples_leaf": 1, "min_samples_split": 3,
    "n_estimators": 22, "random_state": 5532311,
}
RF_CROSS = {
    "criterion": "friedman_mse", "max_depth": 5, "max_features": 3,
    "min_samples_leaf": 1, "min_samples_split": 2,
    "n_estimators": 7, "random_state": 1435,
}
# These are historical choices, not parameters selected afresh by this module.
SUBJECT_RIDGE_ALPHA = 1.0
INTERCEPT_CATBOOST = {
    "iterations": 700, "depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 3.0,
}
LAMBDA_CANDIDATES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
ORIGINAL_FEATURES = [
    "v1-v2", "LOG_R2", "指围-指长", "45. FFM% of Right Arm",
    "199. Measured Circumference of Neck", "206. Measured Circumference of Left Thigh",
    "GLCM_t_0_d_2_SVD15_465", "STHOS_r_2_a_45_LRHGLE_22",
    "LBP_r3_pt_17_342", "LBP_r3_pt_18_343", "ZER_r3_pt_4_80", "ZER_r3_pt_19_95",
]
INITIAL_LOG_FEATURES = [
    "45. FFM% of Right Arm", "199. Measured Circumference of Neck",
]
ADDITIONAL_FEATURES = [
    "ZER_r3_pt_15_91", "GLCM_t_0_d_4_SVD14_524", "STHOS_r_2_a_135_LRLGLE_43",
]
NEW_LOG_FEATURES = [
    "206. Measured Circumference of Left Thigh", "STHOS_r_2_a_45_LRHGLE_22", "LOG_R2",
]
STATIC_FEATURES = [
    "199. Measured Circumference of Neck", "206. Measured Circumference of Left Thigh",
    "45. FFM% of Right Arm", "指围-指长",
]


def _safe_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    minimum = np.min(values)
    return np.log(values - minimum + 1.0) if minimum <= 0 else np.log(values)


def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = pd.read_excel(ROOT / "dataset" / "regression_features.xlsx")
    targets = pd.read_excel(ROOT / "dataset" / "Y.xlsx", index_col=0)
    if len(features) != len(targets):
        raise ValueError("Feature and target rows differ; row order must match the supplied files.")
    target_column = "阈值" if "阈值" in targets.columns else targets.columns[3]
    y = targets[target_column].to_numpy(dtype=float)
    subjects = targets["受试者编号"].to_numpy()
    dynamic = np.hstack([
        features[ORIGINAL_FEATURES].to_numpy(),
        np.column_stack([_safe_log(features[f].values) for f in INITIAL_LOG_FEATURES]),
        features[ADDITIONAL_FEATURES].to_numpy(),
        np.column_stack([_safe_log(features[f].values) for f in NEW_LOG_FEATURES]),
    ])
    static = features[STATIC_FEATURES].to_numpy(dtype=float)
    if dynamic.shape != (128, 20) or len(np.unique(subjects)) != 32:
        raise ValueError("Expected the supplied 128 records, 20 derived features and 32 subjects.")
    if not all(np.isfinite(a).all() for a in (dynamic, static, y)):
        raise ValueError("Regression inputs contain non-finite values.")
    return dynamic, static, y, subjects


def _splits(subjects: np.ndarray, cross: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    seed = CROSS_SEED if cross else WITHIN_SEED
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    if not cross:
        return list(cv.split(subjects))
    # Numeric subject IDs have the same order here as the original list(set(ids)).
    unique = np.unique(subjects)
    return [
        (np.flatnonzero(np.isin(subjects, unique[train])),
         np.flatnonzero(np.isin(subjects, unique[test])))
        for train, test in cv.split(unique)
    ]


def _scores(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
        # MAPE is stored as a fraction; the plot converts it to percent.
        "mape": float(np.mean(np.abs((y - prediction) / np.maximum(np.abs(y), 1e-8)))),
    }


def _record(
    folds: list, oof: list, setting: str, model: str, fold: int,
    train: np.ndarray, test: np.ndarray, y_raw: np.ndarray, y_evaluation: np.ndarray,
    subjects: np.ndarray, prediction: np.ndarray, target: str,
    base_prediction: np.ndarray | None = None, **parameters,
) -> None:
    folds.append({
        "setting": setting, "model": model, "fold": fold,
        "n_train": len(train), "n_test": len(test), "evaluation_target": target,
        **_scores(y_evaluation[test], prediction), **parameters,
    })
    for j, row in enumerate(test):
        oof.append({
            "setting": setting, "model": model, "fold": fold,
            "row": int(row), "subject": subjects[row],
            "y_raw": y_raw[row], "y_evaluation": y_evaluation[row],
            "prediction": prediction[j], "evaluation_target": target,
            "base_prediction": np.nan if base_prediction is None else base_prediction[j],
        })


def _within_enhanced(
    X: np.ndarray, y_raw: np.ndarray, subjects: np.ndarray,
    splits: list, folds: list, oof: list,
) -> np.ndarray:
    # Historical evaluation clips the entire target, including held-out labels.
    # Keep this behavior to reproduce the reported value, and label it explicitly.
    limits = np.quantile(y_raw, [0.02, 0.98])
    y = np.clip(y_raw, *limits)
    for fold, (train, test) in enumerate(splits, 1):
        base = RandomForestRegressor(**RF_WITHIN).fit(X[train], y[train])
        base_test = base.predict(X[test])
        residuals = y[train] - base.predict(X[train])
        unique = np.unique(subjects[train])
        design = (subjects[train, None] == unique[None, :]).astype(float)
        ridge = Ridge(alpha=SUBJECT_RIDGE_ALPHA, fit_intercept=True, random_state=42)
        ridge.fit(design, residuals)
        offsets = dict(zip(unique, ridge.intercept_ + ridge.coef_))
        prediction = base_test + np.array([offsets.get(s, 0.0) for s in subjects[test]])
        _record(
            folds, oof, "within_subject", WITHIN_MODEL, fold, train, test,
            y_raw, y, subjects, prediction, "winsorized_PT", base_test,
            ridge_alpha=SUBJECT_RIDGE_ALPHA,
        )
    return limits


def _fit_intercept(X: np.ndarray, alpha: np.ndarray) -> CatBoostRegressor:
    model = CatBoostRegressor(
        verbose=False, allow_writing_files=False, random_state=42,
        **INTERCEPT_CATBOOST,
    )
    return model.fit(X, alpha)


def _cross_enhanced(
    X: np.ndarray, static: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    splits: list, folds: list, oof: list,
) -> None:
    for fold, (train, test) in enumerate(splits, 1):
        train_subjects = np.unique(subjects[train])
        scaler = StandardScaler().fit(static[train])
        static_scaled = scaler.transform(static)
        subject_features = {
            s: np.mean(static_scaled[subjects == s], axis=0) for s in np.unique(subjects)
        }
        base = RandomForestRegressor(**RF_CROSS).fit(X[train], y[train])
        base_train = base.predict(X[train])
        base_test = base.predict(X[test])
        residuals = y[train] - base_train
        alphas = np.array([np.mean(residuals[subjects[train] == s]) for s in train_subjects])
        features = np.array([subject_features[s] for s in train_subjects])
        intercept_model = _fit_intercept(features, alphas)
        # Historical lambda choice uses leave-one-subject-out intercept predictions
        # within the outer training set. The RF residuals remain fitted in-sample.
        loo = {}
        for j, subject in enumerate(train_subjects):
            keep = np.arange(len(train_subjects)) != j
            loo[subject] = float(_fit_intercept(features[keep], alphas[keep]).predict(features[j:j + 1])[0])
        alpha_train = np.array([loo[s] for s in subjects[train]])
        errors = [mean_squared_error(y[train], base_train + lam * alpha_train) for lam in LAMBDA_CANDIDATES]
        best_lambda = LAMBDA_CANDIDATES[int(np.argmin(errors))]
        # Predict each subject in the same row-wise way as the historical script.
        alpha_test = np.array([
            intercept_model.predict(subject_features[s].reshape(1, -1))[0]
            for s in subjects[test]
        ])
        prediction = base_test + best_lambda * alpha_test
        _record(
            folds, oof, "cross_subject", CROSS_MODEL, fold, train, test,
            y, y, subjects, prediction, "raw_PT", base_test, lambda_used=best_lambda,
        )
        print(f"Regression cross-subject fold {fold}/{N_SPLITS}: lambda={best_lambda}", flush=True)


def _comparison_models() -> dict:
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(
            iterations=500, learning_rate=0.1, depth=6, loss_function="RMSE",
            random_seed=42, verbose=False, allow_writing_files=False,
        ),
    }


def _comparisons(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, splits: list,
                 setting: str, folds: list, oof: list) -> None:
    for name, model in _comparison_models().items():
        for fold, (train, test) in enumerate(splits, 1):
            model.fit(X[train], y[train])
            _record(
                folds, oof, setting, name, fold, train, test, y, y, subjects,
                model.predict(X[test]), "raw_PT",
            )
    for fold, (train, test) in enumerate(splits, 1):
        prediction = np.full(len(test), float(np.mean(y[train])))
        _record(
            folds, oof, setting, "TrainingMean", fold, train, test, y, y,
            subjects, prediction, "raw_PT",
        )


def _summary(folds: pd.DataFrame) -> pd.DataFrame:
    keys = ["setting", "model", "evaluation_target"]
    rows = []
    for values, data in folds.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        row["n_folds"] = len(data)
        for metric in ("r2", "mae", "mape"):
            row[f"{metric}_mean"] = float(data[metric].mean())
            row[f"{metric}_std"] = float(data[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _plots(metrics: pd.DataFrame, output: Path) -> None:
    labels = ["RF + subject\nintercept*", "ExtraTrees", "AdaBoost", "CatBoost", "Training mean"]
    orders = {
        "within_subject": [WITHIN_MODEL, "ExtraTrees", "AdaBoost", "CatBoost", "TrainingMean"],
        "cross_subject": [CROSS_MODEL, "ExtraTrees", "AdaBoost", "CatBoost", "TrainingMean"],
    }
    for metric, ylabel, multiplier in (("r2", "R²", 1), ("mae", "MAE (μs)", 1), ("mape", "MAPE (%)", 100)):
        fig, ax = plt.subplots(figsize=(10, 6.3))
        for offset, (setting, names) in enumerate(orders.items()):
            data = metrics[metrics["setting"] == setting].set_index("model").loc[names]
            values = data[f"{metric}_mean"].to_numpy() * multiplier
            bars = ax.bar(
                np.arange(5) + (offset - 0.5) * 0.37, values, 0.37,
                label="Non-cross subject" if offset == 0 else "Cross subject",
                color="#f3bf83" if offset == 0 else "#91cba0",
            )
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_xticks(np.arange(5), labels)
        ax.set_ylabel(ylabel)
        ax.set_title("Historical regression result reproduction")
        ax.legend(frameon=False)
        ax.margins(y=0.17)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        fig.text(
            0.07, 0.015,
            "* Non-cross: RF + partially pooled subject intercept; cross: RF + CatBoost-predicted intercept.\n"
            "Evaluation target differs: non-cross enhanced RF uses winsorized PT; other bars use raw PT.\n"
            "Historical parameters/preprocessing are retained; these bars are not a corrected unbiased comparison.",
            fontsize=8, va="bottom",
        )
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(output / f"regression_comparison_{metric}.png", dpi=200)
        plt.close(fig)


def _metadata(y: np.ndarray, limits: np.ndarray) -> dict:
    return {
        "purpose": "Recompute historical Fig.16 regression values, without new parameter searches.",
        "feature_file": "dataset/regression_features.xlsx",
        "target_file": "dataset/Y.xlsx",
        "row_alignment": "Supplied feature/target row order; zero-based row IDs in OOF output.",
        "feature_count": 20,
        "features": {
            "original": ORIGINAL_FEATURES, "initial_log": INITIAL_LOG_FEATURES,
            "additional": ADDITIONAL_FEATURES, "new_log": NEW_LOG_FEATURES,
            "subject_intercept_static": STATIC_FEATURES,
        },
        "evaluation": {
            "folds": N_SPLITS, "within_outer_seed": WITHIN_SEED, "cross_outer_seed": CROSS_SEED,
            "cross_split": "Subject-grouped KFold on sorted unique numeric IDs.",
            "aggregate": "Unweighted mean of five fold metrics; std uses ddof=0.",
            "mape_unit": "Fraction in CSV; percent in the MAPE figure.",
        },
        "historical_parameters": {
            "within_rf": RF_WITHIN, "cross_rf": RF_CROSS,
            "within_subject_ridge_alpha": SUBJECT_RIDGE_ALPHA,
            "cross_intercept_catboost": INTERCEPT_CATBOOST,
            "cross_lambda_candidates_training_only": LAMBDA_CANDIDATES,
        },
        "within_enhanced_target": {
            "evaluation_target": "winsorized_PT",
            "quantiles_full_cohort": [0.02, 0.98],
            "clipping_limits_us": limits.tolist(),
            "changed_label_count": int(np.sum(~np.isclose(y, np.clip(y, *limits)))),
            "other_models_evaluation_target": "raw_PT",
        },
        "limitations": [
            "Regression evaluation is not nested. Alpha=1 and fixed CatBoost intercept parameters were historically chosen using the same evaluation folds.",
            "The fixed 20-feature construction is inherited; this module does not reselect features inside folds.",
            "Dynamic MinMax scaling and log-shift minima use the full cohort, as in the historical implementation.",
            "Non-cross enhanced RF trains and scores on full-cohort winsorized PT, including changed held-out labels; its target differs from all other bars.",
            "Cross static StandardScaler is fitted on each outer training set; subject feature means use each subject's available feature rows.",
            "Cross lambda uses training-subject LOO intercept predictions, but residual targets come from the RF fitted in-sample on the same outer training set.",
            "Non-cross subject intercept calibration requires training PT observations from the same subjects; unavailable subject intercepts default to zero.",
            "Packaging simplification preserves historical values; it does not resolve methodological bias or validate new-subject or pre-measurement clinical use.",
        ],
    }


def run(output_dir: Path) -> dict:
    """Run both regression settings and save metrics, OOF predictions and plots."""
    output = Path(output_dir) / "regression"
    output.mkdir(parents=True, exist_ok=True)
    dynamic, static, y, subjects = _load_data()
    # Full-cohort dynamic scaling is intentionally retained for historical parity.
    X = MinMaxScaler().fit_transform(dynamic)
    within = _splits(subjects, cross=False)
    cross = _splits(subjects, cross=True)
    fold_rows, oof_rows = [], []
    limits = _within_enhanced(X, y, subjects, within, fold_rows, oof_rows)
    _cross_enhanced(X, static, y, subjects, cross, fold_rows, oof_rows)
    _comparisons(X, y, subjects, within, "within_subject", fold_rows, oof_rows)
    _comparisons(X, y, subjects, cross, "cross_subject", fold_rows, oof_rows)
    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows).sort_values(["setting", "model", "row"])
    metrics = _summary(folds)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    oof[oof["model"].isin([WITHIN_MODEL, CROSS_MODEL])].to_csv(
        output / "enhanced_oof_predictions.csv", index=False,
    )
    metadata = _metadata(y, limits)
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plots(metrics, output)
    print(metrics.to_string(index=False), flush=True)
    return {"output_dir": str(output.resolve()), "metrics": metrics.to_dict(orient="records")}
