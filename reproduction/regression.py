"""Recompute Fig. 16 regression results with training-fold preprocessing.

This is a result-reproduction module, not a claim of unbiased nested evaluation.
All test scores use raw PT. Dynamic scaling and non-cross RF target clipping
are fitted only on each outer training fold; historical fixed model and feature
choices remain unchanged. See metadata.json for the remaining limitations.
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
    train: np.ndarray, test: np.ndarray, y_raw: np.ndarray,
    subjects: np.ndarray, prediction: np.ndarray,
    base_prediction: np.ndarray | None = None, **parameters,
) -> None:
    folds.append({
        "setting": setting, "model": model, "fold": fold,
        "n_train": len(train), "n_test": len(test), "evaluation_target": "raw_PT",
        **_scores(y_raw[test], prediction), **parameters,
    })
    for j, row in enumerate(test):
        oof.append({
            "setting": setting, "model": model, "fold": fold,
            "row": int(row), "subject": subjects[row],
            "y_raw": y_raw[row], "y_evaluation": y_raw[row],
            "prediction": prediction[j], "evaluation_target": "raw_PT",
            "base_prediction": np.nan if base_prediction is None else base_prediction[j],
        })


def _prepare_fold_inputs(dynamic: np.ndarray, subjects: np.ndarray, splits: list,
                         setting: str, audit: list) -> list:
    """Fit one dynamic scaler per training fold, shared by that fold's models."""
    prepared = []
    for fold, (train, test) in enumerate(splits, 1):
        if np.intersect1d(train, test).size:
            raise ValueError("Training and test rows overlap.")
        if setting == "cross_subject" and np.intersect1d(subjects[train], subjects[test]).size:
            raise ValueError("Cross-subject training and test subjects overlap.")
        scaler = MinMaxScaler().fit(dynamic[train])
        audit.append({
            "setting": setting, "fold": fold,
            "train_rows": train.tolist(), "test_rows": test.tolist(),
            "train_subjects": np.unique(subjects[train]).tolist(),
            "test_subjects": np.unique(subjects[test]).tolist(),
            "dynamic_scaler_fit_rows": train.tolist(),
            "dynamic_scaler_scope": "outer_train_only",
            "dynamic_scaler_data_min": scaler.data_min_.tolist(),
            "dynamic_scaler_data_max": scaler.data_max_.tolist(),
        })
        prepared.append((train, test, scaler.transform(dynamic[train]), scaler.transform(dynamic[test])))
    return prepared


def _within_enhanced(
    y_raw: np.ndarray, subjects: np.ndarray, prepared: list,
    folds: list, oof: list, target_audit: list,
) -> None:
    for fold, (train, test, X_train, X_test) in enumerate(prepared, 1):
        limits = np.quantile(y_raw[train], [0.02, 0.98])
        y_train = np.clip(y_raw[train], *limits)
        target_audit.append({
            "setting": "within_subject", "model": WITHIN_MODEL, "fold": fold,
            "quantiles": [0.02, 0.98], "clipping_limits_us": limits.tolist(),
            "clipping_fit_rows": train.tolist(),
            "clipped_training_rows": train[~np.isclose(y_train, y_raw[train])].tolist(),
            "changed_test_evaluation_rows": [], "evaluation_target": "raw_PT",
        })
        base = RandomForestRegressor(**RF_WITHIN).fit(X_train, y_train)
        base_test = base.predict(X_test)
        residuals = y_train - base.predict(X_train)
        unique = np.unique(subjects[train])
        design = (subjects[train, None] == unique[None, :]).astype(float)
        ridge = Ridge(alpha=SUBJECT_RIDGE_ALPHA, fit_intercept=True, random_state=42)
        ridge.fit(design, residuals)
        offsets = dict(zip(unique, ridge.intercept_ + ridge.coef_))
        prediction = base_test + np.array([offsets.get(s, 0.0) for s in subjects[test]])
        _record(
            folds, oof, "within_subject", WITHIN_MODEL, fold, train, test,
            y_raw, subjects, prediction, base_test,
            ridge_alpha=SUBJECT_RIDGE_ALPHA, clipping_lower=float(limits[0]),
            clipping_upper=float(limits[1]),
        )


def _fit_intercept(X: np.ndarray, alpha: np.ndarray) -> CatBoostRegressor:
    model = CatBoostRegressor(
        verbose=False, allow_writing_files=False, random_state=42, thread_count=1,
        **INTERCEPT_CATBOOST,
    )
    return model.fit(X, alpha)


def _cross_enhanced(
    static: np.ndarray, y: np.ndarray, subjects: np.ndarray,
    prepared: list, folds: list, oof: list,
) -> None:
    for fold, (train, test, X_train, X_test) in enumerate(prepared, 1):
        train_subjects = np.unique(subjects[train])
        scaler = StandardScaler().fit(static[train])
        static_scaled = scaler.transform(static)
        subject_features = {
            s: np.mean(static_scaled[subjects == s], axis=0) for s in np.unique(subjects)
        }
        base = RandomForestRegressor(**RF_CROSS).fit(X_train, y[train])
        base_train = base.predict(X_train)
        base_test = base.predict(X_test)
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
            y, subjects, prediction, base_test, lambda_used=best_lambda,
        )
        print(f"Regression cross-subject fold {fold}/{N_SPLITS}: lambda={best_lambda}", flush=True)


def _comparison_models() -> dict:
    return {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(
            iterations=500, learning_rate=0.1, depth=6, loss_function="RMSE",
            random_seed=42, verbose=False, allow_writing_files=False, thread_count=1,
        ),
    }


def _comparisons(y: np.ndarray, subjects: np.ndarray, prepared: list,
                 setting: str, folds: list, oof: list) -> None:
    for name, model in _comparison_models().items():
        for fold, (train, test, X_train, X_test) in enumerate(prepared, 1):
            model.fit(X_train, y[train])
            _record(
                folds, oof, setting, name, fold, train, test, y, subjects,
                model.predict(X_test),
            )
    for fold, (train, test, _, _) in enumerate(prepared, 1):
        prediction = np.full(len(test), float(np.mean(y[train])))
        _record(
            folds, oof, setting, "TrainingMean", fold, train, test, y,
            subjects, prediction,
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
    """Render corrected metrics using the manuscript Fig.16 visual style."""
    import os
    import warnings

    from matplotlib import font_manager
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    font_families = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    font_override = os.environ.get("ELECTROTACTILITY_FONT")
    if font_override:
        font_file = Path(font_override).expanduser().resolve(strict=True)
        font_manager.fontManager.addfont(str(font_file))
        font_families.insert(0, font_manager.FontProperties(fname=font_file).get_name())
    else:
        # Register newly installed fonts even when Matplotlib's font cache predates them.
        try:
            font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
        except ValueError:
            for candidate in font_manager.findSystemFonts():
                try:
                    if font_manager.FontProperties(fname=candidate).get_name() == "Times New Roman":
                        font_manager.fontManager.addfont(candidate)
                except (OSError, RuntimeError, ValueError):
                    # An unrelated unreadable system font must not prevent plotting.
                    continue
    font_path = font_manager.findfont(font_manager.FontProperties(family=font_families))
    actual_font = font_manager.FontProperties(fname=font_path).get_name()
    if actual_font != "Times New Roman":
        warnings.warn(
            f"Times New Roman is unavailable; Fig.16 uses {actual_font}. "
            "Install Times New Roman or set ELECTROTACTILITY_FONT to its font file for the manuscript typeface.",
            UserWarning, stacklevel=2,
        )
    # Geometry measured from image object 155 on page 42 of the revised PDF.
    # Work in the original 1182 x 307 pixel coordinate system, not the old DOCX.
    from matplotlib.transforms import Bbox
    reference_width, reference_height, reference_dpi = 1182, 307, 200
    reference_placement_pt = (425.55, 110.59)
    frame_pixels = (
        (39.0960451977, 45.1588235294, 378.0287356322, 281.6420454545),
        (431.2588235294, 44.7426900585, 773.0914285714, 281.2022471910),
        (836.5454545455, 47.0175438596, 1175.1314285714, 283.6497175141),
    )
    labels = ["RandomForest", "ExtraTrees", "AdaBoost", "CatBoost", "Baseline"]
    orders = {
        "within_subject": [WITHIN_MODEL, "ExtraTrees", "AdaBoost", "CatBoost", "TrainingMean"],
        "cross_subject": [CROSS_MODEL, "ExtraTrees", "AdaBoost", "CatBoost", "TrainingMean"],
    }
    colors = ["#f7c58a", "#9ad49b"]
    panels = (("r2", "R2"), ("mae", "MAE"), ("mape", "MAPE"))
    style = {
        "font.family": "serif", "font.serif": font_families,
        "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
        "text.color": "black", "axes.facecolor": "white", "figure.facecolor": "white",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(reference_width / reference_dpi, reference_height / reference_dpi))
        for index, ((metric, title), rect) in enumerate(zip(panels, frame_pixels)):
            left, top, right, bottom = (coordinate + .5 for coordinate in rect)
            ax = fig.add_axes([left / reference_width, 1 - bottom / reference_height,
                               (right - left) / reference_width, (bottom - top) / reference_height])
            values_by_setting = []
            width = 0.36
            for offset, (setting, names) in enumerate(orders.items()):
                data = metrics[metrics["setting"] == setting].set_index("model").loc[names]
                values = data[f"{metric}_mean"].to_numpy(dtype=float)
                values_by_setting.append(values)
                ax.bar(np.arange(5) + (offset - 0.5) * width, values, width,
                       label="Non-Cross Subject" if offset == 0 else "Cross Subject",
                       color=colors[offset], linewidth=0)
            within_values, cross_values = values_by_setting
            if metric == "r2":
                limits = (min(-0.05, float(cross_values.min()) - 0.05),
                          max(0.55, float(within_values.max()) + 0.05))
            else:
                limits = (0, float(max(within_values.max(), cross_values.max()))
                          + (12 if metric == "mae" else 0.08))
            ax.set_ylim(*limits)
            for offset, values in enumerate(values_by_setting):
                for j, value in enumerate(values):
                    ax.text(j + (offset - .5) * width, value + .01 * (limits[1] - limits[0]),
                            f"{value:.2f}", ha="center", va="bottom", fontsize=4.25)
            ax.set_xticks(np.arange(5), labels)
            ax.tick_params(axis="x", labelsize=4.4, length=0, pad=1.9)
            ax.tick_params(axis="y", labelsize=4.7, pad=0.8)
            ax.yaxis.set_major_locator(MultipleLocator(10 if metric == "mae" else 0.1))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis="y", which="major", direction="in", length=1.0, width=.30)
            ax.tick_params(axis="y", which="minor", direction="in", length=.48, width=.245)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color("black"); spine.set_linewidth(.21)
                spine.set_snap(False)
            # The measured original title baselines are 43/42/45 pixels.
            title_baseline = (43, 42, 45)[index]
            fig.text((left + right) / (2 * reference_width), 1 - title_baseline / reference_height,
                     title, ha="center", va="baseline", fontsize=6.0)
            ylabel_x, ylabel_y = ((13.5, 163.5), (408.5, 163.5), (810.5, 165.5))[index]
            fig.text(ylabel_x / reference_width, 1 - ylabel_y / reference_height,
                     title, rotation=90, ha="center", va="center", fontsize=6.0)
            # Match the original panel-letter ink locations, including the larger b.
            from matplotlib.textpath import TextPath
            letter = "abc"[index]
            letter_box = TextPath((0, 0), letter, size=12.0,
                                  prop=font_manager.FontProperties(fname=font_path)).get_extents()
            ink_left = (4, 396, 797)[index]
            letter_x = ink_left - letter_box.x0 * reference_dpi / 72
            letter_y = 41 + letter_box.y0 * reference_dpi / 72
            fig.text(letter_x / reference_width, 1 - letter_y / reference_height,
                     letter, ha="left", va="baseline", fontsize=12.0)
            legend_left, legend_top = ((246, 50), (438, 50), (843, 53))[index]
            legend = ax.legend(loc="upper left", bbox_to_anchor=(legend_left / reference_width,
                                1 - legend_top / reference_height), bbox_transform=fig.transFigure,
                               borderaxespad=0, frameon=True, fontsize=4.25,
                               handlelength=2.0, handletextpad=.4, borderpad=.45,
                               labelspacing=.5, fancybox=True)
            legend.get_frame().set_linewidth(.25)

        # 3x the PDF raster resolution; physical dimensions match its PDF placement.
        combined_size = (reference_width * 3, reference_height * 3)
        physical_dpi = tuple(pixels * 72 / points for pixels, points in
                             zip(combined_size, reference_placement_pt))
        fig.savefig(output / "regression_comparison_reference200.png", dpi=200)
        fig.savefig(output / "regression_comparison_combined.png", dpi=600,
                    pil_kwargs={"dpi": physical_dpi})
        # Standalone panels retain the combined figure's measured typography/geometry.
        for (metric, _), (left, right) in zip(panels, ((0, 385), (394, 780), (796, 1182))):
            box = Bbox.from_extents(left / reference_dpi, 0,
                                   right / reference_dpi, reference_height / reference_dpi)
            fig.savefig(output / f"regression_comparison_{metric}.png", dpi=600,
                        bbox_inches=box, pad_inches=0)
        plt.close(fig)

    (output / "plot_style.json").write_text(json.dumps({
        "reference": "Revised Manuscript.pdf page 42, image object 155",
        "requested_font": "Times New Roman", "actual_font": actual_font,
        "font_file": font_path, "font_fallback_used": actual_font != "Times New Roman",
        "font_override": font_override, "colors": colors,
        "mape_unit_all_figures": "fraction", "export_dpi_nominal": 600,
        "reference_raster_size_pixels": [reference_width, reference_height],
        "reference_placement_points": list(reference_placement_pt),
        "reference_placement_cm": [p / 72 * 2.54 for p in reference_placement_pt],
        "frame_coordinates_in_reference_pixels": frame_pixels,
        "combined_size_pixels": list(combined_size), "combined_physical_dpi": physical_dpi,
        "style_note": "Axes positions, thin strokes, title/letter sizes calibrated to revised PDF; raster antialiasing can differ.",
        "display_labels": labels,
        "random_forest_display_label": {"within_subject": WITHIN_MODEL, "cross_subject": CROSS_MODEL},
        "baseline_display_label": "TrainingMean",
        "source_values": "Five-fold unweighted means from metrics.csv; no fixed performance values.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metadata(preprocessing_audit: list, target_audit: list) -> dict:
    return {
        "purpose": "Recompute regression with train-only preprocessing and raw test PT, without new parameter searches.",
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
            "mape_unit": "Fraction in CSV and every regression figure, matching manuscript Fig.16.",
            "evaluation_target_all_models": "raw_PT",
            "dynamic_scaling": "MinMaxScaler fitted separately on each outer training fold; reused by all models in that fold.",
        },
        "historical_parameters": {
            "within_rf": RF_WITHIN, "cross_rf": RF_CROSS,
            "within_subject_ridge_alpha": SUBJECT_RIDGE_ALPHA,
            "cross_intercept_catboost": INTERCEPT_CATBOOST,
            "cross_lambda_candidates_training_only": LAMBDA_CANDIDATES,
        },
        "within_enhanced_target": {
            "evaluation_target": "raw_PT",
            "training_quantiles": [0.02, 0.98],
            "clipping_rule": "Estimate limits from each outer training PT only; clip only training labels. All test labels remain raw.",
            "folds": target_audit,
        },
        "fold_preprocessing": preprocessing_audit,
        "limitations": [
            "Regression evaluation is not nested. Alpha=1 and fixed CatBoost intercept parameters were historically chosen using the same evaluation folds.",
            "The fixed 20-feature construction is inherited; this module does not reselect features inside folds.",
            "The fixed log feature construction is unchanged. _safe_log still defines a full-column minimum shift if non-positive values occur; all five logged columns in the supplied data are strictly positive, so no shift is applied here.",
            "Cross static StandardScaler is fitted on each outer training set; subject feature means use each subject's available feature rows.",
            "Cross lambda uses training-subject LOO intercept predictions, but residual targets come from the RF fitted in-sample on the same outer training set.",
            "Non-cross subject intercept calibration requires training PT observations from the same subjects; unavailable subject intercepts default to zero.",
            "Training-only scaling and target clipping fix the two preprocessing issues; they do not remove historical model-selection bias or validate pre-measurement clinical use.",
        ],
    }


def run(output_dir: Path) -> dict:
    """Run both regression settings and save metrics, OOF predictions and plots."""
    output = Path(output_dir) / "regression"
    output.mkdir(parents=True, exist_ok=True)
    dynamic, static, y, subjects = _load_data()
    preprocessing_audit, target_audit = [], []
    within = _prepare_fold_inputs(
        dynamic, subjects, _splits(subjects, cross=False), "within_subject", preprocessing_audit,
    )
    cross = _prepare_fold_inputs(
        dynamic, subjects, _splits(subjects, cross=True), "cross_subject", preprocessing_audit,
    )
    fold_rows, oof_rows = [], []
    _within_enhanced(y, subjects, within, fold_rows, oof_rows, target_audit)
    _cross_enhanced(static, y, subjects, cross, fold_rows, oof_rows)
    _comparisons(y, subjects, within, "within_subject", fold_rows, oof_rows)
    _comparisons(y, subjects, cross, "cross_subject", fold_rows, oof_rows)
    folds = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows).sort_values(["setting", "model", "row"])
    metrics = _summary(folds)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    oof[oof["model"].isin([WITHIN_MODEL, CROSS_MODEL])].to_csv(
        output / "enhanced_oof_predictions.csv", index=False,
    )
    metadata = _metadata(preprocessing_audit, target_audit)
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plots(metrics, output)
    print(metrics.to_string(index=False), flush=True)
    return {"output_dir": str(output.resolve()), "metrics": metrics.to_dict(orient="records")}
