# Electrotactility: Multimodal Electrotactile Perception Threshold Prediction

This repository contains the official data, feature matrices, modeling pipelines, statistical analyses, and figure-generation scripts for **multimodal electrotactile perception threshold (PT) prediction** across **global physiological**, **bioelectrical**, and **local finger morphological/epidermal texture** modalities ($N = 128$ finger-level records from $32$ participants $\times$ $4$ fingers per participant).

The repository supports two primary predictive tasks and one covariate-adjusted statistical association analysis:

1. **Binary PT Classification (Primary Task)**: Classifies samples into low-threshold ($\text{PT} \le 156.0\ \mu\text{s}$) and high-threshold ($\text{PT} > 156.0\ \mu\text{s}$) categories using a strictly leak-free **Two-Layer Nested Cross-Validation (`Outer 5-Fold CV` + `Inner 3-Fold PLS-VIP Feature Screening`)** pipeline under both **Non-Cross-Subject** and **Cross-Subject** protocols.
2. **Continuous PT Regression**: Predicts continuous perception thresholds ($\mu\text{s}$) under non-cross-subject structured prior modeling and cross-subject static intercept calibration.
3. **Participant-Clustered GEE Association Analysis**: Evaluates univariate and covariate-adjusted (`Sex` and `Sex + Height`) feature–PT associations using **Generalized Estimating Equations (GEE)** clustered by participant ID with Huber–White sandwich standard errors, $95\%$ confidence intervals, and Benjamini–Hochberg false discovery rate (BH-FDR) correction.

---

## 📦 Large Raw Experimental Dataset (Google Drive)

Because the raw multimodal experimental recordings (`5_实验数据/`, approximately **2.7 GB** in total) exceed GitHub's file size limits, the complete raw dataset is hosted on **Google Drive**:

- 🔗 **Google Drive Raw Dataset (`5_实验数据/`, ~2.7 GB)**:  
  **[👉 Download Raw Experimental Dataset from Google Drive](https://drive.google.com/file/d/1VmnDQZH3BVMRZHEvCR3dyZh4Ty7tWfPA/view?usp=drive_link)**

### Contents of `5_实验数据/` (~2.7 GB):
```text
5_实验数据/ (Place in the repository root if needed)
├─ 图像数据/                # High-resolution optical finger images & segmentation masks (32 participants × 4 fingers)
├─ 电化学阻抗谱/            # Raw electrochemical impedance spectroscopy (EIS) frequency sweeps
├─ 电流电压波形/            # Raw pre-/post-stimulation transient current and voltage waveforms
├─ inbody770_data_all.xlsx  # Raw whole-body composition exports from InBody 770
└─ 指围信息_all.xlsx        # Raw manual finger circumference, length, and geometry measurements
```

> **💡 Out-of-the-Box Reproducibility**: The `dataset/` directory (`4.8 MB`) included directly in this GitHub repository already contains the complete pre-extracted three-modality feature matrices ($128$ samples $\times$ $1,468$ features) and target labels. **You do NOT need to download the 2.7 GB raw image/waveform archive to reproduce all feature screening, nested cross-validation, GEE statistical tables, and manuscript figures.**

---

## 📂 Repository Structure

```text
Electrotactility/
├─ dataset/                                           # Pre-extracted multimodal feature matrices & labels (4.8 MB)
│  ├─ X_人体成分_new.xlsx                              # Global physiological & body composition features
│  ├─ X_电阻抗.xlsx                                    # Bioelectrical impedance & transient voltage features
│  ├─ X_纹理特征.xlsx                                  # Local finger geometry & epidermal texture descriptors
│  └─ Y.xlsx                                          # Target PT labels, participant IDs, sex, and finger indices
├─ revision_experiments/                              # Nested CV, GEE analysis & manuscript figures
│  ├─ run_all_unified_revision_closure.py             # Master script: Nested CV classification, regression & GEE tables
│  ├─ generate_manuscript_style_figures.py            # Master script: Generates manuscript Fig. 9 – Fig. 15 (300 DPI)
│  └─ results/                                        # Generated Excel/CSV tables, GEE reports & 300 DPI figures
│     ├─ gee_clustered_associations.xlsx              # GEE results (unadjusted, Sex-adj, Sex+Height-adj, 95% CI, FDR q)
│     ├─ gee_fig13_selected_features.xlsx             # GEE results for the features selected in Inner Folds 1–3
│     ├─ unified_classification_metrics.xlsx          # Nested CV classification accuracy, F1, ROC-AUC & McNemar tests
│     ├─ unified_regression_metrics.xlsx              # Continuous PT regression metrics (R², MAE, MAPE)
│     ├─ consistent_confusion_matrices.xlsx           # Raw unmodified 2×2 confusion matrices & tolerance band breakdown
│     ├─ margin_band_sample_statistics.xlsx           # Sample statistics within the [141, 171] μs (±15 μs) tolerance band
│     ├─ inner_3fold_plsda_selected_features.csv      # Feature lists & VIP scores selected in Inner Folds 1–3
│     ├─ out_of_fold_predictions.csv                  # Sample-level out-of-fold predictions across all evaluated models
│     └─ figures/                                     # Output directory for Fig. 9 – Fig. 15
├─ model/                                             # Task-specific modeling results & figure outputs
│  ├─ binary-class/results/                           # Binary classification CSV/PNG summaries
│  ├─ regression/                                     # Continuous PT regression scripts & model comparison results
│  │  ├─ non-cross-regression/                        # Non-cross-subject structured prior regression
│  │  ├─ cross-regression/                            # Cross-subject CatBoost static intercept calibration model
│  │  └─ scripts/                                     # Regression benchmark & plotting scripts
│  └─ figures/                                        # Synchronized manuscript figures (Fig. 9 – Fig. 18)
├─ feature-enginnering/                               # Reference 50-feature candidate pool table
│  └─ linear-features/experiment1.0-50features/
├─ requirements.txt                                   # Pinned Python dependencies
└─ README.md
```

---

## 🔬 Methodology Overview

### 1. Leak-Free Two-Layer Nested Cross-Validation (`Outer 5-Fold` + `Inner 3-Fold`)
All preprocessing, scaling, feature screening, and hyperparameter selection steps are performed strictly inside the training partitions:
- **Outer 5-Fold Cross-Validation**:
  - **Cross-Subject Setting**: Participant-grouped 5-fold split ($25\text{–}26$ participants for training, $6\text{–}7$ participants for testing per fold; all 4 fingers of any participant remain strictly in the same fold).
  - **Non-Cross-Subject Setting**: Sample-level 5-fold split across the 128 records.
- **Inner 3-Fold PLS-VIP Feature Screening**:
  - Inside each outer training fold, a 3-fold inner split evaluates candidate modality quotas (correlation threshold fixed at $|r| < 0.85$):
    1. **Within-Fold Standardization**: all features are z-scored using the training partition only;
    2. **2-Component PLS Regression on Continuous PT & VIP Ranking**: a PLS regression model (continuous PT as the response) is fitted on the training partition and Variable Importance in Projection (VIP) scores are computed;
    3. **Modality-Wise Quota Selection with Collinearity Deduplication ($|r| < 0.85$)**: within each modality (body composition, bioelectrical, image texture), features are ranked by VIP and greedily retained while filtering out redundant features with pairwise correlation $\ge 0.85$; the modality quota is chosen by inner-fold validation accuracy.
- **Downstream Classifier Evaluation**:
  - Using the compact multimodal feature subset locked by the inner CV ($k = 11$ for cross-subject; $k = 12$ for non-cross-subject), four representative classifiers (`RandomForest`, `ExtraTrees`, `KNN`, `SVM`) and demographic baselines (`Sex`, `Sex + Height`) are trained on the outer training fold and evaluated on the held-out outer test fold.

### 2. Participant-Clustered GEE Association Analysis
- Continuous PT ($\mu\text{s}$) is modeled as the outcome in Gaussian GEE models with an exchangeable working correlation structure clustered on participant ID ($32$ clusters $\times$ $4$ fingers) and Huber–White robust sandwich standard errors.
- Separate models are fitted for each standardized feature ($\mu\text{s/SD}$): (i) unadjusted, (ii) adjusted for `Sex`, and (iii) adjusted for `Sex + Height`, with $95\%$ confidence intervals and BH-FDR corrected $q$-values.

---

## 🚀 Quick-Start Reproduction Guide

### 1. Environment Setup

```bash
conda create -n electrotactility python=3.11 -y
conda activate electrotactility
pip install -r requirements.txt
```

### 2. Reproduce All Unified Nested CV Results, GEE Tables, and Manuscript Figures

Run the two master scripts from the repository root:

```bash
# Step 1: Reproduce all Nested CV classification/regression metrics, confusion matrices, and GEE Excel tables (~15s)
python revision_experiments/run_all_unified_revision_closure.py

# Step 2: Reproduce all manuscript figures Fig. 9 – Fig. 15 at 300 DPI (~25s)
python revision_experiments/generate_manuscript_style_figures.py
```

### 3. Reproduce Continuous Regression Experiments & Figures

```bash
# Non-cross-subject structured prior regression
python model/regression/non-cross-regression/regression_structured_prior_model.py

# Cross-subject static intercept calibration regression
python model/regression/cross-regression/catboost_intercept_model.py

# Regression model comparison benchmarks & plots (R², MAE, MAPE)
python model/regression/scripts/model_comparison_regression_within.py
python model/regression/scripts/model_comparison_regression_cross.py
python model/regression/scripts/plot_regression_model_comparison.py
```

---

## 📊 Summary of Generated Manuscript Figures (`Fig. 9` – `Fig. 15`)

All figures are saved at **300 DPI** in both `model/figures/` and `revision_experiments/results/figures/`:

| Figure | File Name (`model/figures/`) | Description |
| :--- | :--- | :--- |
| **Fig. 9** | `fig9_spearman_50features.png`<br>`fig9_spearman_50features_nogrid.png` | Spearman rank correlation ($\rho$) bar chart of the 50 candidate features grouped by modality (clean grid-free background). |
| **Fig. 10** | `fig10_corr_matrix_50features.png` | $50 \times 50$ pairwise Spearman correlation matrix heatmap across the three modalities. |
| **Fig. 11** | `model_comparison_within_vs_cross_margin0.png`<br>`model_comparison_within_vs_cross_margin15.png` | Nested CV classification accuracy comparison across `RandomForest`, `ExtraTrees`, `KNN`, `SVM`, and `Baseline`. |
| **Fig. 12** | `margin_accuracy_within_vs_cross.png` | Classification accuracy across tolerance margins ($\text{Margin} = 0, 5, 10, 15\ \mu\text{s}$). |
| **Fig. 13** | `fig13_inner_3fold_plsda.png` (`feature_count_trend.png`)<br>`fig13_inner_3fold_plsda_nogrid.png` | Three-column VIP bar chart of compact multimodal feature subsets selected in Inner Folds 1, 2, and 3 (plus `feature_count_trend_curve_backup.png`). |
| **Fig. 14** | `feature_combo_margin0.png`<br>`feature_combo_margin15.png` (`feature_combo.png`) | Ablation comparison of demographic baselines (`Sex`, `Sex+Height`) against all 7 single-, dual-, and tri-modality combinations. |
| **Fig. 15** | `confusion_matrices_margin0.png` (`confusion_matrices.png`)<br>`confusion_matrices_margin15.png` | Raw unmodified $2 \times 2$ confusion matrices for (a) Non-Cross-Subject and (b) Cross-Subject Random Forest classifiers. |
