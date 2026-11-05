# Electrotactility: Classification and Regression Reproduction

This repository recomputes perception-threshold (PT) classification and regression
results from the supplied feature tables: **128 finger records
from 32 participants, with four fingers per participant**.

The release contains one entry point, two task modules and five input tables.
Ablation experiments, GEE association analysis, descriptive correlation figures,
parameter-search scripts and previously generated result files are outside this
release. The original implementation remains available in Git history.

**Regression preprocessing now uses training-fold-only scaling and target
clipping, with raw PT retained for every test score.** Fixed historical features,
model parameters and classification seeds are retained. The remaining evaluation
limitations below mean that this is not a new independent validation.

## Run

Use **Python 3.11** in an activated virtual or conda environment:

```bash
python -m pip install -r requirements.txt

# Both tasks
python reproduce.py

# Or either task separately
python reproduce.py --task classification
python reproduce.py --task regression
```

Results are written to `results/`. To choose another directory:

```bash
python reproduce.py --output-dir /path/to/results
```

No original waveform/image archive or previously generated result workbook is
needed. Paths to inputs are resolved relative to the code, not the shell's
working directory. CPU execution is supported; no GPU is required.

## Files

```text
reproduce.py                         # Single command-line entry point
reproduction/
  classification.py                 # Two classification settings and plots
  regression.py                     # Two regression settings and plots
dataset/
  Y.xlsx                            # PT, participant IDs and metadata
  X_人体成分_new.xlsx                # Body-composition/geometry storage block
  X_电阻抗.xlsx                      # Impedance and waveform storage block
  X_纹理特征.xlsx                    # Fingertip image storage block
  regression_features.xlsx          # Historical fixed regression feature table
requirements.txt
```

The regression feature table is an unchanged copy of the former
`feature-enginnering/.../X_结合相关互相关_spe_0.9_rfr_50.xlsx` input. Its supplied
row order matches `Y.xlsx`. These are pre-extracted tables; this release does not
recompute their extraction from raw recordings.

## Classification

Labels are `0` for PT <= 156 us and `1` for PT > 156 us (64/64 records). The
actual sample median is **157 us**. The non-cross-subject protocol uses sample
splits; the cross-subject protocol splits sorted participant IDs so that all four
fingers of a participant stay together.

The outer loop has five folds, with a three-fold inner loop selecting between
two storage-block quotas separately within each outer training set. Two-component
PLS regression uses **continuous PT**, rather than binary labels, to compute VIP.
Scaling, VIP fitting and quota evaluation use the corresponding training data.
The historical Pearson-correlation threshold is 0.85; the refill-to-quota fallback
is preserved. The Body/Imp/Tex blocks contain 50/61/1422 features. Body includes
five finger-geometry features that belong scientifically to local morphology;
storage-block quotas are not Global/Bioelectrical/Local modality quotas.

The four predictive models are RandomForestRegressor, ExtraTreesRegressor,
KNeighborsRegressor and SVR. Their continuous predictions are dichotomised using
the **training partition's PT median**, while true labels always use 156 us.
Fixed model settings are in `downstream_models()`. Sex-only logistic and
training-majority classifiers are also recomputed on the same splits.

| Setting | Outer seed | Inner base seed | RF strict pooled accuracy | RF margin=15 pooled score |
|---|---:|---:|---:|---:|
| Non-cross-subject | 168 | 5 | 74.21875% | 84.375% |
| Cross-subject | 93 | 42 | 72.65625% | 81.25% |

Each inner split uses `base seed + zero-based outer fold index`. Margin zero is
ordinary strict accuracy. For positive margins, a sample receives credit if its
binary prediction is correct **or** its true PT is within the tolerance band
around 156 us. This is a tolerance score, not a new raw prediction. The random
baseline is calculated from the actual band counts.

Outputs in `results/classification/` include metrics, both settings' sample-level
OOF predictions, per-fold selected features/VIP, fold indices/seeds/quotas, the
random baseline and three pooled-result figures. The confusion matrices use strict
predictions. Three additional paper figures (`paper_model_comparison_fold_mean_margin0.png`,
`paper_model_comparison_fold_mean_margin15.png`, `paper_margin_accuracy_fold_mean.png`)
use unweighted means of the five outer folds. Their theoretical random baselines
are also averaged within folds first; at margin 15 this gives 61.72% non-cross
and 61.67% cross, rather than applying the pooled 61.72% to both. Exact plotted
bar values are exported to `paper_fold_mean_plotted_values.csv`.

## Regression

The non-cross-subject protocol uses sample-wise five-fold CV with seed **53**;
the cross-subject protocol uses participant-grouped five-fold CV with seed **427**.
The fixed 20-dimensional representation contains 15 historical variables and five
log-transformed copies. No alpha or CatBoost parameter-grid search is run.

The enhanced RF models are named explicitly:

- Non-cross: RF plus partially pooled subject intercepts, with historical ridge
  alpha=1. Other training fingers of the same participant provide calibration.
- Cross: RF plus a CatBoost-predicted static subject intercept, with the historical
  fixed CatBoost parameters. Lambda is selected using training data.

Ordinary ExtraTrees, AdaBoost, CatBoost and training-mean baselines are also
recomputed. The enhanced RF result is not substituted into an ordinary RF row.

| Enhanced model | Mean fold R² | Mean fold MAE (us) | Mean fold MAPE |
|---|---:|---:|---:|
| Non-cross RF + subject intercept | 0.489183 | 42.103152 | 27.7840% |
| Cross RF + CatBoost intercept | 0.357845 | 48.119065 | 31.7492% |

Regression values are the arithmetic means of five fold metrics, **not pooled
OOF R²**. CSV MAPE values and all regression figures use fractions, matching
the manuscript (for example, 0.28 represents 28%). Exported fold
standard deviations use `ddof=0`.

Outputs in `results/regression/` include per-fold and mean metrics, all models'
OOF predictions, enhanced-model OOF predictions, method metadata, three separate
comparison figures and a three-panel figure suitable for manuscript Figure 16. `y_raw`, `y_evaluation` and `evaluation_target` make the target used
for every reported error explicit; all held-out records are scored on raw PT.

Dynamic MinMax scaling is fitted separately on each outer training fold and then
applied to that fold's train/test features. For non-cross enhanced RF, the 2%/98%
PT clipping limits are estimated from that outer training fold only and applied
only to its training labels. Other models keep raw training labels. Clipping
limits and scaler fitting rows are recorded in `metadata.json`.

Regression figures use the measured layout of image object 155 on page 42 of
`Revised Manuscript.pdf`, rather than the earlier DOCX graphic. Its original raster
is 1182 x 307 pixels and its PDF placement is 425.55 x 110.59 points
(15.01246 x 3.90137 cm). The combined high-resolution output is 3546 x 921 pixels
at approximately 600 DPI with physical-size metadata matching that placement;
a 1182 x 307 reference-size output is also supplied. Axes rectangles, thin strokes,
text sizes and a/b/c positions follow measured reference coordinates. Standalone
figures are crops of the same calibrated canvas. True updated values and raster
antialiasing mean that pixel identity is not claimed.

Install Times New Roman or set `ELECTROTACTILITY_FONT` to its `.ttf` file for the
same typeface. Missing Times New Roman produces an explicit fallback warning;
`plot_style.json` records the actual font, dimensions and reference geometry.
Fonts are not bundled. Display labels match the manuscript (`RandomForest` and
`Baseline`); these denote the enhanced RF models described above and the
training-mean baseline, respectively. Model identities must remain clear in the
manuscript caption.

Relative to commit `0e3d847`, the non-cross enhanced RF changes from
R²=0.494783 / MAE=41.127737 / MAPE=26.4182% to
R²=0.489183 / MAE=42.103152 / MAPE=27.7840%. Correcting MinMax scaling alone
did not change any prediction for this supplied dataset and the fixed settings.
Classification and the remaining regression model/setting results are unchanged.

## Remaining evaluation limitations

- The historical non-cross classification inner seed was chosen after inspecting
  outer accuracy. Reproducing that configuration does not make it an independent
  test. This release does not rerun that search.
- The regression feature list is inherited; its selection provenance is not
  encoded here. Enhanced-model parameters were historically chosen using the
  evaluation folds. Regression is **not nested hyperparameter evaluation**.
- The fixed regression log-feature construction is inherited. Its helper uses
  the full column minimum only if a nonpositive value occurs; all five supplied
  log-feature inputs are strictly positive, so that shift branch is not used
  for this dataset. A different dataset would need a training-only log policy.
- Cross intercept shrinkage uses training-subject LOO intercept predictions,
  with residual targets from an RF fitted on the outer training set; this is not
  LOO retraining of the complete two-stage model.
- Some waveform features were acquired after PT measurement. Use before a first
  PT measurement requires a separate assessment of feature availability.

The two preprocessing corrections do not remove historical selection bias or
establish clinical or causal validity. No new parameter search is performed.
