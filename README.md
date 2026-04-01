# Electrotactility

电刺激阈值预测项目，当前仓库已经整理为「数据集 -> 特征工程 -> 建模 -> 图表输出」四层结构。

本项目主要包含两类任务：

- 二分类任务：基于阈值中位数将样本划分为两类，分别做被试内与跨被试验证。
- 回归任务：直接预测连续阈值，同时比较被试内、跨被试和带截距建模的方案。

## 目录结构

```text
Electrotactility/
├─ dataset/
│  ├─ X_人体成分_new.xlsx
│  ├─ X_电阻抗.xlsx
│  ├─ X_纹理特征.xlsx
│  └─ Y.xlsx
├─ feature-enginnering/
│  ├─ linear-features/
│  │  ├─ experiment1.0-50features/
│  │  └─ experiment2.0-12&11features/
│  ├─ nonlinear-features/
│  └─ scripts/
├─ model/
│  ├─ binary-class/
│  ├─ regression/
│  └─ figures/
├─ catboost_info/
├─ requirements.txt
└─ README.md
```

## 数据与特征

### `dataset/`

原始或基础整理后的输入数据。

- `X_人体成分_new.xlsx`：人体成分相关特征。
- `X_电阻抗.xlsx`：电阻抗相关特征。
- `X_纹理特征.xlsx`：图像纹理相关特征。
- `Y.xlsx`：标签表，包含阈值、受试者编号、手指信息等。

### `feature-enginnering/linear-features/`

线性特征工程结果。

- `experiment1.0-50features/`
  - 当前建模主输入目录。
  - 关键文件：
    - `X_结合相关互相关_spe_0.9_rfr_50.xlsx`
    - `Y.xlsx`
- `experiment2.0-12&11features/`
  - 用于保存进一步筛选后的说明文件。
  - 当前包含：
    - `分类非跨被试12最优特征.txt`
    - `分类跨被试11最优特征.txt`

### `feature-enginnering/nonlinear-features/`

非线性扩展特征及说明。

- `12特征_非线性扩展_完整.xlsx`
- `12特征_扩展特征_仅新增.xlsx`
- `非线性特征说明.txt`
- `回归20线性+非线性最优特征.txt`

### `feature-enginnering/scripts/`

历史特征工程脚本集中目录。当前已经统一改为项目内相对路径。

重点脚本：

- [`path_utils.py`](/d:/Environment/research/EF/Electrotactility/feature-enginnering/scripts/path_utils.py)
  - 特征工程脚本的统一路径入口。
  - 负责解析 `dataset/`、`linear-features/`、`nonlinear-features/` 和脚本输出目录。
  - 内置旧文件名到当前真实文件名的映射，便于兼容历史脚本。
- `features_props*.py`
  - 各类相关性、互相关、重要性统计脚本。
- `features_select*.py`
  - 特征筛选脚本。
- `bingji.py`
  - 特征表合并脚本。
- `rfc_import_props.py`
  - 分类任务特征重要性分析。
- `overfit_validation_50features.py`
  - 50 特征二分类过拟合验证。
- `overfit_validation_nestedcv_binary.py`
  - 嵌套交叉验证版本的二分类稳定性分析。

说明：

- `scripts/` 下仍保留较多历史实验脚本，命名具有明显阶段性。
- 当前主建模流程主要依赖已经整理好的特征文件，不要求从头重新运行全部历史脚本。

## 模型目录

### `model/binary-class/`

二分类任务脚本。

核心入口：

- [`rf_binary_12features_within.py`](/d:/Environment/research/EF/Electrotactility/model/binary-class/rf_binary_12features_within.py)
  - 被试内二分类主脚本。
  - 使用 12 个最优特征。
  - 输入：
    - `feature-enginnering/linear-features/experiment1.0-50features/X_结合相关互相关_spe_0.9_rfr_50.xlsx`
    - `feature-enginnering/nonlinear-features/12特征_非线性扩展_完整.xlsx`
- [`rf_binary_11features_cross.py`](/d:/Environment/research/EF/Electrotactility/model/binary-class/rf_binary_11features_cross.py)
  - 跨被试二分类主脚本。
  - 使用 11 个最优特征。

辅助脚本位于 `model/binary-class/scripts/`：

- `eval_feature_count_accuracy_*.py`
  - 不同特征数下的性能评估。
- `eval_feature_count_margin15_*.py`
  - 带容忍度的性能评估。
- `eval_feature_combo_accuracy.py`
  - 特征组合对性能的影响分析。
- `model_comparison_*.py`
  - 多模型对比。
- `plot_*.py`
  - 图表绘制脚本。

### `model/regression/`

回归任务脚本。

#### `non-cross-regression/`

- [`regression_structured_prior_model.py`](/d:/Environment/research/EF/Electrotactility/model/regression/non-cross-regression/regression_structured_prior_model.py)
  - 被试内回归主脚本之一。
  - 使用 20 维特征集。
  - 包含结构化先验、目标裁剪、平滑与多种基模型配置。
- [`intercept_model_comparison.py`](/d:/Environment/research/EF/Electrotactility/model/regression/non-cross-regression/intercept_model_comparison.py)
  - 被试内截距模型对比。

#### `cross-regression/`

- [`static_intercept_prediction_model.py`](/d:/Environment/research/EF/Electrotactility/model/regression/cross-regression/static_intercept_prediction_model.py)
  - 跨被试截距预测主脚本。
  - 核心思想：将动态特征预测与静态特征截距预测结合，缩小被试内与跨被试性能差距。
- [`catboost_intercept_model.py`](/d:/Environment/research/EF/Electrotactility/model/regression/cross-regression/catboost_intercept_model.py)
  - CatBoost 截距预测方案。
- [`catboost_intercept_tuning.py`](/d:/Environment/research/EF/Electrotactility/model/regression/cross-regression/catboost_intercept_tuning.py)
  - CatBoost 截距模型调参脚本。

#### `scripts/`

- `model_comparison_regression_cross.py`
- `model_comparison_regression_within.py`
- `plot_regression_model_comparison.py`

用于回归模型对比和图表输出。

### `model/figures/`

已生成的主要图表输出目录，例如：

- `confusion_matrices.png`
- `feature_combo.png`
- `feature_count_trend.png`
- `margin_accuracy_within_vs_cross.png`
- `model_comparison_within_vs_cross.png`
- `regression_model_comparison_r2.png`
- `regression_model_comparison_mae.png`
- `regression_model_comparison_mape.png`

## 当前主数据流

推荐按下面的理解方式阅读项目：

1. `dataset/` 提供基础数据表与标签表。
2. `feature-enginnering/scripts/` 用于生成或分析特征。
3. `feature-enginnering/linear-features/experiment1.0-50features/`
   - 提供当前最核心的 50 维线性特征表。
4. `feature-enginnering/nonlinear-features/`
   - 提供非线性扩展特征。
5. `model/binary-class/` 和 `model/regression/`
   - 基于上述特征直接建模、对比并输出结果。

换句话说，当前仓库最稳定的建模输入是：

- `dataset/Y.xlsx`
- `feature-enginnering/linear-features/experiment1.0-50features/X_结合相关互相关_spe_0.9_rfr_50.xlsx`
- `feature-enginnering/nonlinear-features/12特征_非线性扩展_完整.xlsx`

## 环境安装

项目依赖见 [`requirements.txt`](/d:/Environment/research/EF/Electrotactility/requirements.txt)。

建议使用单独环境：

```bash
conda create -n electrotactility python=3.11
conda activate electrotactility
pip install -r requirements.txt
```

主要依赖包括：

- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `lightgbm`
- `catboost`
- `openpyxl`

## 运行建议

### 二分类

被试内：

```bash
python model/binary-class/rf_binary_12features_within.py
```

跨被试：

```bash
python model/binary-class/rf_binary_11features_cross.py
```

### 回归

被试内结构化先验回归：

```bash
python model/regression/non-cross-regression/regression_structured_prior_model.py
```

跨被试静态截距预测：

```bash
python model/regression/cross-regression/static_intercept_prediction_model.py
```

## 说明与约定

- 项目中的路径已经统一改为相对路径或基于项目根目录推导的路径。
- 特征工程脚本优先通过 `feature-enginnering/scripts/path_utils.py` 做路径解析。
- `catboost_info/` 是 CatBoost 运行时生成目录，可视为训练过程产物。
- `model/figures/` 和各子目录下的结果文件属于实验输出，不是原始输入数据。
- 目录名 `feature-enginnering` 保留了当前仓库中的现有拼写。

## 后续维护建议

- 如果后续继续整理，可优先给 `feature-enginnering/scripts/` 做二次分组，例如拆成：
  - `correlation/`
  - `selection/`
  - `validation/`
  - `archive/`
- 如果某些脚本已经完全退出主流程，可以单独建 `archive/` 目录归档。
- 如果要复现实验，建议优先从 `model/` 下的主入口脚本开始，而不是从所有历史特征工程脚本逐个重跑。
