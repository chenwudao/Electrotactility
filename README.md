# Electrotactility

电触觉刺激阈值预测的研究代码，用于相关论文中的数据处理、特征工程与机器学习建模。

仓库按数据、特征工程、建模和图表这一顺序组织。图表位于 `model/figures/`。

## 任务

本仓库包含两类任务：

- **二分类**：以阈值中位数为分割点，将样本划分为两类，并分别进行被试内验证与跨被试验证。脚本和文件名中的「非跨被试」即被试内设定。
- **回归**：直接预测连续阈值，比较被试内与跨被试两种设定。被试内回归包含带被试截距的方案；跨被试回归包含用静态特征预测截距，再与动态特征预测合并的方案。

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

目录名 `feature-enginnering` 沿用仓库中的现有拼写。

## 数据与特征

### `dataset/`

基础输入表。

- `X_人体成分_new.xlsx`：人体成分特征。
- `X_电阻抗.xlsx`：电阻抗特征。
- `X_纹理特征.xlsx`：图像纹理特征。
- `Y.xlsx`：标签表，包含阈值、受试者编号和手指信息。

### `feature-enginnering/linear-features/`

线性特征工程的结果。

- `experiment1.0-50features/`
  - 当前建模使用的主特征目录。
  - 关键文件：
    - `X_结合相关互相关_spe_0.9_rfr_50.xlsx`：50 维线性特征表。
    - `Y.xlsx`：同目录下的标签表。主建模脚本读取这一份。
- `experiment2.0-12&11features/`
  - 进一步筛选后的特征名单。
  - 当前包含：
    - `分类非跨被试12最优特征.txt`：被试内二分类的 12 个特征。
    - `分类跨被试11最优特征.txt`：跨被试二分类的 11 个特征。

### `feature-enginnering/nonlinear-features/`

非线性扩展特征及说明。`非线性特征说明.txt` 记录了由 12 个特征生成的平方项、立方项、交互项、对数变换、指数变换和分段线性项。

- `12特征_非线性扩展_完整.xlsx`
- `12特征_扩展特征_仅新增.xlsx`
- `非线性特征说明.txt`
- `回归20线性+非线性最优特征.txt`：回归所用的 20 维特征名单，与回归脚本中的特征一致。脚本从 50 维线性特征表构造该输入，并对其中若干列取对数。

### `feature-enginnering/scripts/`

历史特征工程脚本。路径通过 [`path_utils.py`](feature-enginnering/scripts/path_utils.py) 按项目根目录解析；该文件把旧文件名映射到当前文件名，以便兼容历史脚本。

重点脚本：

- [`path_utils.py`](feature-enginnering/scripts/path_utils.py)
  - 特征工程脚本的统一路径入口。
  - 解析 `dataset/`、`linear-features/`、`nonlinear-features/` 和脚本输出目录。
- `features_props*.py`
  - 相关性、互相关和重要性统计。
- `features_select*.py`
  - 特征筛选。
- [`bingji.py`](feature-enginnering/scripts/bingji.py)
  - 合并特征表。
- [`rfc_import_props.py`](feature-enginnering/scripts/rfc_import_props.py)
  - 分类任务的特征重要性分析。
- [`overfit_validation_50features.py`](feature-enginnering/scripts/overfit_validation_50features.py)
  - 50 维特征上的二分类过拟合检验。
- [`overfit_validation_nestedcv_binary.py`](feature-enginnering/scripts/overfit_validation_nestedcv_binary.py)
  - 嵌套交叉验证下的二分类稳定性分析。

说明：

- `scripts/` 中仍保留较多阶段性实验脚本，文件名对应各自的实验步骤。
- 复现当前主结果时，直接使用已经整理好的特征文件即可，不必重跑全部历史脚本。

## 模型

### `model/binary-class/`

二分类脚本。主入口使用随机森林，标签由阈值中位数分割得到。

核心入口：

- [`rf_binary_12features_within.py`](model/binary-class/rf_binary_12features_within.py)
  - 被试内二分类。
  - 从 50 维线性特征表中选取 12 个特征。
  - 读取：
    - `feature-enginnering/linear-features/experiment1.0-50features/X_结合相关互相关_spe_0.9_rfr_50.xlsx`
    - `feature-enginnering/linear-features/experiment1.0-50features/Y.xlsx`
- [`rf_binary_11features_cross.py`](model/binary-class/rf_binary_11features_cross.py)
  - 跨被试二分类，按受试者分组做交叉验证。
  - 使用 11 个特征，读取与上一脚本相同的特征表和标签表。

辅助脚本位于 `model/binary-class/scripts/`：

- `eval_feature_count_accuracy_*.py`
  - 比较不同特征数量下的分类性能。
- `eval_feature_count_margin15_*.py`
  - 在分类边界附近设置容忍区间（margin，单位 mA）后评估性能。
- `eval_feature_combo_accuracy.py`
  - 比较特征组合对性能的影响。
- `model_comparison_*.py`
  - 多模型对比。
- `plot_*.py`
  - 绘制结果图。

### `model/regression/`

回归脚本。

#### `non-cross-regression/`

- [`regression_structured_prior_model.py`](model/regression/non-cross-regression/regression_structured_prior_model.py)
  - 被试内回归主脚本之一。
  - 使用 20 维特征。
  - 包含结构化先验（全局模型加被试截距）、目标裁剪、预测平滑和多种基模型配置。
- [`intercept_model_comparison.py`](model/regression/non-cross-regression/intercept_model_comparison.py)
  - 被试内截距模型对比。

#### `cross-regression/`

- [`static_intercept_prediction_model.py`](model/regression/cross-regression/static_intercept_prediction_model.py)
  - 跨被试截距预测主脚本，按受试者划分训练集与测试集。
  - 先用动态特征预测阈值，再用被试级静态特征预测截距，然后将两部分合并。
- [`catboost_intercept_model.py`](model/regression/cross-regression/catboost_intercept_model.py)
  - CatBoost 截距预测。
- [`catboost_intercept_tuning.py`](model/regression/cross-regression/catboost_intercept_tuning.py)
  - CatBoost 截距模型的调参脚本。

#### `scripts/`

- `model_comparison_regression_cross.py`
- `model_comparison_regression_within.py`
- `plot_regression_model_comparison.py`

用于回归模型对比和图表输出。

### `model/figures/`

已生成的主要图表，例如：

- `confusion_matrices.png`
- `feature_combo.png`
- `feature_count_trend.png`
- `margin_accuracy_within_vs_cross.png`
- `model_comparison_within_vs_cross.png`
- `regression_model_comparison_r2.png`
- `regression_model_comparison_mae.png`
- `regression_model_comparison_mape.png`

## 主数据流

1. `dataset/` 提供基础特征表和标签表。
2. `feature-enginnering/scripts/` 用于生成或分析特征，其中多数脚本属于历史实验。
3. `feature-enginnering/linear-features/experiment1.0-50features/` 提供当前使用的 50 维线性特征表，以及主建模脚本读取的 `Y.xlsx`。
4. `feature-enginnering/nonlinear-features/` 保存非线性扩展结果，以及回归所用的 20 维特征名单。
5. `model/binary-class/` 与 `model/regression/` 基于上述特征建模、对比并输出结果。

当前主入口直接读取：

- `feature-enginnering/linear-features/experiment1.0-50features/X_结合相关互相关_spe_0.9_rfr_50.xlsx`
- `feature-enginnering/linear-features/experiment1.0-50features/Y.xlsx`

`dataset/Y.xlsx` 是基础标签表，`nonlinear-features/` 下的扩展表是特征工程产物。当前主入口不从这两处读取。

## 环境

依赖及固定版本见 [`requirements.txt`](requirements.txt)。

建议使用独立环境：

```bash
conda create -n electrotactility python=3.11
conda activate electrotactility
pip install -r requirements.txt
```

主要依赖：

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
- `xlrd`

## 运行

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

## 约定

- 脚本中的路径为相对路径，或由项目根目录推导。
- 特征工程脚本通过 `feature-enginnering/scripts/path_utils.py` 解析路径。
- `catboost_info/` 是 CatBoost 训练时生成的日志，不属于输入数据。
- `model/figures/` 及各子目录中的结果文件是实验输出，不是原始输入。
- 目录名 `feature-enginnering` 保持现有拼写。

## 后续整理

若继续整理仓库，可优先将 `feature-enginnering/scripts/` 按用途分组，例如：

- `correlation/`
- `selection/`
- `validation/`
- `archive/`

已退出主流程的脚本可以移入 `archive/`。复现实验时，从 `model/` 下的主入口开始，不必逐个重跑历史特征工程脚本。
