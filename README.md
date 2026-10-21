# Electrotactility

电触觉感知阈值（Perception Threshold, PT）多模态预测项目。仓库整理为「原始数据与特征表 -> 特征工程 -> 双层嵌套交叉验证与建模 -> 论文图表与文档输出」结构。

本项目包含两类核心任务：

- **二分类任务（主任务）**：基于训练折感知阈值中位数（`156.0 μs`）将样本划分为高/低阈值两类，采用严格的 **双层嵌套交叉验证（Outer 5-Fold + Inner 3-Fold Multi-Block PLS-DA VIP）** 进行非跨被试（Non-Cross Subject）与跨被试（Cross-Subject）评估。
- **回归任务**：直接预测连续感知阈值，比较非跨被试、跨被试与静态截距校准方案。

---

## 📦 大型原始数据集（Google Drive 下载）

由于原始多模态实验采集数据（`5_实验数据/`，总大小约 **2.7 GB**）超出 GitHub 大文件存储限制，完整原始数据集已托管至 **Google Drive**：

- 🔗 **Google Drive 原始数据集下载链接（`5_实验数据/`, 2.7 GB）**：  
  **[👉 点击前往 Google Drive 下载原始实验数据集 (5_实验数据)](https://drive.google.com/drive/folders/YOUR_GOOGLE_DRIVE_FOLDER_ID)** *(上传完成后将此处链接替换为您的 Google Drive 分享链接)*

### `5_实验数据/` 包含的原始数据内容：
```text
5_实验数据/ (约 2.7 GB，从 Google Drive 下载后放置于项目根目录即可)
├─ 图像数据/                # 32 名受试者手指高清光学图像与局部形态分割原始图
├─ 电化学阻抗谱/            # 原始电化学阻抗谱 (EIS) 频谱测量数据
├─ 电流电压波形/            # 电刺激前后原始电流与电压暂态波形数据
├─ inbody770_data_all.xlsx  # InBody 770 全身人体成分分析仪原始导出总表
└─ 指围信息_all.xlsx        # 受试者各手指几何尺寸与指围原始测量表
```

> **💡 快速复现提示**：仓库内的 `dataset/` 目录（仅 `4.8 MB`）已直接内置从上述原始数据中提取完毕的三模态完整特征矩阵（共 128 条样本 × 1,468 维原始特征）与阈值标签表。**无需下载 2.7 GB 原始图像/波形文件，克隆本仓库后即可直接运行所有特征筛选、嵌套交叉验证与论文图表生成脚本！**

---

## 📂 目录结构

```text
Electrotactility/
├─ dataset/                                 # 已提取的三模态完整特征表与标签（随仓库直接提供）
│  ├─ X_人体成分_new.xlsx                    # 全身生理与人体成分特征 (31 维)
│  ├─ X_电阻抗.xlsx                          # 生物电阻抗与暂态电压特征 (15 维)
│  ├─ X_纹理特征.xlsx                        # 手指局部几何形态与皮肤纹理特征 (1,422 维)
│  └─ Y.xlsx                                # 标签表（感知阈值、受试者编号、性别、手指编号等）
├─ revision_experiments/                    # 【最新】论文返修统一双层嵌套交叉验证与作图流水线
│  ├─ run_all_unified_revision_closure.py   # 外五折 + 内三折 Multi-Block PLS-DA 特征筛选与全套实验主脚本
│  ├─ generate_manuscript_style_figures.py  # 生成论文最终版 Fig. 9 – Fig. 15 全套 300 DPI 图表
│  └─ results/                              # 统一协议导出的 CSV/Excel 统计表与高清图表
├─ docs/                                    # 论文图表生成说明、方法学通俗解读与审稿回复文档
│  ├─ Final_Figures_and_Nested_CV_Methodology_Guide.docx
│  ├─ First_Author_Handover_and_Revision_Guide.docx
│  └─ Point_by_Point_Response_to_Reviewers.docx
├─ feature-enginnering/                     # 特征工程与早期筛选结果
│  ├─ linear-features/
│  │  ├─ experiment1.0-50features/
│  │  └─ experiment2.0-12&11features/
│  ├─ nonlinear-features/
│  └─ scripts/
├─ model/                                   # 二分类、回归建模脚本与论文图表目录
│  ├─ binary-class/
│  ├─ regression/
│  └─ figures/                              # 论文图表同步输出目录 (Fig. 9 – Fig. 15)
├─ requirements.txt
└─ README.md
```

---

## 🔬 最新核心实验方法（双层嵌套交叉验证 + Multi-Block PLS-DA）

在最新返修版本中，全项目所有二分类实验与图表（`Fig. 9` – `Fig. 15`）均统一遵循**零数据泄漏双层嵌套交叉验证协议**：

1. **外层 5 折交叉验证（Outer 5-Fold CV）**：
   - **Cross-Subject（跨被试）**：使用 `GroupKFold(n_splits=5)` 按 32 名受试者分组（每折 25 人 / 100 指训练，7 人 / 28 指盲测，同一受试者的 4 根手指绝不跨折）。
   - **Non-Cross Subject（非跨被试）**：使用 `StratifiedKFold(n_splits=5)` 在样本级划分训练与盲测折。
2. **内层 3 折 Multi-Block PLS-DA VIP 特征筛选（Inner 3-Fold CV）**：
   - 仅在外层训练折内部划分 3 个内层折（17 人训练 / 8 人验证），依次执行：
     1. **Z-Score 标准化**；
     2. **多块等方差缩放（Multi-Block Equal-Variance Scaling）**：对全身生理（$p_1=31$）、生物电阻抗（$p_2=15$）、局部形态纹理（$p_3=1422$）分别按 $1/\sqrt{p_b}$ 缩放，使三大模态先验方差权重严格为 $1:1:1$；
     3. **有监督 PLS-DA 潜变量投影（$C=2$）与 VIP（Variable Importance in Projection）计算**；
     4. **共线性去重（$|\rho| < 0.85$）与内层投票确定最优模态配额 $k^*$**（跨被试 $k^*=10\sim 11$，非跨被试 $k^*=12$）。
3. **外层多类机器学习算法全面评测与择优展示**：
   - 外层训练折按内层选定的配额锁定唯一特征集，原封不动输入十余种主流机器学习分类器（涵盖 `RandomForest`、`ExtraTrees`、`XGBoost`、`LightGBM`、`CatBoost`、`AdaBoost`、`GBDT`、`KNN`、`SVM`、`LogisticRegression` 等）进行盲测评估。
   - 最终挑选各流派表现最优的 4 个代表模型展示于 `Fig. 11`，并以全场准确率最高的 **`RandomForest`**（零容差 `Margin=0` 下 **74.22% / 72.66%**；`Margin=15 μs` 下 **84.38% / 81.25%**）作为主分类器完成容差灵敏度分析（`Fig. 12`）、包含 `Sex` 与 `Sex+Height` 基线的 9 组多模态消融实验（`Fig. 14`）及无后处理真实混淆矩阵（`Fig. 15`）。

---

## 🚀 快速运行指南

### 1. 环境安装

项目依赖见 [`requirements.txt`](./requirements.txt)：

```bash
conda create -n electrotactility python=3.11
conda activate electrotactility
pip install -r requirements.txt
```

### 2. 一键复现最新双层嵌套交叉验证全套结果与论文图表（推荐）

```bash
# 1. 运行双层嵌套交叉验证、PLS-DA 筛选、性别/身高混杂检验与多模态消融实验
python revision_experiments/run_all_unified_revision_closure.py

# 2. 一键生成论文最终版全套图表 (Fig. 9 - Fig. 15，自动同步至 model/figures/ 与 revision_experiments/results/figures/)
python revision_experiments/generate_manuscript_style_figures.py
```

### 3. 运行单独建模脚本

#### 二分类独立脚本
```bash
# 非跨被试二分类
python model/binary-class/rf_binary_12features_within.py

# 跨被试二分类
python model/binary-class/rf_binary_11features_cross.py
```

#### 回归任务脚本
```bash
# 非跨被试结构化先验回归
python model/regression/non-cross-regression/regression_structured_prior_model.py

# 跨被试静态截距预测
python model/regression/cross-regression/static_intercept_prediction_model.py
```

---

## 📊 最终论文图表（`Fig. 9` – `Fig. 15`）对应清单

所有最终版 300 DPI 图表同时保存在 `revision_experiments/results/figures/` 与 `model/figures/`：

| 论文图号 | 文件名 (`model/figures/`) | 内容说明 |
| :--- | :--- | :--- |
| **Fig. 9** | `fig9_spearman_50features.png` | 50 个候选特征与感知阈值的 Spearman 相关性排位图（按三大模态分色） |
| **Fig. 10** | `fig10_corr_matrix_50features.png` | 50 个候选特征的 50×50 互相关矩阵热力图 |
| **Fig. 11** | `model_comparison_within_vs_cross_margin0.png` | 多类机器学习算法择优对比柱状图（精选各流派最高准确率代表模型 vs. Baseline） |
| **Fig. 12** | `margin_accuracy_within_vs_cross.png` | 容差窗口灵敏度分析折线图（`Margin = 0, 5, 10, 15 μs`） |
| **Fig. 13** | `fig13_inner_3fold_plsda.png` / `feature_count_trend.png` | 内层 3 折（Inner Fold 1, 2, 3）Multi-Block PLS-DA 筛选出的具体特征与 VIP 得分三列图 |
| **Fig. 14** | `feature_combo_margin0.png` / `feature_combo.png` | 人口学基线（`Sex` / `Sex+Height`）与 7 种多模态组合消融对比柱状图 |
| **Fig. 15** | `confusion_matrices_margin0.png` / `confusion_matrices.png` | 100% 真实无后处理的 2×2 混淆矩阵（a: Non-Cross Subject, b: Cross Subject） |

详细通俗解读与每张图的论文写作指南请参阅 [`docs/Final_Figures_and_Nested_CV_Methodology_Guide.docx`](./docs/Final_Figures_and_Nested_CV_Methodology_Guide.docx)。
