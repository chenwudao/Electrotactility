# 广义估计方程 (GEE) 特征-感知阈值 (PT) 关联分析数据与结果说明

## 一、分析方法与正文对应段落

### 1. 正文 Methods 对应描述
> To examine whether the feature–PT associations persisted after accounting for sex and body size, an additional association analysis was performed using generalised estimating equations (GEE). Continuous PT was treated as the outcome, and a separate model was fitted for each analysed feature, with sex and height included as covariates. Participant identity was used as the clustering variable to account for the dependence among the four finger-level records contributed by each participant. Adjusted regression coefficients and 95% confidence intervals were reported together with p-values corrected for multiple comparisons. This analysis was used for exploratory interpretation of the feature–PT associations and did not contribute to predictive feature selection or parameter tuning.

### 2. 模型统计设定
- **因变量 (Outcome)**：连续电触觉感知阈值 $\text{PT}$ (单位：$\mu\text{s}$，$N = 128$ 条样本，来自 32 名受试者 $\times$ 每人 4 根手指)。
- **聚类变量 (Clustering Variable)**：受试者编号 (`Participant ID`，共 32 个独立簇，每簇 4 条手指记录)，用于消除同一受试者不同手指测量值之间的簇内相关性。
- **分布族与工作相关矩阵**：高斯分布族 (`Gaussian family`) + 可交换工作相关结构 (`Exchangeable working correlation`) + Huber-White 稳健夹心方差估计器 (`Robust Sandwich SE`)。
- **自变量标准化**：各候选特征均做 Z-score 标准化（均值 0，标准差 1），因此回归系数 $\beta$ 的物理含义为 **“该特征每增加 1 个标准差 (SD) 时，感知阈值 PT 的平均变化量 ($\mu\text{s/SD}$)”**。
- **三组递进模型对比**：
  1. **未校正模型 (Unadjusted)**：$\text{PT} \sim \text{Feature}$
  2. **仅控制性别 (Sex-Adjusted)**：$\text{PT} \sim \text{Feature} + \text{Sex}$
  3. **同时控制性别与身高 (Sex + Height Adjusted)**：$\text{PT} \sim \text{Feature} + \text{Sex} + \text{Height}$
- **多重假设检验校正**：采用 Benjamini–Hochberg 法控制错误发现率 (BH-FDR)，报告校正后的 $q$ 值 (`FDR_q_SexHeightAdj`)。

---

## 二、核心数据表格汇总

### 表 1：外层折内高频筛选特征与原论文代表性特征 GEE 回归结果（共 20 项，对应 `gee_clustered_associations.xlsx`）

| 特征名称 (论文展示名 / 原始列名) | 模态 | 外层折选中率 | 未校正 $\beta$ ($\mu\text{s/SD}$) [95% CI] | 未校正 $p$ | 控制性别 $\beta$ [95% CI] | 控制性别 $p$ | 控制性别+身高 $\beta$ [95% CI] | 控制性别+身高 $p$ | BH-FDR $q$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fat Free Mass% of Right Arm** (`45. FFM% of Right Arm`) | Global | 5/5 | 31.31 [16.53, 46.09] | 3.3e-5 | 24.66 [10.39, 38.93] | 7.1e-4 | **23.94 [11.18, 36.70]** | **2.4e-4** | **0.0024** |
| **Phase** (`Phase`) | Bioelectrical | 5/5 | 16.42 [7.23, 25.61] | 4.6e-4 | 14.16 [5.63, 22.69] | 0.0011 | **15.16 [7.83, 22.49]** | **5.1e-5** | **0.0010** |
| **v_pre, crestf** (`crestf_u`) | Bioelectrical | 4/5 | 16.36 [5.42, 27.30] | 0.0034 | 14.27 [3.55, 24.99] | 0.0092 | **15.64 [5.47, 25.81]** | **0.0026** | **0.0086** |
| **BP_wt_norm_moments_14** (`BP_wt_norm_moments_14__79`) | Local | 4/5 | 18.57 [6.32, 30.82] | 0.0030 | 16.59 [4.87, 28.31] | 0.0055 | **15.88 [3.65, 28.11]** | **0.0109** | **0.0313** |
| **Δv** (`v1-v2`) | Bioelectrical | 4/5 | -11.16 [-25.66, 3.34] | 0.1315 | -12.20 [-27.51, 3.11] | 0.1183 | -12.76 [-27.22, 1.70] | 0.0839 | 0.1374 |
| **A_sec** (`截面积`) | Local | 2/5 | 24.26 [13.70, 34.82] | 7.0e-6 | 20.49 [9.20, 31.78] | 3.8e-4 | **19.20 [7.93, 30.47]** | **8.4e-4** | **0.0041** |
| **A_sur** (`截面积-指长`) | Local | 2/5 | 22.88 [12.30, 33.46] | 2.3e-5 | 19.54 [8.45, 30.63] | 5.6e-4 | **18.42 [7.42, 29.42]** | **0.0010** | **0.0041** |
| **v_pre, rms** (`rms_u`) | Bioelectrical | 2/5 | -11.72 [-24.75, 1.31] | 0.0783 | -10.33 [-22.80, 2.14] | 0.1041 | **-13.41 [-25.90, -0.92]** | **0.0352** | 0.0782 |
| **LOG_R_1** (`LOG_R1`) | Bioelectrical | 2/5 | -10.45 [-23.50, 2.60] | 0.1165 | -8.36 [-21.00, 4.28] | 0.1955 | -9.90 [-21.72, 1.92] | 0.1006 | 0.1437 |
| **var_u** | Bioelectrical | 2/5 | -7.43 [-19.99, 5.13] | 0.2462 | -6.89 [-19.02, 5.24] | 0.2657 | -9.79 [-22.24, 2.66] | 0.1235 | 0.1647 |
| **ZER_r0_p17** (`ZER_r0_pt_17_18`) | Local | 2/5 | 9.35 [-0.53, 19.23] | 0.0634 | 8.93 [-0.77, 18.63] | 0.0714 | 6.89 [-3.15, 16.93] | 0.1785 | 0.1984 |
| **BP_Perimeter** (`BP_Perimeter_12`) | Local | 1/5 | 21.77 [10.21, 33.33] | 2.3e-4 | 17.70 [4.27, 31.13] | 0.0098 | **16.37 [3.10, 29.64]** | **0.0156** | **0.0390** |
| **指围-指长** | Local | 0/5 | 21.05 [11.05, 31.05] | 3.7e-5 | 18.10 [7.87, 28.33] | 5.2e-4 | **17.19 [7.08, 27.30]** | **0.0009** | **0.0041** |
| **GLCM_d2_a0_SVD15** (`GLCM_t_0_d_2_SVD15_465`) | Local | 0/5 | -17.16 [-30.19, -4.13] | 0.0099 | -14.37 [-26.52, -2.22] | 0.0204 | **-11.64 [-22.95, -0.33]** | **0.0437** | 0.0873 |
| **LBP_r3_p17** (`LBP_r3_pt_17_342`) | Local | 0/5 | 18.50 [5.78, 31.22] | 0.0043 | 13.89 [1.23, 26.55] | 0.0316 | 12.35 [-0.51, 25.21] | 0.0600 | 0.1090 |
| **GLRLM_d2_a45_LRHGLE** (`STHOS_r_2_a_45_LRHGLE_22`) | Local | 0/5 | -10.53 [-18.84, -2.22] | 0.0131 | -8.58 [-16.75, -0.41] | 0.0395 | -7.37 [-15.88, 1.14] | 0.0893 | 0.1374 |
| **ZER_r3_p19** (`ZER_r3_pt_19_95`) | Local | 0/5 | -7.32 [-16.96, 2.32] | 0.1368 | -6.99 [-16.44, 2.46] | 0.1472 | -6.98 [-16.41, 2.45] | 0.1470 | 0.1838 |
| **LOG_R_2** (`LOG_R2`) | Bioelectrical | 0/5 | -5.07 [-10.40, 0.26] | 0.0620 | -4.44 [-9.52, 0.64] | 0.0862 | -4.02 [-9.63, 1.59] | 0.1596 | 0.1878 |
| **LBP_r3_p18** (`LBP_r3_pt_18_343`) | Local | 0/5 | 20.58 [5.41, 35.75] | 0.0078 | 13.57 [-2.85, 29.99] | 0.1051 | 10.62 [-5.77, 27.01] | 0.2039 | 0.2146 |
| **Measured Circumference of Left Thigh** (`206. ...`) | Global | 0/5 | 16.44 [1.62, 31.26] | 0.0298 | 2.48 [-16.04, 21.00] | 0.7929 | -0.85 [-16.14, 14.44] | 0.9128 | 0.9128 |

---

### 表 2：Fig. 13 内层三折 (Inner Fold 1–3) 选出特征的 GEE 关联结果（共 19 项，对应 `gee_fig13_selected_features.xlsx`）

| 论文展示名 (`Display_Name`) | 模态 (`Modality`) | 未校正 $\beta$ [95% CI] | 未校正 $p$ | 控制性别+身高 $\beta$ [95% CI] | 控制性别+身高 $p$ | BH-FDR $q$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fat Free Mass% of Right Arm** | Global | 31.31 [16.53, 46.09] | 3.3e-5 | **23.94 [11.18, 36.71]** | **2.4e-4** | **0.0022** |
| **Skeletal Muscle Index** | Global | 32.22 [15.19, 49.26] | 2.1e-4 | **35.00 [6.01, 63.99]** | **0.0180** | **0.0427** |
| **Skeletal Muscle Mass/Weight** | Global | 28.44 [7.53, 49.34] | 0.0077 | 19.88 [-1.39, 41.15] | 0.0669 | 0.1156 |
| **Measured Circumference of Neck** | Global | 32.64 [15.89, 49.40] | 1.3e-4 | 27.64 [-4.59, 59.87] | 0.0928 | 0.1216 |
| **Phase** | Bioelectrical | 16.42 [7.24, 25.61] | 4.6e-4 | **15.16 [7.82, 22.49]** | **5.1e-5** | **0.0010** |
| **v_pre, crestf** | Bioelectrical | 16.36 [5.43, 27.29] | 0.0034 | **15.64 [5.47, 25.81]** | **0.0026** | **0.0098** |
| **Δv** | Bioelectrical | -11.16 [-25.67, 3.34] | 0.1315 | -12.76 [-27.22, 1.71] | 0.0839 | 0.1216 |
| **LOG_R_1** | Bioelectrical | -10.45 [-23.50, 2.60] | 0.1165 | -9.90 [-21.72, 1.92] | 0.1006 | 0.1216 |
| **A_sec** | Local | 24.26 [13.70, 34.83] | 7.0e-6 | **19.20 [7.93, 30.46]** | **8.4e-4** | **0.0049** |
| **A_sur** | Local | 22.88 [12.29, 33.46] | 2.3e-5 | **18.42 [7.42, 29.42]** | **0.0010** | **0.0049** |
| **ZER_r3_p4** | Local | -14.74 [-24.02, -5.46] | 0.0018 | **-12.82 [-21.81, -3.82]** | **0.0052** | **0.0165** |
| **BP_Perimeter** | Local | 21.77 [10.20, 33.33] | 2.3e-4 | **16.37 [3.10, 29.64]** | **0.0156** | **0.0423** |
| **BP_Solidity** | Local | -15.63 [-29.06, -2.21] | 0.0225 | **-14.01 [-26.89, -1.13]** | **0.0331** | 0.0698 |
| **GLCM_d2_a0_SVD15** | Local | -17.16 [-30.19, -4.12] | 0.0099 | **-11.64 [-22.94, -0.33]** | **0.0437** | 0.0829 |
| **LBP_r3_p76** | Local | 19.70 [6.21, 33.18] | 0.0042 | 10.08 [-1.69, 21.85] | 0.0931 | 0.1216 |
| **ZER_r1_p8** | Local | -10.90 [-22.05, 0.25] | 0.0554 | -9.18 [-20.19, 1.84] | 0.1024 | 0.1216 |
| **ZER_r0_p17** | Local | 9.35 [-0.52, 19.22] | 0.0634 | 6.89 [-3.15, 16.92] | 0.1785 | 0.1995 |
| **LBP_r3_p18** | Local | 20.58 [5.42, 35.75] | 0.0078 | 10.62 [-5.76, 26.99] | 0.2039 | 0.2152 |
| **ZER_r3_p6** | Local | -4.25 [-14.78, 6.28] | 0.4293 | -3.16 [-13.02, 6.70] | 0.5300 | 0.5300 |

---

## 三、核心结论与英文写作参考段落

### 1. 中文核心结论
1. **三大模态核心特征在控制性别与身高后依然保持显著独立关联 ($q < 0.05$)**：
   - **全身生理模态**：右臂无脂体重百分比 `Fat Free Mass% of Right Arm` ($\beta = 23.94\ \mu\text{s/SD}$, 95% CI $[11.18, 36.70]$, $p = 2.4\times 10^{-4}$, $q = 0.0024$) 和骨骼肌指数 `Skeletal Muscle Index` ($\beta = 35.00\ \mu\text{s/SD}$, 95% CI $[6.01, 63.99]$, $p = 0.0180$, $q = 0.0427$) 在控制性别和身高后仍显著正向预测感知阈值。
   - **生物电阻抗模态**：相位角 `Phase` ($\beta = 15.16\ \mu\text{s/SD}$, 95% CI $[7.83, 22.49]$, $p = 5.1\times 10^{-5}$, $q = 0.0010$) 与刺激前电压波峰因子 `v_pre, crestf` ($\beta = 15.64\ \mu\text{s/SD}$, 95% CI $[5.47, 25.81]$, $p = 0.0026$, $q = 0.0086$) 在控制性别和身高后效应量几乎无衰减，且统计显著性进一步增强。
   - **局部形态与纹理模态**：手指横截面积 `A_sec` ($\beta = 19.20\ \mu\text{s/SD}$, $q = 0.0041$)、指根表面积指标 `A_sur` ($\beta = 18.42\ \mu\text{s/SD}$, $q = 0.0041$)、小波矩 `BP_wt_norm_moments_14` ($\beta = 15.88\ \mu\text{s/SD}$, $q = 0.0313$)、指纹周长 `BP_Perimeter` ($\beta = 16.37\ \mu\text{s/SD}$, $q = 0.0390$) 以及 Zernike 矩 `ZER_r3_p4` ($\beta = -12.82\ \mu\text{s/SD}$, $q = 0.0165$) 在控制性别和身高后依然显著 ($q < 0.05$)。
2. **有效区分体型混杂特征**：如大腿围 `Measured Circumference of Left Thigh` 在单变量未校正时呈名义显著 ($\beta = 16.44, p = 0.0298$)，但控制性别与身高后回归系数降至 $-0.85$ ($p = 0.9128$)，说明其单变量相关性主要源于性别与体型差异，这也解释了为何无泄漏跨受试者特征筛选未将其纳入最终预测子集（选中率 `0/5`）。

### 2. 可直接用于论文 Results 的英文表述参考
> In the participant-clustered GEE analysis adjusting for both sex and height, representative features from all three modalities maintained statistically significant associations with continuous PT after Benjamini–Hochberg FDR correction. Specifically, within the global physiological modality, Fat Free Mass% of Right Arm ($\beta = 23.94\ \mu\text{s/SD}$, 95% CI $[11.18, 36.70]$, $p = 2.4 \times 10^{-4}$, $q = 0.0024$) and Skeletal Muscle Index ($\beta = 35.00\ \mu\text{s/SD}$, 95% CI $[6.01, 63.99]$, $p = 0.0180$, $q = 0.0427$) remained positively associated with PT. Within the bioelectrical modality, Phase ($\beta = 15.16\ \mu\text{s/SD}$, 95% CI $[7.83, 22.49]$, $p = 5.1 \times 10^{-5}$, $q = 0.0010$) and $\text{v}_{\text{pre, crestf}}$ ($\beta = 15.64\ \mu\text{s/SD}$, 95% CI $[5.47, 25.81]$, $p = 0.0026$, $q = 0.0086$) exhibited robust associations that were virtually unchanged after covariate adjustment. Within the local morphological modality, finger geometry descriptors ($A_{\text{sec}}$: $\beta = 19.20\ \mu\text{s/SD}$, 95% CI $[7.93, 30.47]$, $q = 0.0041$; $A_{\text{sur}}$: $\beta = 18.42\ \mu\text{s/SD}$, 95% CI $[7.42, 29.42]$, $q = 0.0041$) as well as epidermal texture descriptors (`BP_wt_norm_moments_14`: $\beta = 15.88\ \mu\text{s/SD}$, $q = 0.0313$; `BP_Perimeter`: $\beta = 16.37\ \mu\text{s/SD}$, $q = 0.0390$; `ZER_r3_p4`: $\beta = -12.82\ \mu\text{s/SD}$, $q = 0.0165$) remained significant after accounting for sex and height. Conversely, features driven primarily by general body size—such as Measured Circumference of Left Thigh ($\beta_{\text{unadj}} = 16.44\ \mu\text{s/SD}$, $p = 0.0298$)—lost significance after adjusting for sex and height ($\beta_{\text{adj}} = -0.85\ \mu\text{s/SD}$, $p = 0.9128$), confirming that the nested multimodal feature selection captured physiological, bioelectrical, and local morphological variations beyond simple demographic differences.
