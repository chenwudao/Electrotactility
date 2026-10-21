# Electrotactility 项目技术与实验文档中心

本项目针对电触觉感知阈值预测模型的研究与顶级期刊审稿意见返修，建立了系统的实验日志、方法学基准及数据归档体系。

## 文档索引

1. [实验进程全记录 (docs/experiment_progress.md)](file:///home/jesse/桌面/Electrotactility/docs/experiment_progress.md)
   * 记录审稿人 4 大核心质疑点及其完备解决方案；
   * 严格基于 **1533 维全特征池**（人体成分 + 电阻抗 + 图像纹理）的特征数量扫描与无泄漏嵌套交叉验证结果；
   * 跨被试（Cross-Subject，Group-KFold）与非跨被试（Within-Subject）性能对照表；
   * 人口学消融基线（Demographic Baselines）与 GEE 聚类统计检验数据；
   * 连续阈值回归（$R^2$ 提升）最新实验结果。

2. **核心代码与结果文件指引**
   * 全特征池系统性扫描脚本：[exp5_full_feature_pool_comprehensive_scan.py](file:///home/jesse/桌面/Electrotactility/revision_experiments/exp5_full_feature_pool_comprehensive_scan.py)
   * 特征数量扫描结果表：`revision_experiments/results/all_features_k_scan_results.xlsx`
   * 无泄漏 Nested CV 网格结果表：`revision_experiments/results/all_features_nested_cv_grid_results.xlsx`
   * 主打跨被试 11 特征模型脚本：[rf_binary_11features_cross.py](file:///home/jesse/桌面/Electrotactility/model/binary-class/rf_binary_11features_cross.py)
