"""模型脚本。

文件：rf_binary_11features_cross.py
统一使用项目内的相对路径。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import random

# ============================================================================
# 配置参数
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
FEAT_FILE = BASE_DIR / "feature-enginnering" / "linear-features" / "experiment1.0-50features" / "X_结合相关互相关_spe_0.9_rfr_50.xlsx"
LABEL_FILE = BASE_DIR / "feature-enginnering" / "linear-features" / "experiment1.0-50features" / "Y.xlsx"

SUBJECT_COL = "受试者编号"
FINGER_COL_CANDIDATES = ["手指（优食-0 非优中-1 非优尾-2 非优食-3）", "手指"]
DROP_SUB = []

# KFold 参数（手动指定）
N_SPLITS = 5
SHUFFLE = True
KF_RANDOM_STATE = 168  # 手动指定的KFold随机状态

# RF参数（手动指定）
RF_CONFIG = {
    'n_estimators': 400,
    'max_depth': 11,
    'max_features': 'log2',
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'bootstrap': True,
    'random_state': 35,  # 手动指定的RF随机状态
}

# 11个最优特征（跨被试子集）
SELECTED_11_FEATURES = [
    'v1-v2',
    'LOG_R2',
    '指围-指长',
    '45. FFM% of Right Arm',
    '206. Measured Circumference of Left Thigh',
    'BP_Perimeter_11',
    'GLCM_t_0_d_2_SVD15_465',
    'STHOS_r_2_a_45_LRHGLE_22',
    'LBP_r3_pt_17_342',
    'LBP_r3_pt_18_343',
    'ZER_r3_pt_19_95',
]

# 测试的 margin 值（容忍度）
MARGIN_VALUES = [0, 5, 10, 15]

# ============================================================================
# 数据加载
# ============================================================================
def load_data():
    """加载数据并创建二分类标签"""
    df_feat = pd.read_excel(FEAT_FILE)
    tf = pd.read_excel(LABEL_FILE, index_col=0)
    
    # 识别手指列
    finger_col = None
    for col in FINGER_COL_CANDIDATES:
        if col in tf.columns:
            finger_col = col
            break
    
    # 获取连续标签
    if "阈值" in tf.columns:
        y_cont = tf["阈值"].to_numpy()
    else:
        y_cont = tf.iloc[:, 3].to_numpy()
    
    # 创建二分类标签（中位数分割）
    threshold = np.median(y_cont)
    y_binary = np.array([1 if val > threshold else 0 for val in y_cont])
    
    # 获取所有特征列
    meta_cols = {"数据编号", SUBJECT_COL, "手指", FINGER_COL_CANDIDATES[0]}
    all_feature_cols = [c for c in df_feat.columns if c not in meta_cols]
    X_all = df_feat[all_feature_cols].to_numpy()
    
    # 获取受试者编号
    subjects = tf[SUBJECT_COL].to_numpy()
    
    return X_all, y_binary, y_cont, subjects, threshold, all_feature_cols

# ============================================================================
# 容忍度评分函数
# ============================================================================
def tolerant_scores(y_true, y_pred, y_cont, boundary, margin):
    """计算带宽容的 Acc/F1（只判断真实值离边界）"""
    correct = (y_true == y_pred)
    near_true = np.abs(y_cont - boundary) <= margin
    corrected = correct | near_true
    acc = corrected.mean()

    y_pred_adj = y_pred.copy()
    flip_mask = (~correct) & near_true
    y_pred_adj[flip_mask] = y_true[flip_mask]
    f1 = f1_score(y_true, y_pred_adj, zero_division=0)

    return acc, f1

# ============================================================================
# 跨被试验证函数
# ============================================================================
def cross_subject_validation(X_scaled_sub, y_binary, y_cont, subjects, boundary, margin=0):
    """
    跨被试验证（按受试者分组的KFold）
    X_scaled_sub: 已经全局缩放并提取子集的特征
    """
    subs = np.array(list(set(subjects)))
    kf = KFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=KF_RANDOM_STATE)
    
    acc_list = []
    f1_list = []
    all_y_true = []
    all_y_pred = []
    all_y_cont = []

    for fold, (train_sub_idx, test_sub_idx) in enumerate(kf.split(subs), 1):
        test_subs = subs.take(test_sub_idx).tolist()
        train_mask = ~np.isin(subjects, test_subs)
        test_mask = np.isin(subjects, test_subs)

        X_train = X_scaled_sub[train_mask]
        X_test = X_scaled_sub[test_mask]
        y_train = y_binary[train_mask]
        y_test = y_binary[test_mask]
        y_cont_test = y_cont[test_mask]
        
        # 注意：这里不再做缩放，因为已经全局缩放过了
        model = RandomForestClassifier(**RF_CONFIG)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 计算准确率
        if margin > 0:
            acc, f1 = tolerant_scores(y_test, y_pred, y_cont_test, boundary, margin)
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        
        acc_list.append(acc)
        f1_list.append(f1)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_cont.extend(y_cont_test)

        print(f"  Fold {fold:2d}/{N_SPLITS}: {len(y_test):3d}样本, Acc={acc:.4f}, F1={f1:.4f}")
    
    # 整体混淆矩阵与全局指标
    cm = confusion_matrix(all_y_true, all_y_pred)
    if margin > 0:
        global_acc, global_f1 = tolerant_scores(
            np.array(all_y_true),
            np.array(all_y_pred),
            np.array(all_y_cont),
            boundary,
            margin,
        )
    else:
        global_acc = accuracy_score(all_y_true, all_y_pred)
        global_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    
    return {
        "margin": margin,
        "acc_mean": float(np.mean(acc_list)),
        "acc_std": float(np.std(acc_list)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "confusion_matrix": cm,
        "global_acc": float(global_acc),
        "global_f1": float(global_f1),
    }

# ============================================================================
# 主函数
# ============================================================================
def main():
    print("="*100)
    print("二分类任务 - 跨被试验证 + 全局MinMaxScaler + 手动指定随机种子")
    print("="*100)
    
    # 加载数据
    X_all, y_binary, y_cont, subjects, threshold, all_feature_cols = load_data()
    
    print(f"\n数据信息:")
    print(f"  样本总数: {len(y_binary)}")
    print(f"  全部特征数: {len(all_feature_cols)}")
    print(f"  使用特征数: {len(SELECTED_11_FEATURES)}")
    print(f"  二分类阈值: {threshold:.1f} mA")
    print(f"  类别0: {np.sum(y_binary==0)}, 类别1: {np.sum(y_binary==1)}")
    print(f"  受试者数量: {len(set(subjects))}")
    
    # 步骤1：全局MinMaxScaler（50特征）
    print(f"\n步骤1: 全局MinMaxScaler（{len(all_feature_cols)}个特征）")
    scaler = MinMaxScaler()
    scaler.fit(X_all)
    X_all_scaled = scaler.transform(X_all)
    
    # 检查是否有负数
    if np.any(np.less(X_all_scaled, 0)):
        print("  ⚠️  警告: 存在负数!")
    else:
        print("  ✓ 所有数据已归一化到[0,1]")
    
    # 步骤2：提取11个最优特征子集
    print(f"\n步骤2: 提取11个最优特征子集")
    feature_indices = []
    for feat in SELECTED_11_FEATURES:
        if feat in all_feature_cols:
            feature_indices.append(all_feature_cols.index(feat))
        else:
            print(f"  ✗ {feat} (未找到!)")
    
    if len(feature_indices) != len(SELECTED_11_FEATURES):
        print(f"\n⚠️  错误: 只找到 {len(feature_indices)}/{len(SELECTED_11_FEATURES)} 个特征")
        return
    
    X_scaled_sub = X_all_scaled[:, feature_indices]
    print(f"  ✓ 提取完成: {X_scaled_sub.shape}")
    
    print(f"\n【使用的跨被试特征子集】")
    for i, feat in enumerate(SELECTED_11_FEATURES, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\n【配置信息】")
    print(f"  KF_RANDOM_STATE = {KF_RANDOM_STATE}")
    print(f"  RF_RANDOM_STATE = {RF_CONFIG['random_state']}")
    print(f"  N_SPLITS        = {N_SPLITS}")
    
    print(f"\n【RF参数配置】")
    for key, val in RF_CONFIG.items():
        if key != 'random_state':
            print(f"  {key:20s}: {val}")
    
    # 步骤3：跨被试验证（多margin测试）
    print(f"\n" + "="*100)
    print(f"步骤3: 开始跨被试验证 (KFold by Subject)...")
    print("="*100)
    
    results = []
    
    for margin in MARGIN_VALUES:
        print(f"\n【Margin = {margin} mA】")
        result = cross_subject_validation(X_scaled_sub, y_binary, y_cont, subjects, threshold, margin)
        results.append(result)
        
        print(f"\n  平均准确率: {result['acc_mean']:.4f} (±{result['acc_std']:.4f})")
        print(f"  平均F1分数: {result['f1_mean']:.4f} (±{result['f1_std']:.4f})")
        print(f"  全局准确率: {result['global_acc']:.4f}")
        print(f"  全局F1分数: {result['global_f1']:.4f}")
        
        cm = result['confusion_matrix']
        print(f"\n  混淆矩阵:")
        print(f"             预测为0  预测为1")
        print(f"  实际为0: [{cm[0,0]:4d}]  [{cm[0,1]:4d}]")
        print(f"  实际为1: [{cm[1,0]:4d}]  [{cm[1,1]:4d}]")
    
    # 结果汇总
    print(f"\n" + "="*100)
    print("【结果汇总】")
    print("="*100)
    
    print(f"\n{'Margin(mA)':>10s} | {'Acc(mean)':>11s} | {'Acc(std)':>10s} | {'F1(mean)':>11s} | {'F1(std)':>10s}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['margin']:>10.0f} | "
              f"{result['acc_mean']:>11.4f} | "
              f"{result['acc_std']:>10.4f} | "
              f"{result['f1_mean']:>11.4f} | "
              f"{result['f1_std']:>10.4f}")
    
    # 保存结果
    output_dir = Path(__file__).resolve().parent
    output_file = output_dir / "rf_binary_cross_subject_global_scale_results.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 整体结果
        summary_df = pd.DataFrame([
            {
                'Margin (mA)': r['margin'],
                'Accuracy': r['acc_mean'],
                'Accuracy_Std': r['acc_std'],
                'F1-Score': r['f1_mean'],
                'F1_Std': r['f1_std'],
                'Global_Acc': r['global_acc'],
                'Global_F1': r['global_f1'],
            }
            for r in results
        ])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 配置信息
        config_df = pd.DataFrame([
            {'Parameter': 'KF_RANDOM_STATE', 'Value': KF_RANDOM_STATE},
            {'Parameter': 'RF_RANDOM_STATE', 'Value': RF_CONFIG['random_state']},
            {'Parameter': 'N_SPLITS', 'Value': N_SPLITS},
            {'Parameter': 'n_estimators', 'Value': RF_CONFIG['n_estimators']},
            {'Parameter': 'max_depth', 'Value': RF_CONFIG['max_depth']},
            {'Parameter': 'max_features', 'Value': RF_CONFIG['max_features']},
            {'Parameter': 'min_samples_split', 'Value': RF_CONFIG['min_samples_split']},
            {'Parameter': 'min_samples_leaf', 'Value': RF_CONFIG['min_samples_leaf']},
        ])
        config_df.to_excel(writer, sheet_name='Config', index=False)
    
    print(f"\n✓ 结果已保存: {output_file}")
    
    print(f"\n" + "="*100)
    print("【关键结果】")
    print("="*100)
    
    print(f"\n1. 跨被试验证（泛化能力测试）:")
    print(f"   - 标准准确率 (margin=0): {results[0]['acc_mean']:.4f} (±{results[0]['acc_std']:.4f})")
    best_result = max(results, key=lambda r: r['acc_mean'])
    print(
        f"   - 最佳宽容准确率 (margin={best_result['margin']}): "
        f"{best_result['acc_mean']:.4f} (±{best_result['acc_std']:.4f})"
    )
    
    print(f"\n2. 配置:")
    print(f"   - KF_RANDOM_STATE = {KF_RANDOM_STATE}")
    print(f"   - RF_RANDOM_STATE = {RF_CONFIG['random_state']}")
    print(f"   - 特征数: {len(SELECTED_11_FEATURES)}")
    
    print(f"\n" + "="*100)
    print("分析完成！")
    print("="*100)
    
    return {
        'kf_state': KF_RANDOM_STATE,
        'rf_seed': RF_CONFIG['random_state'],
        'results': results,
    }

if __name__ == "__main__":
    result = main()
