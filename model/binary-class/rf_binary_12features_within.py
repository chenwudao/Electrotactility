"""模型脚本。

文件：rf_binary_12features_within.py
统一使用项目内的相对路径。
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# ============================================================================
# 配置参数（手动指定）
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
FEAT_FILE = BASE_DIR / "feature-enginnering" / "linear-features" / "experiment1.0-50features" / "X_结合相关互相关_spe_0.9_rfr_50.xlsx"
LABEL_FILE = BASE_DIR / "feature-enginnering" / "linear-features" / "experiment1.0-50features" / "Y.xlsx"
NONLINEAR_FILE = BASE_DIR / "feature-enginnering" / "nonlinear-features" / "12特征_非线性扩展_完整.xlsx"
SUBJECT_COL = "受试者编号"
FINGER_COL_CANDIDATES = ["手指（优食-0 非优中-1 非优尾-2 非优食-3）", "手指"]
DROP_SUB = []

# 跨被试开关（True=跨被试，False=非跨被试）
CROSS_SUB = False

# KFold 参数
N_SPLITS = 5
SHUFFLE = True
KF_RANDOM_STATE = 3477  # 手动指定

# RF参数（手动指定）
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 1612,  # 手动指定
}

# 12个最优特征
SELECTED_12_FEATURES = [
    'v1-v2',
    'LOG_R2',
    '指围-指长',
    '45. FFM% of Right Arm',
    '199. Measured Circumference of Neck',
    '206. Measured Circumference of Left Thigh',
    'GLCM_t_0_d_2_SVD15_465',
    'STHOS_r_2_a_45_LRHGLE_22',
    'LBP_r3_pt_17_342',
    'LBP_r3_pt_18_343',
    'ZER_r3_pt_4_80',
    'ZER_r3_pt_19_95',
]

# 测试的 margin 值（容忍度，单位mA）
MARGIN_VALUES = [0, 5, 10, 15]

# ============================================================================
# 辅助类
# ============================================================================
class my_KFold_Binary:
    """被试内验证器"""
    def __init__(self, tf, subject_col, finger_col, n_splits=5, shuffle=True, random_state=42, drop_sub=[]):
        self.tf = tf
        self.subject_col = subject_col
        self.finger_col = finger_col
        self.n_splits = n_splits
        self.drop_sub = set(drop_sub)
        self.res = []
        
        tf_filtered = self.tf[~self.tf[self.subject_col].isin(self.drop_sub)]
        data = tf_filtered.index
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_i, test_i in kf.split(data):
            train_idx = data[train_i].tolist()
            test_idx = data[test_i].tolist()
            self.res.append((train_idx, test_idx))
    
    def split(self, X=None, y=None, groups=None):
        for train_idx, test_idx in self.res:
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class my_KFold_CrossSub:
    """跨被试验证器：按受试者分组"""
    def __init__(self, tf, subject_col, finger_col, n_splits=5, shuffle=True, random_state=42, drop_sub=[]):
        self.tf = tf
        self.subject_col = subject_col
        self.finger_col = finger_col
        self.n_splits = n_splits
        self.drop_sub = set(drop_sub)
        self.res = []

        subs = np.array(list(set(self.tf[self.subject_col]) - set(self.drop_sub)))
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_i, test_i in kf.split(subs):
            test_subs = subs.take(test_i).tolist()
            train_idx = self.tf[~(self.tf[self.subject_col].isin(test_subs + list(self.drop_sub)))].index.tolist()
            test_idx = self.tf[(self.tf[self.subject_col].isin(test_subs))].index.tolist()
            self.res.append((train_idx, test_idx))

    def split(self, X=None, y=None, groups=None):
        for train_idx, test_idx in self.res:
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

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
    
    # 获取所有特征列
    meta_cols = {"数据编号", SUBJECT_COL, "手指", FINGER_COL_CANDIDATES[0]}
    all_feature_cols = [c for c in df_feat.columns if c not in meta_cols]
    X_all = df_feat[all_feature_cols].to_numpy()
    
    # 创建二分类标签（中位数分割）
    threshold = np.median(y_cont)
    y_binary = np.array([1 if val > threshold else 0 for val in y_cont])
    
    return X_all, y_binary, y_cont, tf, finger_col, all_feature_cols, threshold

# ============================================================================
# 容忍度评分函数
# ============================================================================
def tolerant_scores(y_true, y_pred, y_cont, boundary, margin):
    """计算带宽容的 Acc/F1。"""
    correct = (y_true == y_pred)
    near_boundary = np.abs(y_cont - boundary) <= margin
    corrected = correct | near_boundary
    acc = corrected.mean()
    
    y_pred_adj = y_pred.copy()
    flip_mask = (~correct) & near_boundary
    y_pred_adj[flip_mask] = y_true[flip_mask]
    f1 = f1_score(y_true, y_pred_adj, zero_division=0)
    
    return acc, f1

# ============================================================================
# 评估函数
# ============================================================================
def evaluate_model(X_scaled_sub, y_binary, y_cont, tf, finger_col, threshold, cv):
    """
    评估模型性能（多margin）
    X_scaled_sub: 已经全局缩放并提取子集的特征
    """
    results = []
    
    for margin in MARGIN_VALUES:
        acc_list = []
        f1_list = []
        acc_std_list = []
        f1_std_list = []
        
        for train_idx, test_idx in cv.split():
            X_train, X_test = X_scaled_sub[train_idx], X_scaled_sub[test_idx]
            y_train, y_test = y_binary[train_idx], y_binary[test_idx]
            y_cont_test = y_cont[test_idx]
            
            # 注意：这里不再做缩放，因为已经全局缩放过了
            model = RandomForestClassifier(**RF_CONFIG)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 标准评估
            acc_std = accuracy_score(y_test, y_pred)
            f1_std = f1_score(y_test, y_pred, zero_division=0)
            acc_std_list.append(acc_std)
            f1_std_list.append(f1_std)
            
            # 容忍度评估
            if margin > 0:
                acc, f1 = tolerant_scores(y_test, y_pred, y_cont_test, threshold, margin)
            else:
                acc, f1 = acc_std, f1_std
            
            acc_list.append(acc)
            f1_list.append(f1)
        
        results.append({
            'margin': margin,
            'acc_mean': float(np.mean(acc_list)),
            'acc_std': float(np.std(acc_list)),
            'f1_mean': float(np.mean(f1_list)),
            'f1_std': float(np.std(f1_list)),
            'acc_std_mean': float(np.mean(acc_std_list)),
            'f1_std_mean': float(np.mean(f1_std_list)),
        })
    
    return results

# ============================================================================
# 主函数
# ============================================================================
def main():
    print("="*80)
    print("二分类任务 - 12特征 + 全局MinMaxScaler + 手动指定随机种子")
    print("="*80)
    
    # 加载数据
    X_all, y_binary, y_cont, tf, finger_col, all_feature_cols, threshold = load_data()
    
    print(f"\n数据信息:")
    print(f"  样本总数: {len(y_binary)}")
    print(f"  全部特征数: {len(all_feature_cols)}")
    print(f"  二分类阈值: {threshold:.1f} mA")
    print(f"  类别0: {np.sum(y_binary==0)}, 类别1: {np.sum(y_binary==1)}")
    
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
    
    # 步骤2：提取12个最优特征子集
    print(f"\n步骤2: 提取12个最优特征子集")
    feature_indices = []
    for feat in SELECTED_12_FEATURES:
        if feat in all_feature_cols:
            feature_indices.append(all_feature_cols.index(feat))
        else:
            print(f"  ✗ {feat} (未找到!)")
    
    if len(feature_indices) != len(SELECTED_12_FEATURES):
        print(f"\n⚠️  错误: 只找到 {len(feature_indices)}/{len(SELECTED_12_FEATURES)} 个特征")
        return
    
    X_scaled_sub = X_all_scaled[:, feature_indices]
    print(f"  ✓ 提取完成: {X_scaled_sub.shape}")
    
    # 步骤3：使用手动指定的随机种子评估性能
    print(f"\n步骤3: 使用指定随机种子评估性能")
    print(f"  验证方式        = {'跨被试' if CROSS_SUB else '非跨被试'}")
    print(f"  KF_RANDOM_STATE = {KF_RANDOM_STATE}")
    print(f"  RF_RANDOM_STATE = {RF_CONFIG['random_state']}")
    
    if CROSS_SUB:
        cv = my_KFold_CrossSub(tf, SUBJECT_COL, finger_col, N_SPLITS, SHUFFLE, KF_RANDOM_STATE, DROP_SUB)
    else:
        cv = my_KFold_Binary(tf, SUBJECT_COL, finger_col, N_SPLITS, SHUFFLE, KF_RANDOM_STATE, DROP_SUB)
    
    results = evaluate_model(X_scaled_sub, y_binary, y_cont, tf, finger_col, threshold, cv)
    
    # 最终结果
    print(f"\n" + "="*80)
    print("评估结果（多margin容忍度）:")
    print("="*80)
    
    print(f"\n{'Margin(mA)':>10s} | {'Acc(mean)':>12s} | {'Acc(std)':>10s} | {'F1(mean)':>12s} | {'F1(std)':>10s}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['margin']:>10.0f} | "
              f"{res['acc_mean']:>12.4f} | "
              f"{res['acc_std']:>10.4f} | "
              f"{res['f1_mean']:>12.4f} | "
              f"{res['f1_std']:>10.4f}")
    
    print(f"\n配置信息:")
    print(f"  KF_RANDOM_STATE = {KF_RANDOM_STATE}")
    print(f"  RF_RANDOM_STATE = {RF_CONFIG['random_state']}")
    print(f"  特征数          = {len(SELECTED_12_FEATURES)}")
    
    print(f"\nRF参数配置:")
    for k, v in RF_CONFIG.items():
        print(f"  {k:20s} = {v}")
    print("="*80)
    
    return {
        'kf_state': KF_RANDOM_STATE,
        'rf_seed': RF_CONFIG['random_state'],
        'results': results,
    }

if __name__ == "__main__":
    result = main()
