# -*- coding: utf-8 -*-
"""特征工程脚本。

文件：overfit_validation_50features.py
统一使用项目内的相对路径。
"""

from path_utils import resolve_input_file, resolve_output_file, LINEAR_WORK_DIR
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

COL_NUM = 3   # 特征从第 4 列开始
RANDOM_STATE = 83  # 与 rfc_import_props 跨被试一致

def load_y(data_dir):
    tf = pd.read_excel(resolve_input_file('Y.xlsx'), index_col=0)
    y = tf.loc[:, :].values[:, 3].astype(float).ravel()
    return y

def discretize_y_to_binary(y_cont, threshold=None):
    """
    将连续 y 离散化为二分类标签：
    - threshold=None 时使用中位数阈值（>=median 为 1，否则 0）
    - 可传入自定义 threshold
    """
    y_cont = np.asarray(y_cont, dtype=float).ravel()
    thr = float(np.median(y_cont)) if threshold is None else float(threshold)
    y_bin = (y_cont >= thr).astype(int)
    return y_bin, thr

def load_50features(data_dir):
    path = resolve_input_file('X_结合相关互相关_spe_0.9_new_rfr_50.xlsx')
    df = pd.read_excel(path, index_col=0)
    X = df.iloc[:, COL_NUM:].astype(float).values
    return X, df.columns.tolist()[COL_NUM:]

def load_raw_datasets(data_dir):
    """尝试读取原始数据表，用于说明特征来源。"""
    info = []
    for name in ['X_人体成分_new', 'X_人体成分', 'X_纹理特征_去常数项', 'X_纹理特征', 'X_电阻抗']:
        path = resolve_input_file(name + '.xlsx')
        if path and path.endswith('.xlsx'):
            df = pd.read_excel(path, index_col=0)
            n_feat = max(0, df.shape[1] - COL_NUM)
            info.append((name, df.shape[0], n_feat))
    return info

def main():
    print('数据目录:', LINEAR_WORK_DIR)
    # 1) 读取 Y 与 50 特征
    y_cont = load_y(LINEAR_WORK_DIR)
    # 二分类离散化（默认：中位数阈值）
    Y_THRESHOLD = None  # 可改成具体数值，如 0.7 / 100 等
    y, thr = discretize_y_to_binary(y_cont, threshold=Y_THRESHOLD)
    X_50, feat_names = load_50features(LINEAR_WORK_DIR)
    n_samples, n_features = X_50.shape
    print('样本数: {}, 50 特征数: {}'.format(n_samples, n_features))
    print('二分类阈值:', thr, '正类比例:', float(np.mean(y)))

    # 2) 原始数据集概况（若存在）
    raw_info = load_raw_datasets(LINEAR_WORK_DIR)
    if raw_info:
        print('已发现原始数据集: ', raw_info)

    # 3) 二分类器（随机森林分类）
    clf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=300,
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        n_jobs=-1,
    )

    # 4) 5 折分层交叉验证（分类指标：Accuracy/F1/AUC）
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_res = cross_validate(
        clf,
        X_50,
        y,
        cv=skf,
        scoring={
            'acc': 'accuracy',
            'f1': 'f1',
            'auc': 'roc_auc',
        },
        return_train_score=False,
        n_jobs=-1,
    )
    acc_mean = float(np.mean(cv_res['test_acc']))
    acc_std = float(np.std(cv_res['test_acc']))
    f1_mean = float(np.mean(cv_res['test_f1']))
    f1_std = float(np.std(cv_res['test_f1']))
    auc_mean = float(np.mean(cv_res['test_auc']))
    auc_std = float(np.std(cv_res['test_auc']))

    # 5) 留一法 LOO（样本量较大时很慢，默认仅在 n<=200 时运行；可改 RUN_LOO=True 强制运行）
    RUN_LOO = False  # 改为 True 可强制跑完整 LOO
    use_loo = RUN_LOO or n_samples <= 200
    if use_loo:
        loo = LeaveOneOut()
        loo_res = cross_validate(
            clf,
            X_50,
            y,
            cv=loo,
            scoring={
                # LOO 下 AUC 可能不稳定/不可用（单样本），这里只统计 acc
                'acc': 'accuracy',
            },
            return_train_score=False,
            n_jobs=-1,
        )
        loo_acc_mean = float(np.mean(loo_res['test_acc']))
        loo_acc_std = float(np.std(loo_res['test_acc']))
    else:
        loo_acc_mean = loo_acc_std = np.nan
        print('样本数较大，本次未运行 LOO（可把 RUN_LOO=True 强制运行；或仅在小样本子集上运行）')

    # 6) 训练集拟合得分（仅作参考；更重要看 CV 指标）
    clf.fit(X_50, y)
    y_pred_train = clf.predict(X_50)
    y_proba_train = clf.predict_proba(X_50)[:, 1]
    train_acc = float(accuracy_score(y, y_pred_train))
    train_f1 = float(f1_score(y, y_pred_train))
    train_auc = float(roc_auc_score(y, y_proba_train))

    # 7) 汇总与过拟合风险判断（用训练 vs CV 的指标差距判断）
    acc_gap = train_acc - acc_mean
    auc_gap = train_auc - auc_mean
    risk_note = (
        '存在过拟合风险(训练指标明显更高)'
        if (acc_gap > 0.10) or (auc_gap > 0.10)
        else ('可能存在轻度乐观偏差' if (acc_gap > 0.05) or (auc_gap > 0.05) else '训练/泛化差距较小')
    )

    summary = {
        'n_samples': n_samples,
        'n_features_50': n_features,
        'y_threshold': round(float(thr), 6),
        'y_pos_ratio': round(float(np.mean(y)), 6),
        'train_acc': round(train_acc, 6),
        'train_f1': round(train_f1, 6),
        'train_auc': round(train_auc, 6),
        '5fold_acc_mean': round(acc_mean, 6),
        '5fold_acc_std': round(acc_std, 6),
        '5fold_f1_mean': round(f1_mean, 6),
        '5fold_f1_std': round(f1_std, 6),
        '5fold_auc_mean': round(auc_mean, 6),
        '5fold_auc_std': round(auc_std, 6),
        'acc_gap(train-cv)': round(acc_gap, 6),
        'auc_gap(train-cv)': round(auc_gap, 6),
        'loo_acc_mean': round(loo_acc_mean, 6) if use_loo else None,
        'loo_acc_std': round(loo_acc_std, 6) if use_loo else None,
        'overfit_risk_note': risk_note,
    }

    res_df = pd.DataFrame([summary])
    out_path = resolve_output_file('overfit_validation_50features_result.xlsx', target='result')
    res_df.to_excel(out_path, index=False)
    print('结果已保存:', out_path)
    print(res_df.T)

    # 8) 简短结论与留一法说明写入文本
    readme_path = resolve_output_file('overfit_validation_50features_readme.txt', target='result')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('50 特征过拟合验证说明\n')
        f.write('='*50 + '\n')
        f.write('数据: Y.xlsx + X_结合相关互相关_spe_0.9_rfr_50.xlsx\n')
        f.write('二分类任务：将 Y 第 4 列离散化（默认用中位数阈值，>=阈值为 1，否则 0）。\n')
        f.write('评估：Accuracy / F1 / AUC（5 折分层交叉验证）。\n')
        f.write('训练指标 vs 5 折 CV 指标 差距过大则提示过拟合风险。\n\n')
        f.write('留一法(LOO): 可行。\n')
        f.write('- 优点: 每次用 n-1 样本训练、1 个测试，无随机划分，估计稳定。\n')
        f.write('- 缺点: 计算量大(n 次拟合)；样本量大时建议用 5/10 折 CV 更实用。\n')
        f.write('- 注意: LOO 下 AUC 不稳定/可能不可用，因此脚本仅统计 LOO 的 Accuracy。\n')
    print('说明已保存:', readme_path)
    return summary

if __name__ == '__main__':
    main()
