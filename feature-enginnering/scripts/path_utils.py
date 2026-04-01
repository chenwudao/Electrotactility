"""特征工程脚本。

文件：path_utils.py
统一使用项目内的相对路径。
"""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_ENG_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = FEATURE_ENG_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
LINEAR_FEATURE_DIR = FEATURE_ENG_DIR / "linear-features"
LINEAR_CANONICAL_DIR = LINEAR_FEATURE_DIR / "experiment1.0-50features"
LINEAR_WORK_DIR = LINEAR_FEATURE_DIR / "experiment2.0-12&11features"
NONLINEAR_FEATURE_DIR = FEATURE_ENG_DIR / "nonlinear-features"
RESULT_DIR = SCRIPT_DIR / "results"

LINEAR_WORK_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

APPROXIMATE_NAME_MAP = {
    "X.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
    "X_人体成分.xlsx": "X_人体成分_new.xlsx",
    "X_电阻抗谱数据.xlsx": "X_电阻抗.xlsx",
    "X_纹理特征结果.xlsx": "X_纹理特征.xlsx",
    "X_纹理特征_去常数项.xlsx": "X_纹理特征.xlsx",
    "X_结合相关互相关_0.9.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
    "X_结合相关互相关_0.9_rfr_60.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
    "X_结合相关互相关_spe_0.7.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
    "X_结合相关互相关_spe_0.8.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
    "X_结合相关互相关_spe_0.9.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
    "X_结合相关互相关_spe_0.9_new.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
    "X_结合相关互相关_spe_0.9_new_rfr_50.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
    "X_结合相关互相关_非线性优先_rfr.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
    "X_pea_sro_结合相关互相关_0.9.xlsx": "X_结合相关互相关_spe_0.9_rfr_50.xlsx",
}

def normalize_name(name: str) -> str:
    """把旧文件名映射到当前项目中的真实文件名。"""
    return APPROXIMATE_NAME_MAP.get(Path(name).name, Path(name).name)

def resolve_input_file(name: str) -> str:
    """解析输入文件路径，优先命中当前项目里的真实文件。"""
    normalized = normalize_name(name)
    candidates = [
        LINEAR_WORK_DIR / normalized,
        LINEAR_CANONICAL_DIR / normalized,
        DATASET_DIR / normalized,
        NONLINEAR_FEATURE_DIR / normalized,
        RESULT_DIR / normalized,
        SCRIPT_DIR / normalized,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    for base_dir in (LINEAR_WORK_DIR, LINEAR_CANONICAL_DIR, DATASET_DIR, NONLINEAR_FEATURE_DIR):
        matches = list(base_dir.rglob(normalized))
        if matches:
            return str(matches[0])

    return str(LINEAR_WORK_DIR / normalized)

def resolve_output_file(name: str, target: str = "work") -> str:
    """生成输出文件路径。"""
    file_name = Path(name).name
    if target == "dataset":
        base_dir = DATASET_DIR
    elif target == "canonical":
        base_dir = LINEAR_CANONICAL_DIR
    elif target == "result":
        base_dir = RESULT_DIR
    else:
        base_dir = LINEAR_WORK_DIR

    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / file_name)
