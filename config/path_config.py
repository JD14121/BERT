"""集中路径配置 - 修复 D1 硬编码路径（calibrate_l1l2l4_vs_google.py 原硬编码
C:\\PAEMS-... 与 C:\\Users\\10124\\Desktop\\... 在本机不存在）。

注意 R5 路径不对称：Google-data 在父级 alphaquibit-main（单层），
而 PAEMS/工作区在 alphaquibit-main\\alphaquibit-main（双层）。
"""
from pathlib import Path

# 工作区根（双层 alphaquibit-main）
PROJECT_ROOT = Path(r"D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main")

# 官方 PAEMS 代码
PAEMS_ROOT = PROJECT_ROOT / "PAEMS" / "PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors-main"
PAEMS_SC = PAEMS_ROOT / "Surface_Code_Simulation"          # inject_basic_noise.py / surface_code_generate_circuits.py 所在
TIERS_DIR = PAEMS_SC / "paems_qubit_noise_tiers"           # gen_level_params.py / gen_pair_overrides.py / crosstalk_presets/

# Google 数据（已迁移到 google_paems_data/Google-data）
GOOGLE_DATA = PROJECT_ROOT / "google_paems_data" / "Google-data"
GOOGLE_SC = GOOGLE_DATA / "google_105Q_surface_code_d3_d5_d7"

# 本工程工作目录
WORK_DIR = PROJECT_ROOT / "google_paems_data"
CODE_DIR = WORK_DIR / "code"
DATA_DIR = WORK_DIR / "data"
CONFIG_DIR = WORK_DIR / "configs"
LOG_DIR = WORK_DIR / "logs"

# 每个 distance 选一个代表性 Google patch（XZZX 拓扑模板来源）
# d3: q10_7; d5: q8_7（d5 仅 q4_7/q6_5/q6_9/q8_7）; d7: q6_7
GOOGLE_PATCH = {3: "q10_7", 5: "q8_7", 7: "q6_7"}

# Google 实际可用的 rounds（circuit_ideal.stim 仅这些轮数存在）
GOOGLE_ROUNDS = [1, 10, 13, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]


def google_template_path(distance: int, basis: str, rounds: int) -> Path:
    """Google circuit_ideal.stim 路径（XZZX 噪声无关模板）。rounds 必须在 GOOGLE_ROUNDS 中。"""
    assert rounds in GOOGLE_ROUNDS, f"rounds={rounds} 不在 Google 可用轮数 {GOOGLE_ROUNDS}"
    patch = GOOGLE_PATCH[distance]
    return GOOGLE_SC / f"d{distance}_at_{patch}" / basis.upper() / f"r{rounds:02d}" / "circuit_ideal.stim"
