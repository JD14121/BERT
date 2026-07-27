"""集中路径配置"""
from pathlib import Path

# === 工作区根目录（根据你的情况已配置） ===
PROJECT_ROOT = Path(r"D:\STUDY\Project\BERT\BERT")

# === 官方 PAEMS 代码内部路径 (保留不变) ===
PAEMS_ROOT = PROJECT_ROOT / "PAEMS" / "PAEMS-Precise_and_Adaptive_Error_Model_for_superconducting_Quantum_Processors-main"
PAEMS_SC = PAEMS_ROOT / "Surface_Code_Simulation"
TIERS_DIR = PAEMS_SC / "paems_qubit_noise_tiers"

# === 【重点修改 1】：Google 数据实际的绝对路径 ===
GOOGLE_DATA = Path(r"D:\STUDY\Project\alphaquibit-main\alphaqubit data")
GOOGLE_SC = GOOGLE_DATA / "google_105Q_surface_code_d3_d5_d7"

# === 【重点修改 2】：生成数据的读取目录 ===
# 将其指向你刚才生成数据的 scripts/data 目录
WORK_DIR = PROJECT_ROOT / "scripts"
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = WORK_DIR / "data"        # 现在 DATA_DIR 自动指向了 D:\STUDY\Project\BERT\BERT\scripts\data
CONFIG_DIR = WORK_DIR / "configs"
LOG_DIR = WORK_DIR / "logs"

# === 拓扑与轮数配置 (保留不变) ===
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