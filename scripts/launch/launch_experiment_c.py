#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""launch_experiment_c.py
实验 C 一键启动脚本（待云端服务器就绪后执行）。
1. 预检：GPU 驱动 / bert_pretrain_d7 / 数据 / 导入
2. 上传 two_stage.py
3. 后台 nohup 跑 two_stage.py（stage1+stage2）+ eval_ler
4. 验证启动
"""
import paramiko, time, sys

# 新服务器凭据
HOST, PORT, USER, PWD = '180.127.11.177', 24112, 'root', 'aeF3thiu'
# 备用：移动 223.109.239.36:24112

def connect():
    for host, port, label in [(HOST, PORT, '电信'), ('223.109.239.36', 24112, '移动')]:
        try:
            cli = paramiko.SSHClient(); cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(host, port=port, username=USER, password=PWD, timeout=20)
            print(f"[{label}] CONNECTED"); return cli
        except Exception as e:
            print(f"[{label}] {type(e).__name__}")
    print("SSH 都不通 - 服务器未就绪"); sys.exit(1)

cli = connect()

# ===== 1. 预检 =====
print("\n===== 1. 预检 =====")
CHECKS = [
    ("GPU 驱动", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1"),
    ("bert_pretrain_d7/best.pt", "ls -la /root/beat_mwpm/google_paems_data/bert_experiment/checkpoints/bert_pretrain_d7/best.pt 2>&1"),
    ("real_d7 数据", "ls /root/data/real_d7/train_d7_r10_n*_Z.pt 2>&1"),
    ("d7 npy_compressed", "ls /root/data/d7/npy_compressed/meta.json 2>&1"),
    ("quantum_env 导入测试", "cd /root/beat_mwpm/google_paems_data/bert_experiment && /root/miniconda3/envs/quantum_env/bin/python -c 'from run_experiment import finetune, evaluate_model; from xzzx_decoder import XZZXFineTuneDecoder; from path_config import DATA_DIR, google_template_path; print(\"IMPORT_OK\")' 2>&1"),
]
all_ok = True
for name, cmd in CHECKS:
    _, out, _ = cli.exec_command(cmd, timeout=60)
    res = out.read().decode('utf-8', errors='replace').strip()
    ok = ('IMPORT_OK' in res) or ('MiB' in res and 'failed' not in res.lower()) or ('best.pt' in res) or ('meta.json' in res) or ('train_d7' in res)
    if name == 'quantum_env 导入测试': ok = 'IMPORT_OK' in res
    print(f"  [{'✓' if ok else '✗'}] {name}: {res[:80]}")
    if not ok: all_ok = False

if not all_ok:
    print("\n预检未全过 - 不启动。请先确保 GPU 驱动加载 + 数据/代码就位"); cli.close(); sys.exit(1)
print("\n预检全过 ✓")

# ===== 2. 上传 two_stage.py =====
print("\n===== 2. 上传 two_stage.py =====")
sftp = cli.open_sftp()
sftp.put(r'C:\Users\Administrator\two_stage.py', '/root/two_stage.py')
sftp.close()
print("uploaded -> /root/two_stage.py")

# ===== 3. 后台启动 =====
print("\n===== 3. 后台启动 two_stage + eval_ler =====")
launch = '''cd /root/beat_mwpm/google_paems_data/bert_experiment && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
nohup bash -c "set -o pipefail; \
PYTHONIOENCODING=utf-8 /root/miniconda3/envs/quantum_env/bin/python -B -u /root/two_stage.py \
  2>&1 | tee /root/data/d7_twostage_run.log && \
PYTHONIOENCODING=utf-8 /root/miniconda3/envs/quantum_env/bin/python -B -u eval_ler.py --distances 7 --ft-suffix _twostage \
  2>&1 | tee /root/data/d7_twostage_eval_ler.log && \
cp results_ler_d7.json results_ler_d7_twostage.json && echo D7_TWOSTAGE_DONE" > /root/data/d7_twostage_opt.log 2>&1 < /dev/null &
'''
chan = cli.get_transport().open_session()
chan.exec_command(launch)
time.sleep(5); chan.close()
print("launched, waiting 40s to verify stage1 起步...")
time.sleep(40)

# ===== 4. 验证启动 =====
print("\n===== 4. 验证启动 =====")
_, out, _ = cli.exec_command("ps aux|grep 'two_stage'|grep -v grep|head -1|awk '{print $2,$10}'; echo '---log tail---'; tail -12 /root/data/d7_twostage_run.log 2>&1; echo '---GPU---'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1", timeout=30)
print(out.read().decode('utf-8', errors='replace'))
cli.close()
print("\n=== 实验 C 已启动，stage1 8k 步 ~80min + stage2 5k 步 ~50min + eval 15min = ~2.4h ===")
print("=== 查完成: grep -c D7_TWOSTAGE_DONE /root/data/d7_twostage_opt.log ===")
