#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""launch_55m.py
云端 55M 全流程启动脚本（预训练 -> 两阶段 -> eval_ler）。
链式 nohup，自动串行。

策略优化总结：
1. bs64（V100-32GB 显存约束，55M d7 ~17GB）
2. 两阶段模态微调（12M 实验证最优）
3. Stage 1: 80k 深度对齐, dropout 0.1, patience 10
4. Stage 2: dropout 0.2 防过拟合, lr 5e-5 稳定, patience 10
5. 预训练 30k 步, patience 15
6. 50% 合成掺杂
7. eval_ler 传 55M 配置（避免维度不匹配）

用法：
  python launch_55m.py --distance 7
  python launch_55m.py --distance 5
"""
import paramiko, time, sys

HOST, PORT, USER, PWD = '180.127.11.177', 24112, 'root', 'aiNg4ahp'
BACKUP_HOST, BACKUP_PORT = '223.109.239.36', 24112

def connect():
    for host, port, label in [(HOST, PORT, '电信'), (BACKUP_HOST, BACKUP_PORT, '移动')]:
        try:
            cli = paramiko.SSHClient(); cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(host, port=port, username=USER, password=PWD, timeout=20)
            print(f"[{label}] CONNECTED"); return cli
        except Exception as e:
            print(f"[{label}] {type(e).__name__}: {str(e)[:60]}")
    print("SSH FAIL"); sys.exit(1)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, default=7, choices=[5, 7])
    ap.add_argument('--embed-dim', type=int, default=448)
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--num-transformer-layers', type=int, default=6)
    ap.add_argument('--num-readout-layers', type=int, default=8)
    ap.add_argument('--pretrain-steps', type=int, default=30000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--grad-accum', type=int, default=4, help='bs64×4 or bs128×2 = bs256')
    ap.add_argument('--stage2-steps', type=int, default=None, help='默认 d7=5000, d5=8000')
    ap.add_argument('--stage2-dropout', type=float, default=None, help='默认 d7=0.2, d5=0.15')
    args = ap.parse_args()

    # d5 专用优化（n_stab=24 比 d7 的 48 小，可用更大 bs + 更少累积）
    if args.distance == 5:
        if args.batch_size == 64:  # 未手动指定，用 d5 默认
            args.batch_size = 128
        if args.grad_accum == 4:  # 未手动指定
            args.grad_accum = 2
        if args.stage2_steps is None:
            args.stage2_steps = 8000  # d5 有 160k 真机，需更多步
        if args.stage2_dropout is None:
            args.stage2_dropout = 0.15  # 160k 不易过拟合
    else:  # d7
        if args.stage2_steps is None:
            args.stage2_steps = 5000
        if args.stage2_dropout is None:
            args.stage2_dropout = 0.2

    d = args.distance
    EMBED, HEADS, TLAY, RLAY = args.embed_dim, args.n_heads, args.num_transformer_layers, args.num_readout_layers
    PY = "/root/miniconda3/envs/quantum_env/bin/python"

    cli = connect()

    # 1. 上传 two_stage_55m.py
    sftp = cli.open_sftp()
    sftp.put(r'C:\Users\Administrator\two_stage_55m.py', '/root/two_stage_55m.py')
    sftp.close()
    print("uploaded two_stage_55m.py")

    # 2. 预检
    _, o, _ = cli.exec_command(
        f"nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1; "
        f"ls /root/data/d{'7' if d==7 else '5'}/npy_compressed/meta.json 2>&1 || ls /root/data/d{d}/*.pt 2>&1 | head -3; "
        f"ls /root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py 2>&1", timeout=30)
    print("预检:", o.read().decode('utf-8', 'replace'))

    # 3. 链式 nohup
    launch = f"""cd /root/beat_mwpm/google_paems_data/bert_experiment && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
nohup bash -c 'set -o pipefail; \
echo "=== 1. 预训练 55M ({args.pretrain_steps} 步, bs{args.batch_size}) ===" && \
PYTHONIOENCODING=utf-8 {PY} -B -u bert_pretrain.py \
  --distance {d} --embed-dim {EMBED} --n-heads {HEADS} --num-transformer-layers {TLAY} \
  --steps {args.pretrain_steps} --batch-size {args.batch_size} --lr 1e-4 --mask-ratio 0.25 \
  --grad-accum {args.grad_accum} \
  --save-dir checkpoints/bert_pretrain_d{d}_55m \
  2>&1 | tee /root/data/d{d}_55m_pretrain.log && \
echo "=== 2. 两阶段微调 55M ===" && \
PYTHONIOENCODING=utf-8 {PY} -B -u /root/two_stage_55m.py \
  --distance {d} --embed-dim {EMBED} --n-heads {HEADS} \
  --num-transformer-layers {TLAY} --num-readout-layers {RLAY} \
  --batch-size {args.batch_size} \
  --grad-accum {args.grad_accum} \
  --stage2-steps {args.stage2_steps} \
  --stage2-dropout {args.stage2_dropout} \
  --pretrain-dir checkpoints/bert_pretrain_d{d}_55m \
  2>&1 | tee /root/data/d{d}_55m_twostage.log && \
echo "=== 3. eval_ler 55M ===" && \
PYTHONIOENCODING=utf-8 {PY} -B -u eval_ler.py \
  --distances {d} --ft-suffix _55m \
  --embed-dim {EMBED} --n-heads {HEADS} \
  --num-transformer-layers {TLAY} --num-readout-layers {RLAY} \
  2>&1 | tee /root/data/d{d}_55m_eval_ler.log && \
cp results_ler_d{d}.json results_ler_d{d}_55m.json && \
echo D{d}_55M_DONE' > /root/data/d{d}_55m_opt.log 2>&1 < /dev/null &
"""
    chan = cli.get_transport().open_session()
    chan.exec_command(launch)
    time.sleep(5); chan.close()
    print(f"launched d{d} 55M pipeline, waiting 60s to verify...")
    time.sleep(60)

    # 4. 验证启动
    _, o, _ = cli.exec_command(
        f"ps aux|grep 'bert_pretrain.*{d}'|grep -v grep|head -1|awk '{{print $2,$10}}'; "
        f"echo '---pretrain log---'; tail -8 /root/data/d{d}_55m_pretrain.log 2>&1; "
        f"echo '---GPU---'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1", timeout=30)
    print(o.read().decode('utf-8', 'replace'))
    cli.close()

    print(f"\n=== d{d} 55M 全流程已启动 ===")
    print(f"配置: embed={EMBED} heads={HEADS} T={TLAY} R={RLAY} = ~{54}M params")
    print(f"预训练: {args.pretrain_steps} 步, bs{args.batch_size}×{args.grad_accum}=有效bs{args.batch_size*args.grad_accum}")
    print(f"Stage 2: {args.stage2_steps} 步, dropout={args.stage2_dropout}")
    print(f"查完成: grep -c D{d}_55M_DONE /root/data/d{d}_55m_opt.log")

if __name__ == '__main__':
    main()
