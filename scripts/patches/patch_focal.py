# Patch run_experiment.py: AQ 用 focal loss (gamma=2), BERT 保持 BCE (gamma=0)
f = "/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"
s = open(f).read()

# 1. finetune 函数签名加 focal_gamma 参数
s = s.replace(
    "def finetune(model, train_ds, val_ds, device, steps, lr=1e-4, bs=256, save_dir=None):",
    "def finetune(model, train_ds, val_ds, device, steps, lr=1e-4, bs=256, save_dir=None, focal_gamma=0.0):"
)

# 2. TrainingConfig 加 focal_gamma
s = s.replace(
    "early_stopping_patience=10000)",
    "early_stopping_patience=10000, focal_gamma=focal_gamma)"
)

# 3. AQ pretrain 调用加 focal_gamma=2
s = s.replace(
    'save_dir=str(EXP/"checkpoints"/f"aq_pretrain_d{d}"))',
    'save_dir=str(EXP/"checkpoints"/f"aq_pretrain_d{d}"), focal_gamma=2.0)'
)

# 4. AQ finetune 调用加 focal_gamma=2
s = s.replace(
    'save_dir=str(EXP/"checkpoints"/f"aq_finetune_d{d}"))',
    'save_dir=str(EXP/"checkpoints"/f"aq_finetune_d{d}"), focal_gamma=2.0)'
)

# BERT finetune 保持默认 (focal_gamma=0, 不改)
open(f, "w").write(s)
print("patched: AQ focal_gamma=2, BERT focal_gamma=0 (default)")
