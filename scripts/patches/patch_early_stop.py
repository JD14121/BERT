# Patch: add min_steps to TrainingConfig + early stopping logic
# + add min_steps/patience params to finetune() function

# 1. trainer.py: add min_steps to TrainingConfig + modify early stop check
f = "/root/beat3wpm/alphaqubit/training/trainer.py"
f = "/root/beat_mwpm/alphaqubit/training/trainer.py"
s = open(f).read()

# Add min_steps to TrainingConfig (after early_stopping_patience)
if "min_steps" not in s:
    s = s.replace(
        "    early_stopping_patience: int = 10  # 早停耐心值（eval次数）",
        "    early_stopping_patience: int = 10  # 早停耐心值（eval次数）\n    min_steps: int = 0  # 最小步数（此之前不允许早停）"
    )
    print("added min_steps to TrainingConfig")

# Modify early stop check: add min_steps condition
s = s.replace(
    "                if self.patience_counter >= self.config.early_stopping_patience:",
    "                if step >= self.config.min_steps and self.patience_counter >= self.config.early_stopping_patience:"
)
print("patched early stop check with min_steps guard")

open(f, "w").write(s)

# 2. pretrain_trainer.py: same changes
f2 = "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"
s2 = open(f2).read()
if "min_steps" not in s2:
    s2 = s2.replace(
        "    early_stopping_patience: int = 20  # 早停耐心值",
        "    early_stopping_patience: int = 20  # 早停耐心值\n    min_steps: int = 0  # 最小步数（此之前不允许早停）"
    )
    print("added min_steps to PretrainConfig")
s2 = s2.replace(
    "                if self.patience_counter >= self.config.early_stopping_patience:",
    "                if step >= self.config.min_steps and self.patience_counter >= self.config.early_stopping_patience:"
)
print("patched pretrain early stop check with min_steps guard")
open(f2, "w").write(s2)

# 3. run_experiment.py: add min_steps + patience params to finetune() + set per-stage values
f3 = "/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"
s3 = open(f3).read()

# Add params to finetune() signature
s3 = s3.replace(
    "def finetune(model, train_ds, val_ds, device, steps, lr=1e-4, bs=256, save_dir=None, focal_gamma=0.0):",
    "def finetune(model, train_ds, val_ds, device, steps, lr=1e-4, bs=256, save_dir=None, focal_gamma=0.0, min_steps=0, patience=10000):"
)

# Add min_steps + patience to TrainingConfig in finetune()
s3 = s3.replace(
    "early_stopping_patience=10000, focal_gamma=focal_gamma)",
    "early_stopping_patience=patience, focal_gamma=focal_gamma, min_steps=min_steps)"
)

# AQ pretrain: min_steps=10000, patience=10
s3 = s3.replace(
    'save_dir=str(EXP/"checkpoints"/f"aq_pretrain_d{d}"), focal_gamma=2.0)',
    'save_dir=str(EXP/"checkpoints"/f"aq_pretrain_d{d}"), focal_gamma=2.0, min_steps=10000, patience=10)'
)

# AQ finetune: min_steps=2000, patience=8
s3 = s3.replace(
    'save_dir=str(EXP/"checkpoints"/f"aq_finetune_d{d}"), focal_gamma=2.0)',
    'save_dir=str(EXP/"checkpoints"/f"aq_finetune_d{d}"), focal_gamma=2.0, min_steps=2000, patience=8)'
)

# BERT finetune: min_steps=2000, patience=10 (default patience=10000 -> change to 10)
s3 = s3.replace(
    'save_dir=str(EXP/"checkpoints"/f"bert_finetune_d{d}"))',
    'save_dir=str(EXP/"checkpoints"/f"bert_finetune_d{d}"), min_steps=2000, patience=10)'
)

open(f3, "w").write(s3)
print("patched run_experiment: finetune() with min_steps + patience per stage")
print("\nEarly stopping config:")
print("  AQ pretrain:  min_steps=10000, patience=10 (earliest stop: step 15000/20000)")
print("  AQ finetune:  min_steps=2000,  patience=8  (earliest stop: step 6000/8000)")
print("  BERT finetune: min_steps=2000, patience=10 (earliest stop: step 7000/8000)")
