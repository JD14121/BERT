# Patch: enable checkpoint resume
# 1. pretrain_trainer.py: train loop starts from global_step
f1 = "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"
s = open(f1).read()
s = s.replace(
    "        for step in range(self.config.total_steps):",
    "        for step in range(self.global_step, self.config.total_steps):"
)
open(f1, "w").write(s)
print("patched pretrain_trainer: range(global_step, total_steps)")

# 2. bert_pretrain.py: add --resume flag + load_checkpoint
f2 = "/root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py"
s = open(f2).read()
# Add --resume arg
s = s.replace(
    '    ap.add_argument("--use-round-mask", action="store_true",',
    '    ap.add_argument("--resume", action="store_true", help="Resume from best.pt checkpoint")\n    ap.add_argument("--use-round-mask", action="store_true",'
)
# Add load_checkpoint before trainer.train()
s = s.replace(
    "    trainer.masking = masking",
    "    if args.resume:\n"
    "        ckpt_path = save_dir / 'best.pt'\n"
    "        if ckpt_path.exists():\n"
    "            trainer.load_checkpoint(str(ckpt_path))\n"
    "            print(f'[RESUME] Loaded checkpoint from {ckpt_path}, continuing from step {trainer.global_step}')\n"
    "        else:\n"
    "            print(f'[RESUME] No checkpoint found at {ckpt_path}, starting from scratch')\n"
    "    trainer.masking = masking"
)
open(f2, "w").write(s)
print("patched bert_pretrain: --resume flag + load_checkpoint")
