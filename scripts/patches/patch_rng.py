# Patch trainer + pretrain_trainer: add RNG state to checkpoint (for future runs)
import torch, numpy as np

files = [
    "/root/beat_mwpm/alphaqubit/training/trainer.py",
    "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py",
]
for f in files:
    s = open(f).read()
    if "rng_state" in s:
        print("skip (already patched)", f)
        continue
    # add rng_state to save checkpoint dict (before 'history')
    s = s.replace(
        "            'history': self.history,\n        }",
        "            'history': self.history,\n            'rng_state': {\n                'torch': torch.get_rng_state(),\n                'cuda': torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None,\n                'numpy': np.random.get_state(),\n            },\n        }"
    )
    # add rng restore to load_checkpoint (after best_val_loss restore)
    s = s.replace(
        "        self.best_val_loss = checkpoint['best_val_loss']",
        "        self.best_val_loss = checkpoint['best_val_loss']\n        if 'rng_state' in checkpoint:\n            torch.set_rng_state(checkpoint['rng_state']['torch'])\n            if checkpoint['rng_state']['cuda'] is not None and torch.cuda.is_available():\n                torch.cuda.set_rng_state(checkpoint['rng_state']['cuda'], self.device)\n            np.random.set_state(checkpoint['rng_state']['numpy'])\n            print('[resume] RNG state restored')"
    )
    open(f, "w").write(s)
    print("patched", f)
