# Patch: switch to CompressedNpyDataset + num_workers=8 for d5 optimized training
import re

# 1. bert_pretrain.py: load_single_npy -> load_compressed_npy
bp = "/root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py"
s = open(bp).read()
s = s.replace("from single_npy_dataset import load_single_npy", "from compressed_npy_dataset import load_compressed_npy")
s = s.replace("load_single_npy(", "load_compressed_npy(")
open(bp, "w").write(s)
print("bert_pretrain: load_single_npy -> load_compressed_npy")

# 2. run_experiment.py: load_single_npy -> load_compressed_npy
rf = "/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"
s = open(rf).read()
s = s.replace("from single_npy_dataset import load_single_npy", "from compressed_npy_dataset import load_compressed_npy")
s = s.replace("load_single_npy(", "load_compressed_npy(")
open(rf, "w").write(s)
print("run_experiment: load_single_npy -> load_compressed_npy")

# 3. trainer.py + pretrain_trainer.py: num_workers 4 -> 8
for f in ["/root/beat_mwpm/alphaqubit/training/trainer.py",
          "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"]:
    s = open(f).read()
    s = s.replace("num_workers: int = 4", "num_workers: int = 8")
    open(f, "w").write(s)
    print(f"patched num_workers 4->8: {f.split('/')[-1]}")

print("ALL PATCHES DONE")
