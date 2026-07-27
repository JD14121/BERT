# patch bert_pretrain + run_experiment: train -> load_single_npy; trainer num_workers
bp="/root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py"
s=open(bp).read()
if "load_single_npy" not in s:
    s=s.replace('    train_ds = PTBatchDataset(str(train_pt))','    from single_npy_dataset import load_single_npy; train_ds = load_single_npy(d, r, basis, DATA_DIR)')
    s=s.replace('train={train_pt.name} val={val_pt.name}','val={val_pt.name} train=single_npy')
    open(bp,"w").write(s); print("bert_pretrain patched")
else: print("bert_pretrain already")
rf="/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"
s=open(rf).read()
if "load_single_npy" not in s:
    s=s.replace('syn_train=PTBatchDataset(str(syn_pt("train", args.train_n)))','from single_npy_dataset import load_single_npy; syn_train=load_single_npy(d, r, basis, DATA_DIR)')
    open(rf,"w").write(s); print("run_experiment patched")
else: print("run_experiment already")
for f in ["/root/beat_mwpm/alphaqubit/training/trainer.py","/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"]:
    s=open(f).read()
    s=s.replace('num_workers=self.config.num_workers,','num_workers=(self.config.num_workers if shuffle else 0),')
    s=s.replace('num_workers: int = 0','num_workers: int = 4')
    open(f,"w").write(s)
print("trainer patched (eval nw=0, train default 4)")
