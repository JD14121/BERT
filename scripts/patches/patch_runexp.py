f="/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"
s=open(f).read()
if "load_single_npy" not in s:
    s=s.replace('syn_train=PTBatchDataset(str(syn_pt("train", args.train_n)))','from single_npy_dataset import load_single_npy; syn_train=load_single_npy(d, r, basis, DATA_DIR)')
    open(f,"w").write(s)
    print("run_experiment patched (syn_train -> load_single_npy)")
else:
    print("run_experiment already patched")
