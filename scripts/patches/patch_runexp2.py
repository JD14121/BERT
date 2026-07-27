f="/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"
s=open(f).read()
old = "syn_train=PTBatchDataset(str(syn_pt('train', args.train_n)))"
new = "from single_npy_dataset import load_single_npy; syn_train=load_single_npy(d, r, basis, DATA_DIR)"
if old in s:
    s = s.replace(old, new)
    open(f,"w").write(s)
    print("PATCHED OK")
elif "load_single_npy" in s:
    print("already patched")
else:
    print("old line not found, current syn_train lines:")
    for i,line in enumerate(s.split('\n')):
        if 'syn_train' in line:
            print(f"  line {i+1}: {line.strip()}")
