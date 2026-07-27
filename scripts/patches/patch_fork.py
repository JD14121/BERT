for f in ["/root/beat_mwpm/alphaqubit/training/trainer.py", "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"]:
    s=open(f).read()
    old='num_workers=(self.config.num_workers if shuffle else 0),'
    new="num_workers=(self.config.num_workers if shuffle else 0),\n            multiprocessing_context=('fork' if (shuffle and self.config.num_workers>0) else None),"
    if old in s and 'multiprocessing_context' not in s:
        s=s.replace(old,new)
        open(f,"w").write(s)
        print("patched",f)
    else:
        print("skip",f)
