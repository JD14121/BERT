# Patch trainer + pretrain_trainer: use ShardShuffleSampler for memmap datasets
for f in ["/root/beat_mwpm/alphaqubit/training/trainer.py",
          "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"]:
    s = open(f).read()
    if "ShardShuffleSampler" not in s:
        # Add shard shuffle logic before the return DataLoader line
        old = "        return DataLoader("
        new = """        # Shard-level shuffle for memmap datasets (sequential within shard -> OS prefetch)
        if shuffle and getattr(dataset, 'is_memmap_dataset', False):
            from compressed_npy_dataset import ShardShuffleSampler
            sampler = ShardShuffleSampler(len(dataset), shard_size=1_000_000)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=(self.config.num_workers if shuffle else 0),
                pin_memory=True if self.device.type == 'cuda' else False,
                multiprocessing_context=('fork' if (shuffle and self.config.num_workers > 0) else None),
                persistent_workers=True if (shuffle and self.config.num_workers > 0) else False,
            )
        return DataLoader("""
        s = s.replace(old, new, 1)  # only first occurrence
        open(f, "w").write(s)
        print(f"patched ShardShuffleSampler: {f.split('/')[-1]}")
    else:
        print(f"already patched: {f.split('/')[-1]}")
