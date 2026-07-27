# Patch: wrap RNG state restore in try/except (non-critical for training)
f = "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"
s = open(f).read()
old = """        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            if checkpoint['rng_state']['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['rng_state']['cuda'], self.device)
            np.random.set_state(checkpoint['rng_state']['numpy'])
            print('[resume] RNG state restored')"""
new = """        if 'rng_state' in checkpoint:
            try:
                torch.set_rng_state(checkpoint['rng_state']['torch'])
                if checkpoint['rng_state']['cuda'] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(checkpoint['rng_state']['cuda'], self.device)
                np.random.set_state(checkpoint['rng_state']['numpy'])
                print('[resume] RNG state restored')
            except Exception as e:
                print(f'[resume] RNG state restore skipped: {e}')"""
if old in s:
    s = s.replace(old, new)
    open(f, "w").write(s)
    print("patched: RNG restore wrapped in try/except")
else:
    print("pattern not found, checking current code...")
    for i, line in enumerate(s.split('\n')):
        if 'rng_state' in line and 'checkpoint' in line:
            print(f"  line {i+1}: {line.strip()}")
