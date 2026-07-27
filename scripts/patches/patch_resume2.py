f = "/root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py"
s = open(f).read()
if "--resume" not in s.split("add_argument")[0] and "ap.add_argument('--resume'" not in s:
    # Insert --resume before --use-round-mask (which uses single quotes)
    old = "    ap.add_argument('--use-round-mask'"
    new = "    ap.add_argument('--resume', action='store_true', help='Resume from best.pt')\n" + old
    s = s.replace(old, new, 1)
    open(f, "w").write(s)
    print("ADDED --resume argument")
else:
    print("--resume already exists")
