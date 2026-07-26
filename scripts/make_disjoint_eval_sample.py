"""
Leakage-free evaluation sample: like make_eval_sample.py, but restricted to test
images whose patient/volume GROUP appears in neither the training nor the
validation split. Validation selects the checkpoint, so a group seen there leaks
into the reported number too.

prepare_domain_split.py shuffles a single pool, so sibling images land on both
sides: adjacent OCT B-scans of the same volume, or both eyes of the same patient.
Those are near-duplicates, and they inflate the target-domain gain of any
fine-tuned model (it effectively saw the test image during training). Cross-domain
numbers (Kodak) are unaffected, since Kodak never enters training.

Group id is parsed from the filename:
  oct     CNV-1016042-3.jpeg                      -> CNV-1016042   (volume)
  retina  ..._10015_left.jpeg                     -> 10015         (patient)

Usage:
  python scripts/make_disjoint_eval_sample.py --domain oct
  python scripts/make_disjoint_eval_sample.py --domain retina --n 150
"""
import os, re, glob, random, argparse

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

PRESETS = {
    # domain: (root, regex capturing the group id)
    "oct":    ("datasets/oct",    r"^([A-Za-z]+-\d+)-"),
    "retina": ("datasets/retina", r"(\d+)_(?:left|right)"),
}


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))


def group_of(name, rx):
    m = re.search(rx, name)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(PRESETS), help="Preset root + group regex")
    ap.add_argument("--root", default=None, help="datasets/<domain> (overrides preset)")
    ap.add_argument("--group_regex", default=None, help="Regex whose group(1) is the patient/volume id")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", default="sample_eval_disjoint_manifest.txt")
    ap.add_argument("--sample_dir", default="sample_eval_disjoint")
    args = ap.parse_args()

    root, rx = (PRESETS[args.domain] if args.domain else (None, None))
    root = args.root or root
    rx = args.group_regex or rx
    if not root or not rx:
        raise SystemExit("need --domain, or both --root and --group_regex")

    train_dir = os.path.join(root, "train", "data")
    val_dir = os.path.join(root, "val", "data")
    test_dir = os.path.join(root, "test", "data")
    out_dir = os.path.join(root, "test")
    manifest = os.path.join(out_dir, args.manifest)
    sample_dir = os.path.join(out_dir, args.sample_dir)

    fitted = list_images(train_dir) + (list_images(val_dir) if os.path.isdir(val_dir) else [])
    fitted_groups = {g for g in (group_of(f, rx) for f in fitted) if g}
    test_files = list_images(test_dir)
    ungrouped = [f for f in test_files if group_of(f, rx) is None]
    if ungrouped:
        raise SystemExit(f"{len(ungrouped)} test files did not match the regex, e.g. {ungrouped[:3]}")

    clean = [f for f in test_files if group_of(f, rx) not in fitted_groups]
    test_groups = {group_of(f, rx) for f in test_files}
    clean_groups = {group_of(f, rx) for f in clean}
    leaked_groups = test_groups - clean_groups
    print(f"[{root}] test images {len(test_files)} in {len(test_groups)} groups")
    print(f"  groups also seen in train/val (LEAKED): {len(leaked_groups)} "
          f"({100*len(leaked_groups)/max(len(test_groups),1):.0f}%)")
    print(f"  clean images available: {len(clean)} in {len(clean_groups)} groups")

    if os.path.exists(manifest):
        names = [l.strip() for l in open(manifest) if l.strip()]
        print(f"Reusing existing manifest {manifest} ({len(names)} images)")
    else:
        n = min(args.n, len(clean))
        if n < args.n:
            print(f"  WARNING: only {n} clean images (< {args.n}); reporting on n={n}")
        names = sorted(random.Random(args.seed).sample(clean, n))
        with open(manifest, "w") as f:
            f.write("\n".join(names) + "\n")
        print(f"Wrote manifest {manifest} ({len(names)} of {len(clean)} clean, seed {args.seed})")

    os.makedirs(sample_dir, exist_ok=True)
    for old in glob.glob(os.path.join(sample_dir, "*")):
        os.remove(old)
    for name in names:
        os.symlink(os.path.abspath(os.path.join(test_dir, name)),
                   os.path.join(sample_dir, name))
    print(f"Sample dir {sample_dir}: {len(names)} symlinks")


if __name__ == "__main__":
    main()
