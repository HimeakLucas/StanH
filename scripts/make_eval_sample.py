"""
Fixed random evaluation sample for a domain test set (manifest + symlink dir).

Evaluating on the FIRST N images sorted by name is biased (measured ~0.5 dB
optimistic on X-ray; document names follow the HF streaming order, which may
correlate with class). This builds a reproducible random sample instead:

  - if <test_dir>/<manifest> exists, REUSES it (never resamples silently), so
    re-running after a migration regenerates the same symlinks;
  - otherwise draws N names from <test_dir>/data with the given seed and writes
    the manifest;
  - (re)creates <test_dir>/<sample_dirname>/ with symlinks to the sampled images.

Usage:
  python scripts/make_eval_sample.py --test_dir datasets/documents/test
  python scripts/make_eval_sample.py --test_dir datasets/xrays/test \
      --manifest sample_consolidation_manifest.txt --sample_dir sample_consolidation
"""
import os, argparse, random, glob

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True, help="e.g. datasets/<domain>/test (contains data/)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", default="sample_eval_manifest.txt")
    ap.add_argument("--sample_dir", default="sample_eval")
    args = ap.parse_args()

    data_dir = os.path.join(args.test_dir, "data")
    manifest = os.path.join(args.test_dir, args.manifest)
    sample_dir = os.path.join(args.test_dir, args.sample_dir)

    if os.path.exists(manifest):
        names = [l.strip() for l in open(manifest) if l.strip()]
        print(f"Reusing existing manifest {manifest} ({len(names)} images)")
    else:
        files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(EXTS))
        if len(files) < args.n:
            raise SystemExit(f"only {len(files)} images in {data_dir}, need {args.n}")
        names = sorted(random.Random(args.seed).sample(files, args.n))
        with open(manifest, "w") as f:
            f.write("\n".join(names) + "\n")
        print(f"Wrote manifest {manifest} ({len(names)} of {len(files)} images, seed {args.seed})")

    os.makedirs(sample_dir, exist_ok=True)
    for old in glob.glob(os.path.join(sample_dir, "*")):
        os.remove(old)
    missing = 0
    for name in names:
        src = os.path.abspath(os.path.join(data_dir, name))
        if not os.path.exists(src):  # symlink target must resolve
            missing += 1; print(f"  WARN missing: {name}"); continue
        os.symlink(src, os.path.join(sample_dir, name))
    print(f"Sample dir {sample_dir}: {len(names) - missing} symlinks" + (f" ({missing} MISSING)" if missing else ""))


if __name__ == "__main__":
    main()
