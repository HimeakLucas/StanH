"""
Generic domain split builder for the STanH adapter pipeline.

Takes a staging directory of images (searched recursively) and builds the
ImageFolder layout the pipeline expects:

    datasets/<name>/{train,val,test}/data/<symlinks>

matching the X-ray dataset structure. Deterministic random split (seed), disjoint
across train/val/test. Uses symlinks by default (no extra disk). Any PIL-readable
format works (png/jpg/tif), since ImageFolder does Image.open(...).convert("RGB").

Usage:
    python scripts/prepare_domain_split.py --src /path/to/images --name documents \
        --n_train 8000 --n_val 1000 --n_test 2000 --seed 42
"""
import os, glob, argparse, random

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_images(src):
    # case-insensitive extension match (SOCOFing uses .BMP, etc.)
    files = []
    for f in glob.glob(os.path.join(src, "**", "*"), recursive=True):
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in EXTS:
            files.append(f)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Staging dir with images (recursive)")
    ap.add_argument("--name", required=True, help="Domain name -> datasets/<name>/")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = ap.parse_args()

    files = find_images(args.src)
    need = args.n_train + args.n_val + args.n_test
    print(f"Found {len(files)} images in {args.src} (need {need})")
    if len(files) < need:
        print(f"WARNING: only {len(files)} images (< {need}); will use what is available.")
    random.seed(args.seed)
    random.shuffle(files)

    splits = {
        "train": files[:args.n_train],
        "val":   files[args.n_train:args.n_train + args.n_val],
        "test":  files[args.n_train + args.n_val:args.n_train + args.n_val + args.n_test],
    }
    root = os.path.join("datasets", args.name)
    for split, items in splits.items():
        d = os.path.join(root, split, "data")
        os.makedirs(d, exist_ok=True)
        for x in glob.glob(os.path.join(d, "*")):  # clear stale
            os.unlink(x)
        for f in items:
            # keep a flat unique name (some sets repeat basenames across subdirs)
            rel = os.path.relpath(f, args.src).replace(os.sep, "__")
            dst = os.path.join(d, rel)
            if args.copy:
                import shutil; shutil.copyfile(f, dst)
            else:
                os.symlink(os.path.abspath(f), dst)
        print(f"  {split}: {len(items)} -> {d}")

    # manifest for reproducibility
    with open(os.path.join(root, "split_manifest.txt"), "w") as m:
        m.write(f"# domain={args.name} seed={args.seed} src={args.src}\n")
        for split, items in splits.items():
            for f in items:
                m.write(f"{split}\t{os.path.basename(f)}\n")
    print(f"Manifest: {os.path.join(root, 'split_manifest.txt')}")


if __name__ == "__main__":
    main()
