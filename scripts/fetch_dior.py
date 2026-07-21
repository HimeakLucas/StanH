"""
Stream ~N images from DIOR (remote-sensing / aerial, RGB 800x800) into a staging
dir, without downloading the whole set. Saves PNGs to scripts/staging/dior/.
Falls through the available splits (train/validation/test) until N are collected.
"""
import os, argparse
from datasets import load_dataset, get_dataset_split_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=11500)
    ap.add_argument("--out", default="scripts/staging/dior")
    ap.add_argument("--repo", default="HichTala/dior")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        splits = get_dataset_split_names(args.repo)
    except Exception:
        splits = ["train", "validation", "test"]
    print(f"splits: {splits}", flush=True)

    saved = 0
    for split in splits:
        if saved >= args.n:
            break
        try:
            ds = load_dataset(args.repo, split=split, streaming=True)
        except Exception as e:
            print(f"  skip split {split}: {type(e).__name__}", flush=True)
            continue
        for ex in ds:
            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(os.path.join(args.out, f"dior_{saved:06d}.png"))
            saved += 1
            if saved % 1000 == 0:
                print(f"saved {saved}/{args.n}", flush=True)
            if saved >= args.n:
                break
    print(f"DONE: {saved} images in {args.out}", flush=True)


if __name__ == "__main__":
    main()
