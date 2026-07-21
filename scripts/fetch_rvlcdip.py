"""
Stream ~N images from RVL-CDIP (documents, grayscale) into a staging dir, without
downloading all 400k. Saves PNGs to scripts/staging/documents/.
"""
import os, argparse
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=11500)
    ap.add_argument("--out", default="scripts/staging/documents")
    ap.add_argument("--repo", default="aharley/rvl_cdip")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset(args.repo, split="train", streaming=True, trust_remote_code=True)
    saved = 0
    for ex in ds:
        img = ex["image"]
        if img.mode != "L":
            img = img.convert("L")
        img.save(os.path.join(args.out, f"doc_{saved:06d}.png"))
        saved += 1
        if saved % 1000 == 0:
            print(f"saved {saved}/{args.n}", flush=True)
        if saved >= args.n:
            break
    print(f"DONE: {saved} images in {args.out}", flush=True)

if __name__ == "__main__":
    main()
