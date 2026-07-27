"""
Generic domain split builder for the STanH adapter pipeline.

Takes a staging directory of images (searched recursively) and builds the
ImageFolder layout the pipeline expects:

    datasets/<name>/{train,val,test}/data/<symlinks>

matching the X-ray dataset structure. Deterministic random split (seed), disjoint
across train/val/test. Uses symlinks by default (no extra disk). Any PIL-readable
format works (png/jpg/tif), since ImageFolder does Image.open(...).convert("RGB").

GROUP-AWARE SPLITTING (--group_regex / --group_preset)
------------------------------------------------------
The original version shuffled a single pool of FILES, so sibling images of the
same patient/volume/document landed on both sides of the split. Measured impact
(audit of 25/07): 71.5% of X-ray test patients, 67% of OCT test volumes and 33%
of retina test patients also appeared in train+val.

With --group_regex, the shuffle happens over GROUPS and every image of a group
lands in exactly one split, so the partition is disjoint by construction and no
post-hoc filtering (scripts/make_disjoint_eval_sample.py) is needed. Group ids
are parsed from the file's basename:

    xrays   00000181_061.png            -> 00000181     (NIH ChestX-ray patient)
    oct     CNV-1016042-3.jpeg          -> CNV-1016042  (Kermany volume)
    retina  ..._10015_left.jpeg         -> 10015        (EyePACS patient)

Because groups are indivisible, the requested per-split counts are targets, not
guarantees: the builder fills each split with whole groups until adding the next
one would overshoot, then reports the achieved counts. Use --dry_run to see the
achieved sizes without touching anything on disk.

Domains whose provenance ids were discarded at fetch time (documents, dior, rico
-- see scripts/fetch_*.py, which rename to <domain>_%06d.png) have no recoverable
group and must be split without --group_regex; that limitation is a property of
the staging data, not of this script.

Usage:
    # group-aware (preferred whenever a group id exists)
    python scripts/prepare_domain_split.py --src /path/to/images --name xrays \
        --group_preset xrays --n_train 8000 --n_val 1143 --n_test 2287 --seed 42

    # inspect the achieved split without writing
    python scripts/prepare_domain_split.py --src ... --name xrays \
        --group_preset xrays --dry_run

    # ungrouped (only when no group id is recoverable)
    python scripts/prepare_domain_split.py --src /path/to/images --name documents \
        --n_train 8000 --n_val 1000 --n_test 2000 --seed 42
"""
import os, glob, argparse, random, re
from collections import defaultdict

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Group-id regexes for the domains whose provenance survives in the filename.
# group(1) is the patient/volume id. Matched against the BASENAME.
GROUP_PRESETS = {
    "xrays":  r"^(\d+)_",                 # 00000181_061.png -> 00000181
    "oct":    r"^([A-Za-z]+-\d+)-",       # CNV-1016042-3.jpeg -> CNV-1016042
    "retina": r"(\d+)_(?:left|right)",    # ..._10015_left.jpeg -> 10015
}


def find_images(src):
    # case-insensitive extension match (SOCOFing uses .BMP, etc.)
    files = []
    for f in glob.glob(os.path.join(src, "**", "*"), recursive=True):
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in EXTS:
            files.append(f)
    return sorted(set(files))


def split_by_file(files, n_train, n_val, n_test, seed):
    """Legacy behaviour: shuffle files. Only correct when images are independent."""
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    return {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:n_train + n_val + n_test],
    }


def split_by_group(files, rx, n_train, n_val, n_test, seed):
    """Shuffle GROUPS and assign each group whole to one split.

    Greedy: walk the shuffled groups, filling train, then val, then test, adding a
    group only while the split is still below its target. Groups are indivisible,
    so the achieved sizes are <= the targets; the caller reports both.
    """
    groups = defaultdict(list)
    ungrouped = []
    for f in files:
        m = re.search(rx, os.path.basename(f))
        if m:
            groups[m.group(1)].append(f)
        else:
            ungrouped.append(f)
    if ungrouped:
        raise SystemExit(
            f"{len(ungrouped)} files did not match the group regex {rx!r}, e.g. "
            f"{[os.path.basename(x) for x in ungrouped[:3]]}. Fix the regex or drop "
            f"--group_regex (and accept that the split will not be group-disjoint)."
        )
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    targets = [("train", n_train), ("val", n_val), ("test", n_test)]
    out = {name: [] for name, _ in targets}
    leftover = []
    for k in keys:
        for name, target in targets:
            if len(out[name]) + len(groups[k]) <= target:
                out[name].extend(sorted(groups[k]))
                break
        else:
            leftover.append(k)
    return out, groups, leftover


def report_groups(splits, rx):
    """Print per-split group counts and assert the partition is group-disjoint."""
    gid = lambda f: re.search(rx, os.path.basename(f)).group(1)
    gs = {s: {gid(f) for f in items} for s, items in splits.items()}
    for s, items in splits.items():
        print(f"  {s}: {len(items)} images in {len(gs[s])} groups")
    bad = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = gs[a] & gs[b]
        if inter:
            bad.append(f"{a}/{b}: {len(inter)} shared groups")
    if bad:
        raise SystemExit("GROUP LEAK: " + "; ".join(bad))
    print("  group-disjoint across train/val/test: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Staging dir with images (recursive)")
    ap.add_argument("--name", required=True, help="Domain name -> datasets/<name>/")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    ap.add_argument("--group_preset", choices=sorted(GROUP_PRESETS),
                    help=f"Use a known group regex: {', '.join(sorted(GROUP_PRESETS))}")
    ap.add_argument("--group_regex", default=None,
                    help="Regex whose group(1) is the patient/volume id (overrides --group_preset). "
                         "Omit both only when no group id is recoverable from the filenames.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Report the achieved split and exit without creating or deleting anything")
    args = ap.parse_args()

    rx = args.group_regex or (GROUP_PRESETS[args.group_preset] if args.group_preset else None)

    files = find_images(args.src)
    need = args.n_train + args.n_val + args.n_test
    print(f"Found {len(files)} images in {args.src} (need {need})")
    if len(files) < need:
        print(f"WARNING: only {len(files)} images (< {need}); will use what is available.")

    if rx:
        print(f"Group-aware split, regex {rx!r}")
        splits, groups, leftover = split_by_group(files, rx, args.n_train, args.n_val, args.n_test, args.seed)
        print(f"  {len(groups)} groups, median size "
              f"{sorted(len(v) for v in groups.values())[len(groups)//2]}")
        report_groups(splits, rx)
        got = {s: len(v) for s, v in splits.items()}
        for s, target in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
            if got[s] != target:
                print(f"  NOTE {s}: {got[s]} of {target} requested "
                      f"({target - got[s]} short; groups are indivisible)")
        if leftover:
            print(f"  {len(leftover)} groups ({sum(len(groups[k]) for k in leftover)} images) unassigned")
    else:
        print("WARNING: no --group_regex/--group_preset. Splitting by FILE. If the source has "
              "sibling images (same patient/volume/document), they WILL land on both sides.")
        splits = split_by_file(files, args.n_train, args.n_val, args.n_test, args.seed)
        for s, items in splits.items():
            print(f"  {s}: {len(items)} images")

    if args.dry_run:
        print("dry run: nothing written")
        return

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

    # Manifest for reproducibility. Records the path RELATIVE TO --src, not the
    # basename: the retina pool holds two preprocessed versions of the same photo
    # (resized_train/ and resized_train_cropped/) whose basenames are identical, so
    # a basename-only manifest made the duplication invisible.
    with open(os.path.join(root, "split_manifest.txt"), "w") as m:
        m.write(f"# domain={args.name} seed={args.seed} src={args.src} "
                f"group_regex={rx!r}\n")
        for split, items in splits.items():
            for f in items:
                m.write(f"{split}\t{os.path.relpath(f, args.src)}\n")
    print(f"Manifest: {os.path.join(root, 'split_manifest.txt')}")


if __name__ == "__main__":
    main()
