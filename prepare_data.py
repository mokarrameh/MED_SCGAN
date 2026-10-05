#!/usr/bin/env python3
"""Rebuild the image folders used in the paper from the official dataset downloads.

The manifests in manifests/ list every image used in the paper: split (train/test),
domain (A = WLI, B = NBI), file name, source dataset, path inside the official download,
SHA-256 hash and size in pixels. This script copies each listed file from your copy of
the official download into

    <out>/train/A   <out>/train/B   <out>/test/A   <out>/test/B

which is the layout read by the training and test scripts (--dataroot <out>). With
--layout cut, the folders are named trainA, trainB, testA and testB instead, as expected
by the official CUT code. Every file is checked against its SHA-256 hash. Images are
copied unchanged.

Settings (Table 2 of the paper):
  piccolo_only   PICCOLO only:                   train 1936 WLI / 1164 NBI, test 192 / 141
  combined       PICCOLO + NBI-InfFrames + EAD:  train 2925 WLI / 1363 NBI, test 192 / 141

Download folders (see README, section Data):
  --piccolo_root        folder with train/, validation/ and test/ (each with polyps/),
                        e.g. "piccolo dataset-release0.1"
  --nbi_infframes_root  folder with Fold1/, Fold2/ and Fold3/ (from FRAMES.zip)
  --ead_root            folder with EAD2020_dataType_framesOnly/ (from
                        EndoCV2020_Endoscopy_Artefact_Detection_Segmentation_training_data*.zip)
If a file is not at its recorded path, it is searched for by name under the given folder.
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFESTS = {
    "piccolo_only": HERE / "manifests" / "piccolo_only.csv",
    "combined": HERE / "manifests" / "combined.csv",
}
ROOT_OPTION = {
    "PICCOLO": "piccolo_root",
    "NBI-InfFrames": "nbi_infframes_root",
    "EAD": "ead_root",
}


def sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(bufsize), b""):
            h.update(block)
    return h.hexdigest()


class DatasetRoot:
    """Find files under one dataset download folder."""

    def __init__(self, root):
        self.root = Path(root)
        self._by_name = None

    def candidates(self, name):
        if self._by_name is None:  # index the folder once, on first use
            self._by_name = defaultdict(list)
            for dirpath, _, filenames in os.walk(self.root):
                for n in filenames:
                    self._by_name[n].append(Path(dirpath) / n)
        return self._by_name.get(name, [])

    def locate(self, rel_path, expected_sha256):
        """Return (path, ok). ok is True when the file's SHA-256 matches."""
        recorded = self.root / rel_path
        if recorded.is_file() and sha256(recorded) == expected_sha256:
            return recorded, True
        for c in self.candidates(Path(rel_path).name):
            if c != recorded and sha256(c) == expected_sha256:
                return c, True
        return (recorded if recorded.is_file() else None), False


def out_dir(out, split, domain, layout):
    return out / (f"{split}{domain}" if layout == "cut" else f"{split}/{domain}")


def place(src, dst, mode):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True, choices=sorted(MANIFESTS),
                    help="which training set to build (both use the same PICCOLO test set)")
    ap.add_argument("--out", help="output folder, used as --dataroot (not needed with --check_only)")
    ap.add_argument("--piccolo_root", required=True, help="PICCOLO download folder")
    ap.add_argument("--nbi_infframes_root", help="NBI-InfFrames download folder (combined only)")
    ap.add_argument("--ead_root", help="EAD (EndoCV2020 training data) download folder (combined only)")
    ap.add_argument("--manifest", help="use this manifest CSV instead of manifests/<setting>.csv")
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                    help="how to place the files in --out (default: copy)")
    ap.add_argument("--layout", choices=["code", "cut"], default="code",
                    help="folder names: train/A ... (this repository, default) or trainA ... (CUT)")
    ap.add_argument("--check_only", action="store_true",
                    help="only check that every listed file is present and unchanged; write nothing")
    args = ap.parse_args()

    if not args.check_only and not args.out:
        ap.error("--out is required unless --check_only is given")

    manifest = Path(args.manifest) if args.manifest else MANIFESTS[args.setting]
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{manifest}: {len(rows)} images")

    roots = {}
    for ds in sorted({r["source_dataset"] for r in rows}):
        opt = ROOT_OPTION.get(ds)
        if opt is None:
            sys.exit(f"error: unknown source_dataset '{ds}' in {manifest}")
        folder = getattr(args, opt)
        if not folder:
            ap.error(f"--{opt} is required for this setting ({ds} images)")
        if not Path(folder).is_dir():
            ap.error(f"--{opt}: not a folder: {folder}")
        roots[ds] = DatasetRoot(folder)

    out = Path(args.out) if args.out else None
    missing, changed, conflicts = [], [], []
    placed = already = 0
    for i, r in enumerate(rows, 1):
        src, ok = roots[r["source_dataset"]].locate(r["source_path"], r["sha256"])
        if src is None:
            missing.append(r)
        elif not ok:
            changed.append(r)
        elif out is not None:
            dst = out_dir(out, r["split"], r["domain"], args.layout) / r["file_name"]
            if dst.exists():
                if sha256(dst) == r["sha256"]:
                    already += 1
                else:
                    conflicts.append(dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                place(src, dst, args.mode)
                placed += 1
        if i % 500 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} checked")

    expected = Counter((r["split"], r["domain"]) for r in rows)
    print("\nexpected images per folder:")
    for (split, dom), n in sorted(expected.items()):
        line = f"  {split}/{dom}: {n}"
        if out is not None:
            d = out_dir(out, split, dom, args.layout)
            listed = {r["file_name"] for r in rows if (r["split"], r["domain"]) == (split, dom)}
            present = {p.name for p in d.iterdir()} if d.is_dir() else set()
            line += f"   present: {len(present & listed)}"
            if present - listed:
                line += f"   not in the manifest: {len(present - listed)} (remove them)"
        print(line)

    if out is not None:
        print(f"\nplaced {placed} files ({args.mode}); {already} were already present")
    for label, items in (("missing", missing), ("different content (SHA-256 mismatch)", changed)):
        if items:
            print(f"\n{len(items)} file(s) {label}, for example:")
            for r in items[:10]:
                print(f"  {r['source_dataset']}: {r['source_path']}")
    if conflicts:
        print(f"\n{len(conflicts)} file(s) already in --out with different content (not overwritten):")
        for p in conflicts[:10]:
            print(f"  {p}")
    if missing or changed or conflicts:
        sys.exit(1)
    print("\nall files verified")


if __name__ == "__main__":
    main()
