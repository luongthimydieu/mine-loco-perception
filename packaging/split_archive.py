# -*- coding: utf-8 -*-
"""Split a large release zip into <2 GB parts for GitHub Releases.

GitHub Releases allows at most 2 GB per asset. A PyInstaller one-folder build
with PyTorch is often larger, so split it and attach the parts plus SHA256SUMS.

Usage:
  python packaging/split_archive.py dist/MineLoco-windows-x64.zip --part-mb 1900

Users recombine on Windows:
  copy /b MineLoco-windows-x64.zip.001 + MineLoco-windows-x64.zip.002 MineLoco-windows-x64.zip
or with 7-Zip: 7z x MineLoco-windows-x64.zip.001
"""
import argparse
import hashlib
import os


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--part-mb", type=int, default=1900)
    args = ap.parse_args()

    size = os.path.getsize(args.archive)
    part = args.part_mb * 1024 * 1024
    if size <= part:
        print("archive is %.2f GB; no split needed (limit %d MB)" %
              (size / 1e9, args.part_mb))
        # still emit a single .001 so the release asset glob is uniform
        if not os.path.exists(args.archive + ".001"):
            os.replace(args.archive, args.archive + ".001")
        parts = [args.archive + ".001"]
    else:
        parts = []
        with open(args.archive, "rb") as f:
            idx = 1
            while True:
                chunk = f.read(part)
                if not chunk:
                    break
                p = "%s.%03d" % (args.archive, idx)
                with open(p, "wb") as out:
                    out.write(chunk)
                parts.append(p)
                idx += 1
        os.remove(args.archive)

    sums = os.path.join(os.path.dirname(args.archive), "SHA256SUMS.txt")
    with open(sums, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("%s  %s\n" % (sha256(p), os.path.basename(p)))
    print("parts:")
    for p in parts:
        print("  %s  %.2f GB" % (p, os.path.getsize(p) / 1e9))
    print("checksums:", sums)


if __name__ == "__main__":
    main()
