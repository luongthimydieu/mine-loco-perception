# -*- coding: utf-8 -*-
"""Check that every local asset/link referenced by site/index.html exists.

Usage: python scripts/check_site_links.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SITE = os.path.join(ROOT, "site")


def main():
    index = os.path.join(SITE, "index.html")
    if not os.path.isfile(index):
        print("missing", index)
        return 1
    html = open(index, encoding="utf-8").read()
    links = set(re.findall(r'(?:src|href|poster)="([^"]+)"', html))
    missing = []
    checked = 0
    for link in sorted(links):
        if link.startswith(("http://", "https://", "#", "mailto:")):
            continue
        checked += 1
        if not os.path.exists(os.path.join(SITE, link)):
            missing.append(link)
    print("checked %d local links/assets" % checked)
    if missing:
        print("MISSING:")
        for m in missing:
            print("  ", m)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
