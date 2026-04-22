
# -*- coding: utf-8 -*-
"""
Inspect image dimensions under a directory:
- find minimum width and the files having it
- find minimum height and the files having it

Usage:
  python -m src.inspect_image_dims --config configs/inspect_image_dims.yaml
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import argparse
import sys

try:
    import yaml  # pip install pyyaml
except Exception as e:
    yaml = None


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML not found. Install by: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="YAML cfg path")
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        print(f"[!] Cfg not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    cfg = load_yaml(cfg_path)

    root_dir = Path(cfg["root_dir"])
    img_exts = {s.lower() for s in cfg.get("img_exts", [".png", ".jpg", ".jpeg"])}
    print_files = bool(cfg.get("print_files", True))
    max_print = int(cfg.get("max_print", 50))

    if not root_dir.exists():
        print(f"[!] root_dir not found: {root_dir}", file=sys.stderr)
        sys.exit(2)

    min_w = float("inf")
    min_h = float("inf")
    min_files_w: list[Path] = []
    min_files_h: list[Path] = []

    n_total = 0
    n_ok = 0
    n_fail = 0

    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in img_exts:
            continue

        n_total += 1
        try:
            with Image.open(p) as img:
                w, h = img.size  # (width, height)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[!] 讀取失敗：{p} → {e}", file=sys.stderr)
            continue

        # min width
        if w < min_w:
            min_w = w
            min_files_w = [p]
        elif w == min_w:
            min_files_w.append(p)

        # min height
        if h < min_h:
            min_h = h
            min_files_h = [p]
        elif h == min_h:
            min_files_h.append(p)

    if n_ok == 0:
        print("[!] 沒有成功讀到任何圖片（或資料夾內沒有符合副檔名的圖片）")
        print(f"root_dir: {root_dir}")
        print(f"img_exts: {sorted(img_exts)}")
        sys.exit(1)

    print(f"root_dir: {root_dir}")
    print(f"images scanned: total={n_total}, ok={n_ok}, fail={n_fail}")
    print(f"✦ 最小寬度 (min width) : {int(min_w)}")
    print(f"✦ 最小高度 (min height): {int(min_h)}")

    if print_files:
        print("\n--- files with min width ---")
        for i, fp in enumerate(min_files_w[:max_print], 1):
            print(f"{i:02d}. {fp}")
        if len(min_files_w) > max_print:
            print(f"... ({len(min_files_w) - max_print} more)")

        print("\n--- files with min height ---")
        for i, fp in enumerate(min_files_h[:max_print], 1):
            print(f"{i:02d}. {fp}")
        if len(min_files_h) > max_print:
            print(f"... ({len(min_files_h) - max_print} more)")


if __name__ == "__main__":
    main()
