
# -*- coding: utf-8 -*-
"""
Generate 'blank' class images (pure white RGBA) into:
  {base_dir}/{symbol}/{bg}/blank_XX.png

- Designed to replace a Jupyter cell with a reproducible .py + yaml config.
- By default, keeps your existing folder convention (NO opacity subdir).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

try:
    import yaml  # pip install pyyaml
except Exception as e:
    yaml = None


@dataclass
class Config:
    base_dir: str
    symbol: str
    num_images: int
    target_hw: tuple[int, int]
    opacities: tuple[int, ...]
    bg_ids: tuple[int, ...]
    rgba_value: tuple[int, int, int, int]
    filename_digits: int
    overwrite: bool
    make_json: bool
    json_template: dict
    use_opacity_subdir: bool


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML not installed. Please run: pip install pyyaml\n"
            f"Config path: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_config(d: dict) -> Config:
    base_dir = d["base_dir"]
    symbol = d.get("symbol", "blank")
    num_images = int(d.get("num_images", 280))

    hw = d.get("target_hw", [68, 110])
    if not (isinstance(hw, (list, tuple)) and len(hw) == 2):
        raise ValueError(f"target_hw must be [H, W], got: {hw}")
    target_hw = (int(hw[0]), int(hw[1]))

    opacities = tuple(int(x) for x in d.get("opacities", [25]))
    bg_ids = tuple(int(x) for x in d.get("bg_ids", [0]))

    rgba = d.get("rgba_value", [255, 255, 255, 255])
    if not (isinstance(rgba, (list, tuple)) and len(rgba) == 4):
        raise ValueError(f"rgba_value must be [R,G,B,A], got: {rgba}")
    rgba_value = tuple(int(x) for x in rgba)

    filename_digits = int(d.get("filename_digits", 2))
    overwrite = bool(d.get("overwrite", True))
    make_json = bool(d.get("make_json", False))
    json_template = dict(d.get("json_template", {"category": symbol, "note": "dummy"}))
    use_opacity_subdir = bool(d.get("use_opacity_subdir", False))

    return Config(
        base_dir=base_dir,
        symbol=symbol,
        num_images=num_images,
        target_hw=target_hw,
        opacities=opacities,
        bg_ids=bg_ids,
        rgba_value=rgba_value,
        filename_digits=filename_digits,
        overwrite=overwrite,
        make_json=make_json,
        json_template=json_template,
        use_opacity_subdir=use_opacity_subdir,
    )


def build_out_dir(base_dir: Path, symbol: str, bg: int, op: int, use_opacity_subdir: bool) -> Path:
    # 保持你現在的資料結構：base/symbol/bg/
    # 若未來要分 opacity，再開 use_opacity_subdir：base/symbol/op/bg/
    if use_opacity_subdir:
        return base_dir / symbol / f"{op}" / f"{bg}"
    return base_dir / symbol / f"{bg}"


def main(yaml_obj=None):
    if yaml_obj is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfg", type=str, required=True, help="Path to YAML cfg.")
        args = ap.parse_args()

        cfg_path = Path(args.cfg)
        if not cfg_path.exists():
            print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)

        d = load_yaml(cfg_path)
    else:
        d = yaml_obj
    cfg = to_config(d)

    base_dir = Path(cfg.base_dir)
    if not base_dir.exists():
        print(f"[WARN] base_dir not found, will still create subfolders: {base_dir}")

    H, W = cfg.target_hw
    rgba = cfg.rgba_value
    white_img = np.full((H, W, 4), rgba, dtype=np.uint8)  # RGBA

    total = 0
    for op in cfg.opacities:
        for bg in cfg.bg_ids:
            out_dir = build_out_dir(base_dir, cfg.symbol, bg, op, cfg.use_opacity_subdir)
            out_dir.mkdir(parents=True, exist_ok=True)

            for i in range(cfg.num_images):
                fn = f"{cfg.symbol}_{i:0{cfg.filename_digits}d}.png"
                out_path = out_dir / fn

                if out_path.exists() and not cfg.overwrite:
                    continue

                Image.fromarray(white_img).save(out_path)

                if cfg.make_json:
                    j = dict(cfg.json_template)
                    # 若沒填 category，就補上
                    j.setdefault("category", cfg.symbol)
                    json_path = out_dir / f"{cfg.symbol}_{i:0{cfg.filename_digits}d}.json"
                    if (not json_path.exists()) or cfg.overwrite:
                        with json_path.open("w", encoding="utf-8") as f:
                            json.dump(j, f, ensure_ascii=False, indent=2)

                total += 1

    print("✅ Done!")
    print(f"base_dir: {base_dir}")
    print(f"symbol: {cfg.symbol}")
    print(f"target_hw: {cfg.target_hw} (H,W)")
    print(f"opacities(loop): {cfg.opacities} | use_opacity_subdir={cfg.use_opacity_subdir}")
    print(f"bg_ids: {cfg.bg_ids}")
    print(f"generated(or overwritten) files: {total}")

if __name__ == "__main__":
    main()
