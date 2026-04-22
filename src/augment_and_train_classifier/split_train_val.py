
"""
Train/Val split with deterministic, per-class exact ratio, and stratification by recipe tags.

Behavior aligned to your notebook:
- special tops (BGC/blank/bg_color): output to <split>/<label>/_root/<filename>
- normal: rel=(MAP, CLASS, ...FILE) -> <split>/<CLASS>/<MAP>/<rest...>
- per-class val count = int(round(n * val_ratio))  (banker's rounding same as Python round)
- stratify by (bg, op, nz_mode, nz_bin) parsed from filename
- allocate per-recipe val counts by Largest Remainder to match desired_val exactly
- deterministic selection by md5(relative_path)
- write meta.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # pip install pyyaml
except Exception as e:
    raise RuntimeError("Missing dependency: pyyaml. Please `pip install pyyaml`.") from e

try:
    from tqdm import tqdm  # pip install tqdm
except Exception:
    tqdm = None  # allow no tqdm


# -----------------------------
# Regex / parsing helpers
# -----------------------------
_re_bg = re.compile(r"__bg-([A-Za-z0-9]+)")
_re_op = re.compile(r"__op-(\d{3})")
_re_strip_tail_idx = re.compile(r"(?:__|_)\d+$")


def _strip_tail_index(s: str) -> str:
    return _re_strip_tail_idx.sub("", s)


def parse_recipe_from_name(p: Path) -> Tuple[str, str, str, str]:
    """
    Return recipe key: (bg_code, op_code, nz_mode, nz_bin)

    nz_bin supports:
    - legacy: le25 / ge25 / none / unk
    - new: single / none / unk
    """
    stem = p.stem

    m_bg = _re_bg.search(stem)
    bg = m_bg.group(1) if m_bg else "UNK"

    m_op = _re_op.search(stem)
    op = m_op.group(1) if m_op else "UNK"

    nz_mode = "UNK"
    nz_bin = "unk"

    if "__nz-" in stem:
        nz_part = stem.split("__nz-", 1)[1]
        nz_tag = _strip_tail_index(nz_part)

        if nz_tag:
            nz_mode = nz_tag[0]  # N/L/R...

        if nz_mode == "N":
            nz_bin = "none"
        elif nz_mode in ("L", "R"):
            # legacy: L25-xx / R50-xx
            bin_code = nz_tag[1:3] if len(nz_tag) >= 3 else ""
            if bin_code == "25":
                nz_bin = "le25"
            elif bin_code == "50":
                nz_bin = "ge25"
            else:
                # new: L13 / R27 / ...
                tail = nz_tag[1:]
                nz_bin = "single" if tail.isdigit() else "unk"
        else:
            nz_bin = "unk"
    else:
        nz_mode = "NO_NZ"
        nz_bin = "none"

    return (bg, op, nz_mode, nz_bin)


def md5_rel_key(src_root: Path, p: Path) -> str:
    rel = str(p.relative_to(src_root)).replace("\\", "/")
    return hashlib.md5(rel.encode("utf-8")).hexdigest()


# -----------------------------
# Path layout parsing
# -----------------------------
@dataclass(frozen=True)
class RelInfo:
    label: str
    map_name: Optional[str]
    rest: Tuple[str, ...]  # includes filename at end

def parse_rel(
    rel_parts: Tuple[str, ...],
    special_tops: set[str],
    bg_color_alias_to: str = "BGC",
    default_map_name: Optional[str] = None,
) -> Optional[RelInfo]:
    if not rel_parts:
        return None

    top = rel_parts[0]

    # SPECIAL
    if top in special_tops:
        filename = rel_parts[-1]
        if top in {"BGC", "bg_color"}:
            label = bg_color_alias_to
        else:
            label = "blank"
        return RelInfo(label=label, map_name=None, rest=(filename,))

    # NORMAL (old): (MAP, CLASS, ...FILE)
    if len(rel_parts) >= 3:
        map_name = rel_parts[0]
        label = rel_parts[1]
        rest = rel_parts[2:]
        return RelInfo(label=label, map_name=map_name, rest=tuple(rest))

    # NORMAL (new, single-map): (CLASS, ...FILE)
    if len(rel_parts) >= 2 and default_map_name:
        label = rel_parts[0]
        rest = rel_parts[1:]
        return RelInfo(label=label, map_name=default_map_name, rest=tuple(rest))

    return None

def build_dst(src_root: Path, p: Path, split_root: Path, special_tops: set[str], bg_color_alias_to: str, default_map_name: Optional[str]) -> Path:
    rel_parts = p.relative_to(src_root).parts
    info = parse_rel(rel_parts, special_tops=special_tops, bg_color_alias_to=bg_color_alias_to, default_map_name=default_map_name)
    if info is None:
        raise ValueError(f"Cannot parse label/map from path: {p}")

    if info.map_name is None:
        return split_root / info.label / "_root" / info.rest[0]

    return split_root / info.label / info.map_name / Path(*info.rest)

def copy_to_split(src_root: Path, p: Path, split_root: Path, special_tops: set[str], bg_color_alias_to: str, default_map_name: Optional[str], do_copy: bool):
    dst = build_dst(src_root, p, split_root, special_tops, bg_color_alias_to, default_map_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(p, dst)


# -----------------------------
# Core split logic
# -----------------------------
def largest_remainder_allocation(recipes: List[Tuple[str, str, str, str]], groups: Dict, val_ratio: float, desired_val: int) -> Dict:
    base_cnt = {}
    frac = {}
    base_sum = 0

    for r in recipes:
        nk = len(groups[r])
        raw = nk * val_ratio
        b = int(math.floor(raw))
        base_cnt[r] = b
        frac[r] = raw - b
        base_sum += b

    rem = desired_val - base_sum
    if rem > 0:
        order = sorted(recipes, key=lambda r: (-frac[r], str(r)))
        for i in range(rem):
            base_cnt[order[i % len(order)]] += 1

    return base_cnt


def _tqdm(it, **kwargs):
    if tqdm is None:
        return it
    return tqdm(it, **kwargs)


def run(cfg: dict):
    src = Path(cfg["src"])
    if not src.exists():
        raise FileNotFoundError(f"SRC does not exist: {src}")

    out_base = Path(cfg["out_base"])
    out_base.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = cfg.get("out_prefix", "fixed_train_data_")
    out = out_base #/ f"{out_prefix}{ts}"
    out_train = out / "train"
    out_val = out / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    val_ratio = float(cfg.get("val_ratio", 0.2))
    img_exts = set([e.lower() for e in cfg.get("img_exts", [".png"])])
    special_tops = set(cfg.get("special_tops", ["BGC", "blank"]))
    bg_color_alias_to = cfg.get("bg_color_alias_to", "BGC")
    do_copy = bool(cfg.get("do_copy", True))
    use_tqdm = bool(cfg.get("use_tqdm", True)) and (tqdm is not None)
    default_map_name = cfg.get("default_map_name")
    # scan
    all_imgs = sorted([p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in img_exts])

    per_class: Dict[str, List[Path]] = {}
    for p in all_imgs:
        info = parse_rel(
            p.relative_to(src).parts,
            special_tops=special_tops,
            bg_color_alias_to=bg_color_alias_to,
            default_map_name=default_map_name,
        )

        if info is None:
            continue
        per_class.setdefault(info.label, []).append(p)

    print(f"[scan] classes={len(per_class)} | total_imgs={sum(len(v) for v in per_class.values())}")
    print(f"[out ] {out}")

    meta = {
        "created_at": ts,
        "src": str(src),
        "out": str(out),
        "val_ratio": val_ratio,
        "rule_primary": "stratified by (bg, op, nz_mode, nz_bin); exact per-class val=int(round(n*val_ratio)); deterministic by md5(relative_path)",
        "rule_fallback": "if strata mismatch occurs -> global md5(relative_path) fallback to force exact desired_val",
        "per_class": {},
        "totals": {},
        "do_copy": do_copy,
    }

    total_train = 0
    total_val = 0

    for label in sorted(per_class.keys()):
        paths = sorted(per_class[label], key=lambda x: (x.name, str(x)))
        n = len(paths)
        desired_val = int(round(n * val_ratio))

        # group by recipe
        groups: Dict[Tuple[str, str, str, str], List[Path]] = {}
        for p in paths:
            recipe = parse_recipe_from_name(p)
            groups.setdefault(recipe, []).append(p)

        recipes = sorted(groups.keys(), key=lambda r: str(r))

        # allocate exact desired_val by Largest Remainder
        base_cnt = largest_remainder_allocation(recipes, groups, val_ratio, desired_val)

        # deterministic pick within each recipe by md5(relative_path)
        val_list: List[Path] = []
        train_list: List[Path] = []
        for r in recipes:
            g = groups[r]
            g_sorted = sorted(g, key=lambda x: (md5_rel_key(src, x), x.name, str(x)))
            v = max(0, min(base_cnt.get(r, 0), len(g_sorted)))
            val_list.extend(g_sorted[:v])
            train_list.extend(g_sorted[v:])

        # sanity/fallback to force exact
        if len(val_list) != desired_val:
            all_sorted = sorted(paths, key=lambda x: (md5_rel_key(src, x), x.name, str(x)))
            val_list = all_sorted[:desired_val]
            train_list = all_sorted[desired_val:]

        # copy
        train_iter = train_list
        val_iter = val_list
        if use_tqdm:
            train_iter = _tqdm(train_list, desc=f"copy train [{label}]", leave=False)
            val_iter = _tqdm(val_list, desc=f"copy val   [{label}]", leave=False)

        for p in train_iter:
            copy_to_split(src, p, out_train, special_tops, bg_color_alias_to, default_map_name, do_copy=do_copy)
        for p in val_iter:
            copy_to_split(src, p, out_val,   special_tops, bg_color_alias_to, default_map_name, do_copy=do_copy)


        meta["per_class"][label] = {
            "total": n,
            "train": len(train_list),
            "val": len(val_list),
            "ratio_train": round(len(train_list) / n, 4) if n else 0.0,
            "ratio_val": round(len(val_list) / n, 4) if n else 0.0,
            "strata_keys": len(recipes),
        }

        total_train += len(train_list)
        total_val += len(val_list)

    meta["totals"] = {
        "train": total_train,
        "val": total_val,
        "total": total_train + total_val,
    }

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ split 完成")
    print(f"train={total_train} | val={total_val} | total={total_train + total_val}")
    print("meta.json =", out / "meta.json")

    # quick print
    for k, v in sorted(meta["per_class"].items()):
        print(f"[{k:<8}] train={v['train']:>4} val={v['val']:>4} total={v['total']:>4}  ({v['ratio_train']:.2f}:{v['ratio_val']:.2f})")

    # examples for BGC/blank
    def show_examples(ex_label: str, split_root: Path, limit: int = 3):
        ps = sorted(per_class.get(ex_label, []), key=lambda x: (x.name, str(x)))[:limit]
        print(f"\n[examples] {ex_label} ->")
        for p in ps:
            print("  ", build_dst(src, p, split_root, special_tops, bg_color_alias_to, default_map_name))

    show_examples("BGC", out_train)
    show_examples("blank", out_train)

def main(yaml_obj=None):
    if yaml_obj is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfg", required=True, help="Path to YAML config, e.g., configs/split_train_val.yaml")
        args = ap.parse_args()

        cfg_path = Path(args.cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = yaml_obj
    run(cfg)


if __name__ == "__main__":
    main()
