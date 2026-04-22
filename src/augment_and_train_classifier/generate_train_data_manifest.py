# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime

try:
    import yaml  # pip install pyyaml
except ImportError:
    yaml = None


_BASE_RE = re.compile(r'^([A-Za-z]+)')

def _clean_base_token(tok: str | None) -> str | None:
    if tok is None:
        return None
    s = str(tok).strip()
    s = Path(s).name
    s = s.split('.', 1)[0]
    s = s.split('_', 1)[0]
    m = _BASE_RE.match(s)
    return m.group(1) if m else s

def parse_core_and_components(cls_name: str):
    """
    例：
      "Mag-Mk_17_13" -> core="Mag-Mk", components=["Mag","Mk"]
      "Qy_001"       -> core="Qy",     components=["Qy"]
      "blank"        -> core="blank",  components=["blank"]
    """
    s = str(cls_name).strip()
    s = Path(s).name
    s = s.split('.', 1)[0]

    parts = [p for p in (_clean_base_token(t) for t in s.split('-')) if p]
    core = "-".join(parts) if parts else s
    return core, parts

def load_meta_classes(meta_path: Path):
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到 {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta.get("classes", None)
    if isinstance(classes, list) and all(isinstance(x, str) for x in classes):
        return classes

    per_class = meta.get("per_class", None)
    if isinstance(per_class, dict) and len(per_class) > 0:
        # 建議排序，確保每次輸出順序一致（可重現）
        return sorted(per_class.keys())

    raise ValueError("meta.json 缺少有效的 'classes'(list) 或 'per_class'(dict)")


def backup_if_exists(path: Path):
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.name}.{ts}.bak")
        shutil.move(str(path), str(backup_path))
        print(f"[備份] 已將既有檔案備份為：{backup_path}")

def build_train_data_json_v2(classes, pure_color_enabled: bool, pure_color_list):
    s_classes = set(classes)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 僅保留真的存在於 classes 的（嚴格比對）
    positives = [c for c in pure_color_list if c in s_classes] if pure_color_enabled else []

    data = {
        "version": 2,
        "created_at": now,
        "meta_ref": "meta.json",
        "attributes": {
            "pure_color": {
                "description": "是否屬於純色圖（無紋理）",
                "type": "boolean",
                "enabled": bool(pure_color_enabled),
                "default": False,
                "positives": positives
            }
        },
        "classes": {}
    }

    cores_seen = set()
    cores_list = []

    for cls in classes:
        core, components = parse_core_and_components(cls)
        if core and core not in cores_seen:
            cores_seen.add(core)
            cores_list.append(core)

        is_pure = (cls in positives)
        data["classes"][cls] = {
            "core": core,
            "components": components,
            "attrs": {
                "pure_color": bool(is_pure)
            },
            "extra": {}
        }

    data["derived"] = {
        "unique_cores": cores_list
    }
    return data

def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("缺少 PyYAML：請先 pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _resolve_path(root_dir: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root_dir / pp)

def main(yaml_obj=None):
    if yaml_obj is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfg", required=True, help="YAML config path, e.g. configs/train_data_manifest.yaml")
        args = ap.parse_args()

        cfg_path = Path(args.cfg)
        if not cfg_path.exists():
            print(f"[錯誤] 找不到 config：{cfg_path}")
            sys.exit(1)

        cfg = _load_yaml(cfg_path)
    else:
        cfg = yaml_obj

    root_dir = Path(cfg["root_dir"])
    meta_path = _resolve_path(root_dir, cfg.get("meta_path", "meta.json"))
    out_path  = _resolve_path(root_dir, cfg.get("out_path", "train_data.json"))

    pure_cfg = cfg.get("pure_color", {}) or {}
    pure_enabled = bool(pure_cfg.get("enabled", True))
    pure_list = pure_cfg.get("positives", []) or []

    backup_cfg = cfg.get("backup", {}) or {}
    backup_enabled = bool(backup_cfg.get("enabled", True))

    try:
        classes = load_meta_classes(meta_path)
    except Exception as e:
        print(f"[錯誤] 無法讀取 classes：{e}")
        sys.exit(1)

    data = build_train_data_json_v2(classes, pure_enabled, pure_list)

    if backup_enabled:
        backup_if_exists(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已輸出：{out_path}")
    print(f"[meta] {meta_path}")
    print(f"[pure_color] enabled={pure_enabled}, positives={len(data['attributes']['pure_color']['positives'])}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

