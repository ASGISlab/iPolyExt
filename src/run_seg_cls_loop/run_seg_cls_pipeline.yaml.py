#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def replace_vars(text: str, context: dict[str, Any]) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(f"Missing variable: {key}")
        return str(context[key])

    return _VAR_RE.sub(repl, text)


def flatten_scalars(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_scalars(v, key))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
            if "." not in key:
                out[str(k)] = v
    return out


def substitute_obj(obj: Any, context: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: substitute_obj(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_obj(v, context) for v in obj]
    if not isinstance(obj, str):
        return obj
    return replace_vars(obj, context)


def resolve_mapping_vars(mapping: dict[str, Any], max_passes: int = 8) -> dict[str, Any]:
    """
    讓 master yaml 裡的 meta/common 也能互相用 ${...}。
    例如：
      experiment_name: "map${map_number}_${total_rounds}round"
      pipeline_output_root: "${work_dir}/output/${experiment_name}"
    """
    current = dict(mapping)

    for _ in range(max_passes):
        context = flatten_scalars(current)
        new_current = substitute_obj(current, context)
        if new_current == current:
            return new_current
        current = new_current

    unresolved = []
    flat = flatten_scalars(current)
    for k, v in flat.items():
        if isinstance(v, str) and "${" in v:
            unresolved.append(f"{k}={v}")

    if unresolved:
        raise ValueError("Unresolved placeholders in master yaml:\n" + "\n".join(unresolved))

    return current


def run_step(python_bin: str, script_path: Path, config_path: Path, cwd: Path) -> None:
    cmd = [python_bin, str(script_path), "--config", str(config_path)]
    print("\n" + "=" * 90)
    print("[RUN]", " ".join(cmd))
    print("[CWD]", str(cwd))
    print("=" * 90)

    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_path.name}, code={result.returncode}")


def materialize_config(template_path: Path, out_path: Path, context: dict[str, Any]) -> None:
    if not template_path.exists():
        raise FileNotFoundError(f"Template YAML not found: {template_path}")

    raw = template_path.read_text(encoding="utf-8")
    rendered = replace_vars(raw, context)
    save_text(rendered, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="master pipeline yaml")
    args = parser.parse_args()

    master_path = Path(args.config).resolve()
    master = load_yaml(master_path)

    raw_meta = master["meta"]
    raw_common = master["common"]
    modules = master["modules"]

    # 先把 meta/common 合起來，讓它們自己也能互相展開 ${...}
    merged_global: dict[str, Any] = {}
    merged_global.update(raw_meta)
    merged_global.update(raw_common)
    resolved_global = resolve_mapping_vars(merged_global)

    meta = {k: resolved_global[k] for k in raw_meta.keys()}
    common = {k: resolved_global[k] for k in raw_common.keys()}

    work_dir = resolve_path(str(meta["work_dir"]), master_path.parent)
    python_bin = str(meta.get("python", sys.executable))
    temp_dir = resolve_path(str(meta["temp_config_dir"]), work_dir)

    map_number = str(common["map_number"])
    train_model_method = str(common["train_model_method"])
    total_rounds = int(common["total_rounds"])

    print(f"[Pipeline] work_dir={work_dir}")
    print(f"[Pipeline] map_number={map_number}")
    print(f"[Pipeline] train_model_method={train_model_method}")
    print(f"[Pipeline] total_rounds={total_rounds}")
    print(f"[Pipeline] temp_dir={temp_dir}")

    for round_idx in range(1, total_rounds + 1):
        prev_round = round_idx - 1 if round_idx > 1 else 0

        if round_idx == 1:
            plan = [
                "seg_round1",
                "classify_round1",
                "check_seg_cls_round_integrity",
            ]
        else:
            plan = [
                "seg_multi_round",
                "classify_multi_round",
                "check_seg_cls_round_integrity",
            ]

        print("\n" + "#" * 90)
        print(f"[ROUND {round_idx}]")
        print("#" * 90)

        # 每輪 context：包含所有 meta/common scalar + round runtime
        context: dict[str, Any] = {}
        context.update(flatten_scalars(meta))
        context.update(flatten_scalars(common))
        context.update(
            {
                "round": round_idx,
                "prev_round": prev_round,
                "work_dir": str(work_dir),
            }
        )

        for module_name in plan:
            if module_name not in modules:
                raise KeyError(f"Module not found in master yaml: {module_name}")

            module = modules[module_name]
            script_path = resolve_path(str(module["script"]), work_dir)
            template_path = resolve_path(str(module["config_template"]), work_dir)

            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            temp_cfg = temp_dir / f"round_{round_idx:02d}__{module_name}.yaml"
            materialize_config(template_path, temp_cfg, context)

            run_step(
                python_bin=python_bin,
                script_path=script_path,
                config_path=temp_cfg,
                cwd=work_dir,
            )

    print("\n[Pipeline] ALL DONE")


if __name__ == "__main__":
    main()