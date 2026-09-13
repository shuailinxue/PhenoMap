import ast
from pathlib import Path

import yaml


def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key.subkey=value, got: {item}")
        key_path, value_str = item.split("=", 1)
        node = cfg
        keys = key_path.split(".")
        for key in keys[:-1]:
            if key not in node:
                raise KeyError(f"Unknown config path: {key_path}")
            node = node[key]
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        node[keys[-1]] = value
    return cfg


def require(cfg, paths):
    missing = []
    for path in paths:
        node = cfg
        for key in path.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        if node in (None, ""):
            missing.append(path)
    if missing:
        raise ValueError("Missing required config values:\n" + "\n".join(f"  {p}" for p in missing))
