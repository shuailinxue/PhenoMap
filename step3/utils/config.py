import yaml
import os


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    required_sections = ("data", "model", "loss", "training")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    required_model_values = (
        "keep_vis_dim",
        "keep_text_dim",
        "scgpt_dim",
        "spatial_hidden_dim",
    )
    missing_model = [key for key in required_model_values if config["model"].get(key) is None]
    if missing_model:
        raise ValueError(f"Missing model config values: {', '.join(missing_model)}")
    return True
