"""Load the small YAML configuration, without runtime module injection."""

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "segmentation": {"prob_threshold": 0.35, "min_area_ratio": 0.001,
                     "use_ellipse": False, "max_masks_per_prompt": 10, "device": "cuda"},
    "llm": {"model": "Qwen/Qwen3-VL-8B-Instruct", "server_url": "http://127.0.0.1:8001/v1",
            "api_key": "EMPTY", "max_tokens": 8192, "temperature": 0.5,
            "thinking": {"enabled": True, "budgets": {"orchestrator": 2000, "examine": 5000,
                                                       "judge": 200, "default": 2000}}},
    "inference": {"max_generations": 50, "max_duration": 2.9,
                  "judge_enabled": True, "termination_confidence_threshold": 0.86},
    "scanpath": {"fps": 25, "kappa": 10, "threshold": 1.0, "noise": 7,
                 "eta": 17, "ffi": True, "device": "cuda"},
}


def load_config(path=None):
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        with Path(path).expanduser().open() as stream:
            overrides = yaml.safe_load(stream) or {}
        def merge(base, values):
            if not isinstance(values, dict):
                raise ValueError("Configuration sections must be mappings")
            for key, value in values.items():
                if key not in base:
                    raise ValueError(f"Unknown configuration option: {key}")
                if isinstance(base[key], dict):
                    merge(base[key], value)
                else:
                    base[key] = value
        merge(config, overrides)
    for name in ("max_generations", "max_duration"):
        value = config["inference"][name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"inference.{name} must be positive")
    if not isinstance(config["inference"]["judge_enabled"], bool):
        raise ValueError("inference.judge_enabled must be a boolean")
    threshold = config["inference"]["termination_confidence_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("inference.termination_confidence_threshold must be in [0, 1]")
    return config
