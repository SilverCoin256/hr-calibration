"""
Load config from 00_config.py as a dictionary.
Centralizes import for all pipeline modules.
"""
import importlib.util
import os


def load_config() -> dict:
    """Load 00_config.py and return all uppercase names as a dict."""
    config_path = os.path.join(os.path.dirname(__file__), "00_config.py")
    spec = importlib.util.spec_from_file_location("config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: getattr(mod, k) for k in dir(mod) if k.isupper() or k.startswith("GADRS") or k.startswith("LR") or k.startswith("RF") or k.startswith("GB") or k.startswith("GOV")}
