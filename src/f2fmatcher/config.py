import os
import yaml


class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
        self._config_dict = config_dict

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(d)

    @classmethod
    def from_dict(cls, d):
        return cls(d)

    def get(self, key, default=None):
        keys = key.split(".")
        val = self._config_dict
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
                if val is None:
                    return default
            else:
                return default
        return val

    def merge(self, other):
        merged = {}
        self._deep_merge(merged, self._config_dict)
        self._deep_merge(merged, other._config_dict if isinstance(other, Config) else other)
        return Config(merged)

    @staticmethod
    def _deep_merge(target, source):
        for k, v in source.items():
            if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                Config._deep_merge(target[k], v)
            else:
                target[k] = v

    def to_dict(self):
        return self._config_dict


def load_config(path=None, overrides=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(here, "..", "..", "configs", "default.yaml")
    default_path = os.path.normpath(default_path)

    cfg = Config.from_yaml(default_path)

    if path is not None:
        user_cfg = Config.from_yaml(path)
        cfg = cfg.merge(user_cfg)

    if overrides:
        cfg = cfg.merge(overrides)

    return cfg
