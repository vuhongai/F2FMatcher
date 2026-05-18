import os
import sys
import importlib.util
from pathlib import Path


def import_module_from_path(path):
    path = Path(path).resolve()
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
