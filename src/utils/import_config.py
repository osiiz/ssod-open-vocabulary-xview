"""Utility functions for importing configuration files."""

import importlib.util
from pathlib import Path


def import_py_config(config_file: str):
    """Import configuration from .py file"""

    # Crear un path
    config_path = Path(config_file)

    # Cargar o módulo dinámicamente
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module
