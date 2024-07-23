# src/utils/config_utils.py

import os
import yaml
import logging

def load_config(default_config_path: str = 'config.yaml') -> dict:
    config_path = os.getenv('CONFIG_PATH', default_config_path)
    if not os.path.exists(config_path):
        logging.error(f"Config file not found at {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            logging.debug(f"Configuration loaded successfully from {config_path}")
            return config
        except yaml.YAMLError as exc:
            logging.error(f"Error reading the config file: {exc}")
            return None
