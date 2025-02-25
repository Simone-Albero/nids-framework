import json
import os


class ConfigManager:
    config = None

    @staticmethod
    def load_config(config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

        with open(config_path, 'r') as file:
            ConfigManager.config = json.load(file)

    @staticmethod
    def get_value(section, key, default=None):
        if ConfigManager.config and section in ConfigManager.config:
            return ConfigManager.config[section].get(key, default)
        return default