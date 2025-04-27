# -*- coding: utf-8 -*-
"""Manages configuration loading from YAML and .env files."""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigManager:
    """Loads and provides access to configuration settings."""

    _instance = None # Singleton pattern

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = 'config.yaml', env_path: Optional[str] = '.env'):
        """Initializes the ConfigManager (Singleton).

        Loads configuration from YAML and environment variables from .env file.
        Subsequent calls to __init__ will not reload unless force_reload is used.

        Args:
            config_path: Path to the main YAML configuration file.
            env_path: Path to the .env file (optional).
        """
        # Prevent re-initialization in singleton pattern unless forced
        if hasattr(self, '_initialized') and self._initialized:
             logger.debug("ConfigManager already initialized. Skipping reload.")
             return

        self.config_path = config_path
        self.env_path = env_path
        self._config: Dict[str, Any] = {}
        self._env_loaded = False

        self.load_config() # Load config during initialization

        self._initialized = True
        logger.info("ConfigManager initialized.")


    def load_config(self, force_reload: bool = False):
        """Loads configuration from the YAML file and .env file.

        Args:
            force_reload: If True, forces reloading even if already loaded.
        """
        if self._config and not force_reload:
            logger.debug("Configuration already loaded. Skipping reload.")
            return

        # Load .env file first (if specified and exists)
        self._env_loaded = False
        if self.env_path:
            env_file_path = os.path.abspath(self.env_path)
            if os.path.exists(env_file_path):
                try:
                    load_dotenv(dotenv_path=env_file_path, override=True) # Override existing env vars
                    self._env_loaded = True
                    logger.info("Loaded environment variables from: %s", env_file_path)
                except Exception as e:
                    logger.error("Failed to load .env file from %s: %s", env_file_path, e)
            else:
                logger.warning(".env file specified but not found at: %s", env_file_path)
        else:
             logger.info(".env file path not specified. Relying on system environment variables.")
             # Check if essential keys are in environment anyway
             self._check_essential_env_vars()


        # Load YAML configuration file
        if not os.path.exists(self.config_path):
            logger.error("Configuration file not found: %s", self.config_path)
            # Decide behavior: raise error or return empty config? Raising for now.
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            if not isinstance(self._config, dict):
                 logger.error("Configuration file %s is not a valid YAML dictionary.", self.config_path)
                 self._config = {} # Reset to empty dict on invalid format
                 raise ValueError(f"Invalid YAML format in {self.config_path}")
            logger.info("Loaded configuration from: %s", self.config_path)
        except yaml.YAMLError as e:
            logger.error("Error parsing YAML configuration file %s: %s", self.config_path, e)
            self._config = {} # Reset on error
            raise ValueError(f"Error parsing YAML file {self.config_path}: {e}") from e
        except Exception as e:
            logger.error("Failed to read configuration file %s: %s", self.config_path, e)
            self._config = {} # Reset on error
            raise IOError(f"Failed to read configuration file {self.config_path}: {e}") from e

        # Optionally resolve environment variables within YAML values (e.g., ${VAR_NAME})
        self._resolve_env_vars_in_config(self._config)

        # Validate essential configurations after loading
        self._validate_config()


    def _check_essential_env_vars(self):
        """Checks if essential API keys are present in the environment if .env wasn't loaded."""
        # This is a basic check; specific clients might need different keys.
        required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY'] # Add others like GEMINI_API_KEY if needed
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
             logger.warning(
                 "Essential environment variables missing (and .env not loaded/found): %s. "
                 "LLM clients requiring these keys may fail.",
                 ', '.join(missing_keys)
             )


    def _resolve_env_vars_in_config(self, config_dict: Dict[str, Any]):
        """Recursively resolves ${VAR_NAME} placeholders in config values."""
        for key, value in config_dict.items():
            if isinstance(value, str):
                # Simple substitution: ${VAR_NAME} or $VAR_NAME
                # More robust parsing might be needed for complex cases
                if value.startswith('${') and value.endswith('}'):
                    var_name = value[2:-1]
                    env_value = os.getenv(var_name)
                    if env_value is not None:
                        config_dict[key] = env_value
                        logger.debug("Resolved env var %s for config key '%s'", var_name, key)
                    else:
                        logger.warning("Environment variable %s not found for config key '%s'. Keeping placeholder.", var_name, key)
                # Add more substitution patterns if needed
            elif isinstance(value, dict):
                self._resolve_env_vars_in_config(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._resolve_env_vars_in_config(item)
                    # Could add handling for strings within lists if necessary


    def _validate_config(self):
        """Performs basic validation on the loaded configuration."""
        # Example validations:
        if 'llm' not in self._config or not isinstance(self._config.get('llm'), dict):
             raise ValueError("Missing or invalid 'llm' section in config.yaml")
        if 'provider' not in self._config['llm'] or not self._config['llm']['provider']:
             raise ValueError("Missing 'provider' under 'llm' section in config.yaml")
        if 'model' not in self._config['llm'] or not self._config['llm']['model']:
             raise ValueError("Missing 'model' under 'llm' section in config.yaml")
        if 'paths' not in self._config or not isinstance(self._config.get('paths'), dict):
             raise ValueError("Missing or invalid 'paths' section in config.yaml")
        if 'homework_dir' not in self._config['paths'] or not self._config['paths']['homework_dir']:
             raise ValueError("Missing 'homework_dir' under 'paths' section in config.yaml")
        if 'output_csv' not in self._config['paths'] or not self._config['paths']['output_csv']:
             raise ValueError("Missing 'output_csv' under 'paths' section in config.yaml")
        # Add more checks as needed (e.g., for prompt file paths)
        logger.info("Basic configuration validation passed.")


    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value by key.

        Uses dot notation for nested keys (e.g., "llm.provider").

        Args:
            key: The configuration key (use dot notation for nested keys).
            default: The default value to return if the key is not found.

        Returns:
            The configuration value or the default value.
        """
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    # Handle case where intermediate key is not a dict
                    logger.warning("Config key path invalid at '%s' in key '%s'.", k, key)
                    return default
            return value
        except KeyError:
            logger.debug("Config key '%s' not found. Returning default: %s", key, default)
            return default
        except Exception as e:
             logger.error("Error accessing config key '%s': %s", key, e)
             return default


    def get_llm_config(self) -> Dict[str, Any]:
        """Returns the 'llm' section of the configuration."""
        return self.get('llm', {})

    def get_path_config(self) -> Dict[str, Any]:
        """Returns the 'paths' section of the configuration."""
        return self.get('paths', {})

    def get_api_key(self, provider: str) -> Optional[str]:
        """Gets the API key for a specific LLM provider from environment variables.

        Args:
            provider: The LLM provider name (e.g., "openai", "anthropic").

        Returns:
            The API key string or None if not found.
        """
        provider_upper = provider.upper()
        # Standard environment variable names
        key_name = f"{provider_upper}_API_KEY"
        api_key = os.getenv(key_name)
        if not api_key:
             logger.warning("API key environment variable '%s' not found.", key_name)
        return api_key

# Helper function to get the singleton instance easily
def get_config_manager() -> ConfigManager:
    """Returns the singleton instance of ConfigManager."""
    return ConfigManager()