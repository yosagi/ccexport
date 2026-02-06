"""
Configuration management module

Priority order:
1. Command line arguments
2. Environment variables
3. Configuration file (TOML)
4. Default values
"""

import os
import sys

# Import TOML library
if sys.version_info >= (3, 11):
    try:
        import tomllib
        TOML_BINARY_MODE = True
    except ImportError:
        import toml as tomllib
        TOML_BINARY_MODE = False
else:
    import toml as tomllib
    TOML_BINARY_MODE = False
from pathlib import Path
from typing import Optional, Dict, Any


class Config:
    """Configuration management class"""

    # Default configuration (for ccexport - no LLM related settings)
    DEFAULT_CONFIG = {
        'paths': {
            'projects_dir': '~/.claude/projects'
        },
        'processing': {
            'code_block_handling': 'keep',  # Always 'keep' for ccexport
            # Continuous session integration settings
            'merge_continuous_sessions': True,
            'exclude_continuation_summaries': True
        },
        'output': {
            'format': 'markdown',
            'include_timestamps': True,
            'group_by_date': True
        },
        'debug': {
            'verbose': False
        }
    }
    
    def __init__(self, config_file: Optional[str] = None,
                 projects_dir: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize configuration

        Args:
            config_file: Path to configuration file
            projects_dir: Path to projects directory (CLI argument)
            verbose: Verbose logging flag
        """
        # Keep the configuration file path
        self.config_file = config_file or "config.toml"
        self._config = self._load_config(config_file)

        # Override with CLI arguments
        if projects_dir:
            self._config['paths']['projects_dir'] = projects_dir
        if verbose:
            self._config['debug']['verbose'] = verbose

        # Override with environment variables
        self._apply_env_overrides()

        # Resolve paths
        self._resolve_paths()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration file"""
        config = self.DEFAULT_CONFIG.copy()

        # Load configuration file
        if config_file:
            config_path = Path(config_file)
        else:
            # Default configuration file location
            config_path = Path('config.toml')
            if not config_path.exists():
                config_path = Path.home() / '.config' / 'ccdigest' / 'config.toml'
        
        if config_path.exists():
            try:
                # Python 3.11+ tomllib uses 'rb' mode, toml package uses 'r' mode
                if TOML_BINARY_MODE:
                    with open(config_path, 'rb') as f:
                        file_config = tomllib.load(f)
                else:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_config = tomllib.load(f)
                # Deep merge
                config = self._deep_merge(config, file_config)
            except Exception as e:
                # Simple implementation: error log only
                print(f"Configuration file load error: {e}")
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self):
        """Override configuration with environment variables"""
        # Projects directory
        if projects_dir := os.getenv('CLAUDE_PROJECTS_DIR'):
            self._config['paths']['projects_dir'] = projects_dir

        # Ollama settings
        if ollama_host := os.getenv('OLLAMA_HOST'):
            self._config['ollama']['host'] = ollama_host
        if ollama_model := os.getenv('OLLAMA_MODEL'):
            self._config['ollama']['model'] = ollama_model
    
    def _resolve_paths(self):
        """Resolve paths (e.g., ~ expansion)"""
        projects_dir = self._config['paths']['projects_dir']
        self._config['paths']['projects_dir'] = str(Path(projects_dir).expanduser())

    # Access configuration values as properties
    @property
    def projects_dir(self) -> str:
        return self._config['paths']['projects_dir']

    @property
    def verbose(self) -> bool:
        return self._config['debug']['verbose']

    # Continuous session integration settings
    @property
    def merge_continuous_sessions(self) -> bool:
        """Enable session integration"""
        return self._config['processing']['merge_continuous_sessions']

    @property
    def exclude_continuation_summaries(self) -> bool:
        """Exclude continuation summaries"""
        return self._config['processing']['exclude_continuation_summaries']

    @property
    def code_block_handling(self) -> str:
        """Code block handling method ('remove', 'keep')"""
        return self.get('processing.code_block_handling', 'keep')

    def get(self, key: str, default=None):
        """Get configuration value by nested key (e.g., 'paths.projects_dir')"""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default