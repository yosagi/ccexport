# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 @yosagi
"""
Configuration management module

Priority order:
1. Command line arguments
2. Environment variables
3. Default values
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration management class"""

    def __init__(self,
                 projects_dir: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize configuration

        Args:
            projects_dir: Path to projects directory (CLI argument)
            verbose: Verbose logging flag
        """
        self._projects_dir = projects_dir or os.getenv('CLAUDE_PROJECTS_DIR') or '~/.claude/projects'
        self._projects_dir = str(Path(self._projects_dir).expanduser())
        self._verbose = verbose

    @property
    def projects_dir(self) -> str:
        return self._projects_dir

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def merge_continuous_sessions(self) -> bool:
        """Enable session integration"""
        return True

    @property
    def exclude_continuation_summaries(self) -> bool:
        """Exclude continuation summaries"""
        return True
