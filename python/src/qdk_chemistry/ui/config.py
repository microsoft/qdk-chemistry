"""Configuration for the QDK/Chemistry MCP Server."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import errno
import os
from pathlib import Path


class QDKMCPConfig:
    """Configuration class for QDK/Chemistry MCP Server.

    In general, the agent will work in a dedicated project directory,
    where the directory structure is like:

    scratch/projects/
    ├── project1/
    │   ├── water.structure.json
    │   ├── water.orbitals.json
    │   └── water.wavefunction.json
    ├── project2/
    │   ├── n2.structure.json
    │   ├── n2.orbitals.json
    │   └── n2.wavefunction.json
    """

    def __init__(self):
        """Initialize configuration with default values."""
        default_scratch = Path("/scratch")
        bckp_scratch = Path.home() / ".qdk_chem" / "scratch"

        scratch_dir_from_env = "QDK_SCRATCH_DIR" in os.environ

        if scratch_dir_from_env:
            self.scratch_dir = Path(os.environ["QDK_SCRATCH_DIR"])
        else:
            self.scratch_dir = default_scratch

        self.projects_dir = self.scratch_dir / "projects"

        # Local computation cache — configurable via QDK_CACHE_DIR.
        # Defaults to <scratch>/cache. Algorithm results are persisted here
        # so identical runs are never recomputed.
        if "QDK_CACHE_DIR" in os.environ:
            self.cache_dir = Path(os.environ["QDK_CACHE_DIR"])
        else:
            self.cache_dir = self.scratch_dir / "cache"

        # Server settings
        self.server_name = "qdk-chem-mcp"
        self.server_version = "1.0.0"

        try:
            self._setup_directories()
        except OSError as error:
            if scratch_dir_from_env or not self._should_fallback_to_home_scratch(error):
                raise
            self.scratch_dir = bckp_scratch
            self.projects_dir = self.scratch_dir / "projects"
            if "QDK_CACHE_DIR" not in os.environ:
                self.cache_dir = self.scratch_dir / "cache"
            self._setup_directories()

    @staticmethod
    def _should_fallback_to_home_scratch(error: OSError) -> bool:
        """Return whether the default system scratch path is unavailable."""
        return error.errno in {errno.EACCES, errno.EPERM, errno.EROFS}

    def _setup_directories(self):
        """Set up scratch and project directories."""
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = QDKMCPConfig()
