"""
Repository utilities for cloning and extracting code repositories.

This module provides tools to clone repositories from various platforms
(GitHub, GitLab, Zenodo, Figshare, OSF) and extract their content for
analysis by language models.

Main Components
---------------
- :func:`clone_and_extract_tree`: Main entry point for repository extraction.
- :class:`RepoStatus`: Enum indicating extraction success/failure status.

Submodules
----------
- :mod:`repo_providers`: Platform-specific cloners (Git, Zenodo, etc.)
- :mod:`main_extractor`: Core extraction logic and data models
- :mod:`tokenizer`: Text extraction and tokenization utilities

Example
-------
>>> from repo_utils import clone_and_extract_tree, RepoStatus
>>> result = clone_and_extract_tree(
...     repo_url=\"https://github.com/user/repo\",
...     context_window=100000,
...     clone_dir=\"./repos\",
... )
>>> if result.status == RepoStatus.OK:
...     print(result.output)
"""

from .main_extractor import clone_and_extract_tree, RepoStatus
