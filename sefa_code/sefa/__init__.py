"""Symbolic Emergence Field Analysis (SEFA) Package.

Provides the SEFA class for performing the analysis based on SEFA.md.
"""

__version__ = "0.1.0" # Placeholder version

from .config import SEFAConfig, DEFAULT_CONFIG, SavgolParams
from .sefa_core import SEFA

__all__ = [
    "SEFA",
    "SEFAConfig",
    "DEFAULT_CONFIG",
    "SavgolParams"
]