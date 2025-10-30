"""Model Comparison Analysis Package

A comprehensive package for comparing language model outputs using various
analytical techniques including UMAP visualizations, similarity analysis,
vocabulary diversity metrics, and more.
"""

from .analyzer import OrganizedModelAnalyzer
from .main import main

__version__ = "1.0.0"
__all__ = ["OrganizedModelAnalyzer", "main"]
