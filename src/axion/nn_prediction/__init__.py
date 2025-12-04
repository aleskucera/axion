# Standalone NeRD Model Integration
"""
This package provides a standalone implementation for using pretrained NeRD models
as black-box robot dynamics predictors.

Main components:
    - NeRDPredictor: Main interface for loading and using pretrained models
    - models: Neural network architectures
    - utils: Utility functions for quaternions, coordinate transforms, etc.
    - integrators: State processing and conversion utilities
"""

from .nerd_predictor import NeRDPredictor

__all__ = ['NeRDPredictor']

