#!/usr/bin/env python3
"""
Extract normalization statistics from saved model and print them in a format
that can be hardcoded into example_usage.py.
"""

import torch
import numpy as np
from pathlib import Path


def extract_and_print_stats(model_path):
    """Load model and extract normalization statistics with correct slicing."""
    model, robot_name = torch.load(model_path, map_location='cpu', weights_only=False)
    
    for key, rms in model.input_rms.items():
        mean = rms.mean
        var = rms.var
        # Convert to numpy for easier printing
        mean_np = mean.cpu().numpy()
        var_np = var.cpu().numpy()

        print(f"key= {key}, rms shape= {mean.shape}")
        

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract normalization statistics from saved model')
    parser.add_argument(
        '--model-path',
        type=str,
        default='src/axion/nn_prediction/trained_models/NeRD_pretrained/pendulum/model.pt',
        help='Path to the saved model file (.pt)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        exit(1)
    
    extract_and_print_stats(args.model_path)

