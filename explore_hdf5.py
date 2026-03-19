#!/usr/bin/env python3
"""Explore the structure of an HDF5 file."""

import h5py
import numpy as np

HDF5_PATH = "/home/maros/axion/src/axion/neural_solver/datasets/Pendulum/pendulumContactsValid250klen500envs250seed1.hdf5"
MAX_PREVIEW = 20

def explore_group(group, prefix=""):
    for key in sorted(group.keys()):
        item = group[key]
        path = prefix + "/" + key if prefix else key
        if isinstance(item, h5py.Group):
            print(path + "/ (Group)")
            explore_group(item, path)
        elif isinstance(item, h5py.Dataset):
            print(path + " (Dataset)")
            print("  shape:", item.shape)
            print("  dtype:", item.dtype)
            total = np.prod(item.shape)
            if total <= MAX_PREVIEW:
                print("  values:", item[:])
            elif total > 0:
                print("  first values:", item[:].flatten()[:MAX_PREVIEW])

with h5py.File(HDF5_PATH, "r") as f:
    explore_group(f)