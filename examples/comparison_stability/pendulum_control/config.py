"""Shared constants for control_stability benchmark."""
import math

DURATION = 3.0
LINK_LENGTH = 1.0
LINK_MASS = 1.0
LINK_INERTIA = LINK_MASS * LINK_LENGTH**2 / 3.0  # rod about end
Q_INIT = 0.0
Q_TARGET = math.pi / 3
STABILITY_TOL = math.pi  # rad — full rotation = definitively unstable

DT_SWEEP_KP = 1000.0
DT_SWEEP_KD = 25.0
DT_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]

GAIN_SWEEP_DT = 0.05
KP_VALUES = [10, 50, 100, 200, 500, 1000, 5000]

BSEARCH_KP = DT_SWEEP_KP
BSEARCH_KD = DT_SWEEP_KD
BSEARCH_MAX = 2.0
BSEARCH_TOL = 0.002
BSEARCH_DIVERGE = math.pi
