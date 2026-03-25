"""Shared constants for stacked_boxes stability benchmark."""

DURATION = 3.0
DENSITY = 1500.0

HX1 = 0.2
HX2 = 0.8
HX3 = 1.6

Z1 = HX1
Z2 = 2 * HX1 + HX2
Z3 = 2 * HX1 + 2 * HX2 + HX3

STABILITY_TOL = 0.1

KE = 1e7
KD = 2e5
KF = 5e4
MU = 0.1

BSEARCH_TOL = 0.001
BSEARCH_MAX = 2.0
