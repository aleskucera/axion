# Track Simulation Plan

## Current Implementation
- Bodies are enforced to be on a track curve using `EQ_TYPE_TRACK`.
- Bodies are linked together in a chain using spherical joints (`newton.JointType.BALL`).
- Dynamics (forces) play a role in how links move relative to each other within the constraints.

## Proposed Implementation
- **Individual Bodies:** Each track segment (body) is treated individually.
- **Zero DOF:** Each body will have 0 degrees of freedom relative to the track constraint. All 6 DOFs (position and orientation) will be fully constrained by the track geometry.
- **Parametric Control:** The position of each body on the track is defined by a single scalar parameter (e.g., `u`).
- **Speed Control:** The simulation will control the "track speed," which directly updates the parameter `u` for all bodies, moving them along the fixed path.

## Goal
To simulate a driven track system (like a conveyor or tank tread) where the motion is kinematically driven by the track speed rather than by forces acting on a dynamic chain.
