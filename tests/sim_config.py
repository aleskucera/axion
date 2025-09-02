import math
from dataclasses import dataclass
from dataclasses import field
from typing import Optional


@dataclass
class SimConfigHybrid:
    """
    A flexible hybrid configuration.
    It takes a *target* sim_dt and calculates the actual sim_dt and substeps
    required to perfectly synchronize with the rendering FPS.
    """

    sim_duration: float = 3.0
    fps: int = 30
    target_sim_dt: float = 1e-3  # User's desired physics timestep

    # --- Read-only, derived properties that will be calculated ---
    frame_dt: float = field(init=False, repr=False)
    substeps_per_frame: int = field(init=False)
    actual_sim_dt: float = field(init=False)  # The real dt that will be used
    num_frames: int = field(init=False)
    total_sim_steps: int = field(init=False)

    def __post_init__(self):
        self.frame_dt = 1.0 / self.fps

        # Calculate ideal number of substeps (can be fractional)
        ideal_substeps = self.frame_dt / self.target_sim_dt
        # Round to the nearest whole number of steps
        self.substeps_per_frame = round(ideal_substeps)
        if self.substeps_per_frame == 0:
            self.substeps_per_frame = 1

        # Calculate the *actual* sim_dt that fits perfectly
        self.actual_sim_dt = self.frame_dt / self.substeps_per_frame

        self.num_frames = math.ceil(self.sim_duration / self.frame_dt)
        self.total_sim_steps = self.num_frames * self.substeps_per_frame

        # Warn the user if the adjustment was significant
        adjustment = abs(self.actual_sim_dt - self.target_sim_dt) / self.target_sim_dt
        if adjustment > 0.01:  # e.g., >1% change
            print(
                f"Warning: Target sim_dt {self.target_sim_dt:.6f}s was adjusted to "
                f"{self.actual_sim_dt:.6f}s to match {self.fps} FPS."
            )


# --- Usage ---
config = SimConfigHybrid(sim_duration=3.0, fps=30, target_sim_dt=1e-3)

print(f"Target dt: {config.target_sim_dt}s")
print(f"Substeps per frame: {config.substeps_per_frame}")  # -> 33
print(f"Actual dt used: {config.actual_sim_dt:.6f}s")  # -> 0.001010s
print(f"Total frames: {config.num_frames}")
print(f"Total sim steps: {config.total_sim_steps}")
