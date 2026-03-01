from dataclasses import dataclass
from enum import Enum
from typing import Literal
from typing import Optional

import newton


class SyncMode(Enum):
    ALIGN_FPS_TO_DT = 1
    ALIGN_DT_TO_FPS = 2


@dataclass
class SimulationConfig:
    """Parameters defining the simulation's timeline."""

    duration_seconds: float = 3.0
    target_timestep_seconds: float = 1e-3
    num_worlds: int = 1
    sync_mode: SyncMode = SyncMode.ALIGN_FPS_TO_DT


@dataclass
class RenderingConfig:
    """Parameters for rendering the simulation to a USD file."""

    vis_type: Literal["gl", "usd", "null", None] = "gl"
    target_fps: int | None = 30
    usd_file: str | None = "sim.usd"
    usd_scaling: float | None = 100.0
    start_paused: bool = True
    # GL viewer: where the scene is placed (spawn location). Default (20, 20, 0) if None.
    world_offsets: Optional[tuple[float, float, float]] = None
    # GL viewer: initial camera position (x, y, z) and heading (pitch, yaw in degrees). If None, viewer defaults are used.
    camera_pos: Optional[tuple[float, float, float]] = None
    camera_pitch: Optional[float] = None
    camera_yaw: Optional[float] = None

    def create_viewer(self, model: newton.Model, num_segments: int | None):
        """
        Factory method to create the appropriate viewer instance.
        """
        if self.vis_type == "usd":
            return newton.viewer.ViewerUSD(
                output_path=self.usd_file,
                fps=self.target_fps,
                up_axis="Z",
                num_frames=num_segments,
            )
        elif self.vis_type == "gl":
            return newton.viewer.ViewerGL()
        elif self.vis_type == "null" or self.vis_type is None:
            return newton.viewer.ViewerNull(num_segments)
        else:
            raise ValueError(f"Unsupported rendering type: {self.vis_type}")


@dataclass
class ExecutionConfig:
    """Parameters controlling the performance and execution strategy."""

    use_cuda_graph: bool = True
    headless_steps_per_segment: int = 10
