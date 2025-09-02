import math
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import warp as wp
import warp.sim.render
from axion import AxionEngine
from axion import EngineConfig
from tqdm import tqdm


@dataclass
class SimConfig:
    """Configuration for the core physics properties of the simulation."""

    # The total time the simulation should run, in seconds. This is the user's
    # target, which may be slightly adjusted for performance optimization.
    sim_duration: float = 3.0
    # The desired physics timestep (dt), in seconds. This may be slightly adjusted
    # when rendering to synchronize with the framerate.
    target_sim_dt: float = 1e-3


@dataclass
class RenderConfig:
    """Configuration for visual output of the simulation."""

    # If True, the simulation will be rendered to a USD file.
    enable: bool = True
    # Target frames per second for the rendered output.
    fps: int = 30
    # Scaling factor applied to the scene for rendering purposes.
    scaling: float = 100.0
    # File path for the output Universal Scene Description (USD) file.
    usd_file: str = "sim.usd"


@dataclass
class ExecConfig:
    """Configuration for the execution strategy and performance tuning."""

    # If True, attempts to use CUDA Graphs to accelerate repeated computations.
    # This provides a major speedup by reducing CPU overhead.
    use_cuda_graph: bool = True
    # When in headless mode, this is the number of physics steps to batch
    # together into a single CUDA graph. Larger values can improve performance.
    headless_graph_steps: int = 10


@dataclass
class ProfileConfig:
    """Configuration for debugging and performance profiling."""

    # Enables verbose printing and disables optimizations for easier debugging.
    debug: bool = False
    # Forces CPU-GPU synchronization, useful for accurate timing.
    sync: bool = False
    # Enables NVTX markers for profiling with tools like NVIDIA Nsight.
    nvtx: bool = False
    # Enables CUDA timeline activity collection in Warp timers.
    cuda_timeline: bool = False


class BaseSimulator(ABC):
    """
    An abstract base class for running GPU-accelerated physics simulations using NVIDIA Warp.

    This class provides a robust framework that handles:
    - Both rendering and headless (non-rendering) simulation modes.
    - Automatic synchronization of physics timesteps with rendering framerates.
    - A "no remainders" policy for CUDA graphs in headless mode, ensuring maximum performance
      by adjusting simulation length to be a perfect multiple of the graph size.
    """

    def __init__(
        self,
        sim_config: SimConfig,
        render_config: RenderConfig,
        exec_config: ExecConfig,
        profile_config: ProfileConfig,
        engine_config: EngineConfig,
        logger: Optional[object] = None,
    ):
        """
        Initializes the simulator by resolving timing, building the model, and setting up the engine.

        Args:
            sim_config: Core physics parameters (duration, timestep).
            render_config: Visual output settings.
            exec_config: Performance tuning and execution strategy.
            profile_config: Debugging and profiling options.
            engine_config: Configuration for the underlying Axion physics engine.
            logger: An optional logger object for external logging.
        """
        self.sim_config = sim_config
        self.render_config = render_config
        self.exec_config = exec_config
        self.profile_config = profile_config
        self.engine_config = engine_config
        self.logger = logger

        # This private method resolves all timing and step counts based on the configs.
        self._resolve_timing_parameters()

        # Build the user-defined model (e.g., robots, objects).
        self.model = self.build_model()
        self.integrator = AxionEngine(self.model, self.engine_config, logger=self.logger)

        self.state_current = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()

        # Initialize the renderer only if enabled.
        self.renderer: Optional[wp.sim.render.SimRenderer] = None
        if self.render_config.enable:
            self.renderer = wp.sim.render.SimRenderer(
                self.model,
                self.render_config.usd_file,
                scaling=self.render_config.scaling,
                fps=self.render_config.fps,
            )

        self.graph = None
        self.use_cuda_graph = (
            self.exec_config.use_cuda_graph
            and wp.get_device().is_cuda
            and not self.profile_config.debug
        )

    def simulate_steps(self):
        if self.use_cuda_graph:
            if self.graph is None:
                with wp.ScopedCapture() as capture:
                    for _ in range(self.graph_capture_steps):
                        self._sim_step()
                self.graph = capture.graph
            else:
                wp.capture_launch(self.graph)
        else:
            self._sim_step()

    def _resolve_timing_parameters(self):
        """
        Calculates all operational timing parameters (dt, step counts, duration)
        based on the user's configuration, enforcing all performance and sync policies.
        """
        if self.render_config.enable:
            # --- Rendering Mode ---
            frame_dt = 1.0 / self.render_config.fps
            ideal_substeps = frame_dt / self.sim_config.target_sim_dt
            self.substeps_per_frame = round(ideal_substeps) or 1
            self.actual_sim_dt = frame_dt / self.substeps_per_frame
            self.num_frames = math.ceil(self.sim_config.sim_duration / frame_dt)
            self.total_sim_steps = self.num_frames * self.substeps_per_frame
            self.actual_sim_duration = self.total_sim_steps * self.actual_sim_dt

            adj_ratio = (
                abs(self.actual_sim_dt - self.sim_config.target_sim_dt)
                / self.sim_config.target_sim_dt
            )
            if adj_ratio > 0.01:
                print(f"Info: Target sim_dt adjusted to {self.actual_sim_dt:.6f}s for rendering.")
        else:
            # --- Headless Mode ---
            self.actual_sim_dt = self.sim_config.target_sim_dt
            initial_total_steps = math.ceil(self.sim_config.sim_duration / self.actual_sim_dt)

            # Enforce "no remainders" policy for CUDA graphs.
            if self.exec_config.use_cuda_graph and self.exec_config.headless_graph_steps > 0:
                graph_size = self.exec_config.headless_graph_steps
                # Round the number of chunks UP to ensure we simulate for at least the requested duration.
                num_chunks = math.ceil(initial_total_steps / graph_size)
                self.total_sim_steps = num_chunks * graph_size
            else:
                self.total_sim_steps = initial_total_steps

            self.actual_sim_duration = self.total_sim_steps * self.actual_sim_dt
            self.substeps_per_frame = 0
            self.num_frames = 0

            # Notify user if simulation duration was adjusted for performance.
            if not math.isclose(self.actual_sim_duration, self.sim_config.sim_duration):
                print(
                    f"Info: Sim duration adjusted from {self.sim_config.sim_duration:.4f}s to "
                    f"{self.actual_sim_duration:.4f}s to align with CUDA graph chunks."
                )

    @property
    def graph_capture_steps(self) -> int:
        """Determines how many steps to capture in the CUDA graph based on the mode."""
        if self.render_config.enable:
            return self.substeps_per_frame
        else:
            return self.exec_config.headless_graph_steps

    def simulate(self):
        """Runs the full simulation loop according to the chosen configuration."""
        if self.render_config.enable:
            self._simulate_with_rendering()
        else:
            self._simulate_headless()

    def _simulate_headless(self):
        """Runs the simulation without rendering, optimized for performance."""
        print(
            f"Running headless simulation for {self.total_sim_steps} steps ({self.actual_sim_duration:.2f}s)."
        )

        if self.use_cuda_graph:
            graph_chunk_size = self.exec_config.headless_graph_steps
            num_chunks = self.total_sim_steps // graph_chunk_size

            print(f"Executing {num_chunks} CUDA graph chunk(s) of {graph_chunk_size} steps each.")
            for _ in tqdm(
                range(num_chunks),
                desc="Simulating (Graph Chunks)",
                disable=self.profile_config.debug,
            ):
                self.simulate_steps()
        else:
            # Fallback for no CUDA graph: run step-by-step
            for _ in tqdm(
                range(self.total_sim_steps),
                desc="Simulating (Headless)",
                disable=self.profile_config.debug,
            ):
                self.simulate_steps()

        wp.synchronize()
        print("Headless simulation complete.")

    def _simulate_with_rendering(self):
        """Runs the simulation and renders frames to a USD file."""
        print(f"Running simulation for {self.num_frames} frames ({self.actual_sim_duration:.2f}s).")
        for frame_idx in tqdm(
            range(self.num_frames), desc="Simulating (Render)", disable=self.profile_config.debug
        ):
            if self.use_cuda_graph:
                self.simulate_steps()
            else:
                for _ in range(self.substeps_per_frame):
                    self.simulate_steps()

            wp.synchronize()
            time = (frame_idx + 1) * (1.0 / self.render_config.fps)
            self.renderer.begin_frame(time)
            self.renderer.render(self.state_current)
            self.renderer.end_frame()

        self.renderer.save()
        print(f"Rendering complete. Output saved to {self.render_config.usd_file}")

    @abstractmethod
    def build_model(self) -> wp.sim.Model:
        """
        Builds the physics model for the simulation.

        This method MUST be implemented by any subclass. It should define all the
        rigid bodies, joints, and other physical properties of the scene.
        """
        pass

    def _sim_step(self):
        """Performs a single, fundamental physics step of the simulation."""
        wp.sim.collide(self.model, self.state_current)
        self.integrator.simulate(
            self.model, self.state_current, self.state_next, self.actual_sim_dt, self.control
        )
        wp.copy(dest=self.state_current.body_q, src=self.state_next.body_q)
        wp.copy(dest=self.state_current.body_qd, src=self.state_next.body_qd)
