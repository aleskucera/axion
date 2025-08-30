# base_sim.py
import numpy as np
import warp as wp
import warp.sim.render
from axion import AxionEngine
from axion import EngineConfig
from axion import HDF5Logger
from tqdm import tqdm


class BaseSimulator:
    def __init__(
        self,
        fps=30,
        num_frames=90,
        sim_substeps=10,
        usd_file="sim.usd",
        debug=False,
        profile_sync=False,
        profile_nvtx=False,
        profile_cuda_timeline=False,
        friction=0.8,
        restitution=1.0,
        render=True,
        use_multistep=False,  # Flag to choose between step or multistep
        logger=None,
    ):
        self.fps = fps
        self.num_frames = num_frames
        self.sim_substeps = sim_substeps
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        self.sim_steps = int(self.sim_duration // self.sim_dt)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps + 1)
        self._timestep = 0

        self.DEBUG = debug
        self.PROFILE_SYNC = profile_sync
        self.PROFILE_NVTX = profile_nvtx
        self.PROFILE_CUDA_TIMELINE = profile_cuda_timeline
        self.FRICTION = friction
        self.RESTITUTION = restitution
        self.RENDER = render
        self.use_multistep = use_multistep

        if self.PROFILE_CUDA_TIMELINE:
            self.cuda_activity_filter = wp.TIMING_ALL
        else:
            self.cuda_activity_filter = 0

        np.set_printoptions(suppress=False, precision=2)

        self.model = self.build_model()
        self.logger = logger if self.DEBUG else None

        self.engine_config = self.get_engine_config()
        self.integrator = AxionEngine(self.model, self.engine_config, logger=self.logger)
        self.renderer = wp.sim.render.SimRenderer(self.model, usd_file, scaling=100.0, fps=self.fps)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.use_cuda_graph = wp.get_device().is_cuda and not self.DEBUG
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                if self.use_multistep:
                    self.multistep()
                else:
                    self.step()
            self.step_graph = capture.graph

    def build_model(self) -> wp.sim.Model:
        raise NotImplementedError("Subclasses must implement model building.")

    def get_engine_config(self) -> EngineConfig:
        # Default config; subclasses can override
        return EngineConfig(newton_iters=16, linear_iters=4, linesearch_steps=1)

    def log_model(self):
        if self.logger:
            with self.logger.scope("simulation_info"):
                self.logger.log_scalar("fps", self.fps)
                self.logger.log_scalar("num_frames", self.num_frames)
                self.logger.log_scalar("sim_substeps", self.sim_substeps)
                self.logger.log_scalar("frame_dt", self.frame_dt)
                self.logger.log_scalar("sim_dt", self.sim_dt)
                self.logger.log_scalar("sim_duration", self.sim_duration)

            with self.logger.scope("model_info"):
                m = self.model
                self.logger.log_scalar("body_count", m.body_count)
                self.logger.log_scalar("joint_count", m.joint_count)
                self.logger.log_scalar("rigid_contact_max", m.rigid_contact_max)

    def step(self):
        with (
            self.logger.scope(f"timestep_{self._timestep:04d}")
            if self.logger
            else open("/dev/null")
        ) as _:
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(
                self.model,
                self.state_0,
                self.state_1,
                self.sim_dt,
                self.control,
            )
            wp.copy(dest=self.state_0.body_q, src=self.state_1.body_q)
            wp.copy(dest=self.state_0.body_qd, src=self.state_1.body_qd)
            self._timestep += 1

    def multistep(self):
        for _ in range(self.sim_substeps):
            with (
                self.logger.scope(f"timestep_{self._timestep:04d}")
                if self.logger
                else open("/dev/null")
            ) as _:
                if self.logger:
                    self.logger.log_attribute("time", self.time[self._timestep])
                    self.logger.log_scalar("time", self.time[self._timestep])
                wp.sim.collide(self.model, self.state_0)
                self.integrator.simulate(
                    self.model,
                    self.state_0,
                    self.state_1,
                    self.sim_dt,
                    control=self.control,
                )
                wp.copy(dest=self.state_0.body_q, src=self.state_1.body_q)
                wp.copy(dest=self.state_0.body_qd, src=self.state_1.body_qd)
                self._timestep += 1

    def _perform_step(self):
        if self.use_cuda_graph:
            wp.capture_launch(self.step_graph)
        elif self.use_multistep:
            self.multistep()
        else:
            self.step()

    def simulate(self):
        self._timestep = 0
        frame_interval = 1.0 / self.fps
        last_rendered_time = 0.0

        with self.logger if self.logger else open("/dev/null") as _:
            self.log_model()

            loop_range = range(self.sim_steps) if not self.use_multistep else range(self.num_frames)
            for i in tqdm(loop_range, desc="Simulating", disable=self.DEBUG):
                with wp.ScopedTimer(
                    "step",
                    active=self.DEBUG,
                    synchronize=self.PROFILE_SYNC,
                    use_nvtx=self.PROFILE_NVTX,
                    cuda_filter=self.cuda_activity_filter,
                ):
                    self._perform_step()

                if self.RENDER:
                    t = (
                        self.time[self._timestep - 1]
                        if not self.use_multistep
                        else (i + 1) * self.frame_dt
                    )
                    if t >= last_rendered_time:
                        with wp.ScopedTimer(
                            "render",
                            active=self.DEBUG,
                            synchronize=self.PROFILE_SYNC,
                            use_nvtx=self.PROFILE_NVTX,
                            cuda_filter=self.cuda_activity_filter,
                        ):
                            wp.synchronize() if not self.use_multistep else None
                            self.renderer.begin_frame(t)
                            self.renderer.render(self.state_0)
                            self.renderer.end_frame()
                        last_rendered_time += frame_interval

        if self.RENDER:
            self.renderer.save()
