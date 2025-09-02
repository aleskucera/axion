import numpy as np
import openmesh
import warp as wp
import warp.sim.render
from axion import AxionEngine
from axion import EngineConfig
from axion import HDF5Logger
from tqdm import tqdm

# Options
RENDER = True
USD_FILE = "mesh.usd"
DEBUG = True
PROFILE_SYNC = False
PROFILE_NVTX = False
PROFILE_CUDA_TIMELINE = False

FRICTION = 0.8
RESTITUTION = 0.6

# Optional micro-profiling visibility settings
np.set_printoptions(suppress=False, precision=2)
if PROFILE_CUDA_TIMELINE:
    cuda_activity_filter = wp.TIMING_ALL
else:
    cuda_activity_filter = 0


def ball_world_model(gravity: bool = True) -> wp.sim.Model:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

    surface_m = openmesh.read_trimesh("data/helhest/wheel2.obj")
    mesh_points = np.array(surface_m.points())
    mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()
    surface_mesh = wp.sim.Mesh(mesh_points, mesh_indices)

    wheel_m_col = openmesh.read_trimesh("data/helhest/wheel_collision.obj")
    mesh_points = np.array(wheel_m_col.points())
    mesh_indices = np.array(wheel_m_col.face_vertex_indices(), dtype=np.int32).flatten()
    wheel_mesh_col = wp.sim.Mesh(mesh_points, mesh_indices)

    surface = builder.add_body(
        origin=wp.transform(
            (0.0, 0.0, 1.5), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.radians(15.0))
        ),
        name="cube",
    )
    builder.add_shape_mesh(
        body=surface,
        mesh=surface_mesh,
        density=10.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        has_ground_collision=False,
        has_shape_collision=False,
    )
    builder.add_shape_mesh(
        body=surface,
        mesh=wheel_mesh_col,
        density=10.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        is_visible=False,
    )

    # wheel2 = builder.add_body(
    #     origin=wp.transform(
    #         (0.0, 0.0, 3.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.radians(15.0))
    #     ),
    #     name="cube",
    # )
    # builder.add_shape_mesh(
    #     body=wheel2,
    #     mesh=wheel_mesh,
    #     density=10.0,
    #     mu=FRICTION,
    #     restitution=RESTITUTION,
    #     thickness=0.0,
    #     has_ground_collision=False,
    #     has_shape_collision=False,
    # )
    # builder.add_shape_mesh(
    #     body=wheel2,
    #     mesh=wheel_mesh_col,
    #     density=10.0,
    #     mu=FRICTION,
    #     restitution=RESTITUTION,
    #     thickness=0.0,
    #     is_visible=False,
    # )
    #
    # ball1 = builder.add_body(origin=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), name="ball1")
    # builder.add_shape_sphere(
    #     body=ball1,
    #     radius=1.0,
    #     density=10.0,
    #     ke=2000.0,
    #     kd=10.0,
    #     kf=200.0,
    #     mu=FRICTION,
    #     restitution=RESTITUTION,
    #     thickness=0.0,
    # )

    surface_m = openmesh.read_trimesh("data/surface.obj")
    mesh_points = np.array(surface_m.points())
    mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()
    surface_mesh = wp.sim.Mesh(mesh_points, mesh_indices)

    # surface = builder.add_body(
    #     origin=wp.transform(
    #         (0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.radians(0.0))
    #     ),
    #     name="surface",
    # )
    builder.add_shape_mesh(
        body=-1,
        mesh=surface_mesh,
        density=10.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        has_ground_collision=False,
    )

    # builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=FRICTION, restitution=RESTITUTION)
    model = builder.finalize()
    # model.rigid_contact_max = 400
    return model


class BallBounceSim:
    def __init__(self):
        # Time & simulation config
        self.fps = 30
        self.num_frames = 90
        self.sim_substeps = 20
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        self.sim_steps = int(self.sim_duration // self.sim_dt)

        self.model = ball_world_model(gravity=True)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps + 1)
        self._timestep = 0

        self.logger = HDF5Logger("ball_bounce_log.h5") if DEBUG else None

        engine_config = EngineConfig(newton_iters=8, linear_iters=4, linesearch_steps=0)

        self.integrator = AxionEngine(self.model, engine_config, logger=self.logger)
        self.renderer = wp.sim.render.SimRenderer(self.model, USD_FILE, scaling=100.0, fps=self.fps)

        # Alloc states and controls
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.use_cuda_graph = wp.get_device().is_cuda and not DEBUG
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.step()
            self.step_graph = capture.graph

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

    def simulate(self):
        self._timestep = 0
        frame_interval = 1.0 / self.fps
        last_rendered_time = 0.0

        with self.logger if self.logger else open("/dev/null") as _:

            for i in tqdm(range(self.sim_steps), desc="Simulating", disable=DEBUG):
                with wp.ScopedTimer(
                    "step",
                    active=DEBUG,
                    synchronize=PROFILE_SYNC,
                    use_nvtx=PROFILE_NVTX,
                    cuda_filter=cuda_activity_filter,
                ):
                    if self.use_cuda_graph:
                        wp.capture_launch(self.step_graph)
                    else:
                        self.step()

                if RENDER:
                    with wp.ScopedTimer("render", active=DEBUG):
                        wp.synchronize()
                        t = self.time[self._timestep]
                        if t >= last_rendered_time:  # render only if enough time has passed
                            self.renderer.begin_frame(t)
                            self.renderer.render(self.state_0)
                            self.renderer.end_frame()
                            last_rendered_time += frame_interval  # update to next frame time

                self._timestep += 1

            if RENDER:
                self.renderer.save()

    def simulate_multistep(self):
        t = 0.0
        frame_interval = 1.0 / self.fps
        last_rendered_time = 0.0
        with self.logger if self.logger else open("/dev/null") as _:
            self.log_model()
            for _ in tqdm(range(self.num_frames), desc="Simulating", disable=DEBUG):
                with wp.ScopedTimer(
                    "step",
                    active=DEBUG,
                    synchronize=PROFILE_SYNC,
                    use_nvtx=PROFILE_NVTX,
                    cuda_filter=cuda_activity_filter,
                ):
                    if self.use_cuda_graph:
                        wp.capture_launch(self.step_graph)
                    else:
                        self.multistep()

                t += self.frame_dt
                if RENDER and t >= last_rendered_time:
                    with wp.ScopedTimer(
                        "render",
                        active=DEBUG,
                        synchronize=PROFILE_SYNC,
                        use_nvtx=PROFILE_NVTX,
                        cuda_filter=cuda_activity_filter,
                    ):
                        self.renderer.begin_frame(t)
                        self.renderer.render(self.state_0)
                        self.renderer.end_frame()
                    last_rendered_time += frame_interval

            if RENDER:
                self.renderer.save()


def ball_bounce_simulation():
    sim = BallBounceSim()
    sim.simulate()


if __name__ == "__main__":
    ball_bounce_simulation()
