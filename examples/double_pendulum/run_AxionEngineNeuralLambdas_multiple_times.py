from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import override

import hydra
import newton
from hydra import compose
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Add local source roots to sys.path so the script can be run directly.
_EXAMPLES_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = REPO_ROOT / "src"
for _path in (_EXAMPLES_DIR, _SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from axion import EngineConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.neural_solver.train.trained_models.selected_trained_models import CONTACT_MODELS
from pendulum_articulation_definition import build_pendulum_model
from pendulum_utils import generalized_to_maximal
from pendulum_utils import set_tilted_plane_from_coefficients

# ---------------------------------------------------------------------------
# USER-FACING KNOBS - edit these
# ---------------------------------------------------------------------------
NEURAL_MODELS = CONTACT_MODELS
INITIAL_STATE: tuple[float, float, float, float] | None = (
    -0.5704,
    2.8907,
    -3.6530,
    -7.6918,
)
PLANE_COEFFICIENTS = (-0.2354, -0.0000, 0.9719, -2.3318)
N_STEPS = 300

OUTPUT_DIR = REPO_ROOT / "data" / "logs" / "axionEngineWithNeuralLambdasComparison" / "run2"
# ---------------------------------------------------------------------------

CONFIG_PATH = _EXAMPLES_DIR.parent / "conf"


class Simulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        plane_coefficients: tuple[float, float, float, float],
        initial_state: tuple[float, float, float, float] | None,
    ):
        self.plane_coefficients = plane_coefficients
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )
        if initial_state is not None:
            q0, q1, qd0, qd1 = initial_state
            generalized_to_maximal(
                self.model,
                self.current_state,
                q0=q0,
                q1=q1,
                qd0=qd0,
                qd1=qd1,
            )

    @override
    def control_policy(self, state: newton.State):
        pass

    @override
    def _render(self, segment_num: int):
        """Minimal render for ViewerNull while still letting the run loop finish."""
        sim_time = segment_num * self.steps_per_segment * self.clock.dt
        self.viewer.begin_frame(sim_time)
        self.viewer.end_frame()

    def build_model(self) -> newton.Model:
        model = build_pendulum_model(num_worlds=1, device="cuda:0")
        a, b, c, d = self.plane_coefficients
        set_tilted_plane_from_coefficients(model, a, b, c, d, world_idx=0)
        return model


def _load_cfg():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
        return compose(
            config_name="axion_w_neural_lambdas_pendulum",
            overrides=[
                "rendering=headless",
                "simulation.use_cuda_graph=false",
                "project_name=axionEngineWithNeuralLambdasComparison",
            ],
        )


def _safe_model_log_stem(model_path: Path) -> str:
    return "__".join(model_path.parts).replace(" ", "_").replace("/", "__")


def main():
    cfg = _load_cfg()

    base_sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    base_logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    base_engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    dt = base_sim_config.target_timestep_seconds
    sim_config = replace(base_sim_config, duration_seconds=N_STEPS * dt)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for run_idx, neural_model in enumerate(NEURAL_MODELS, start=1):
        neural_model = Path(neural_model)
        log_path = OUTPUT_DIR / f"{run_idx:02d}_{_safe_model_log_stem(neural_model)}.h5"
        print(
            f"[{run_idx}/{len(NEURAL_MODELS)}] Running AxionEngineWithNeuralLambdas "
            f"with model={neural_model}"
        )
        print(f"Writing neural lambda log to: {log_path}")

        engine_config = replace(
            base_engine_config,
            neural_model_dir=str(neural_model),
        )
        logging_config = replace(
            base_logging_config,
            neural_lambdas_log_file=str(log_path),
            neural_lambdas_simulation_steps=N_STEPS,
        )

        simulator = Simulator(
            sim_config=sim_config,
            render_config=render_config,
            engine_config=engine_config,
            logging_config=logging_config,
            plane_coefficients=PLANE_COEFFICIENTS,
            initial_state=INITIAL_STATE,
        )
        simulator.run()


if __name__ == "__main__":
    main()