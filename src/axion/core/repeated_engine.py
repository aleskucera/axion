from typing import Optional

import numpy as np
import warp as wp
from newton import Contacts
from newton import Control
from newton import Model
from newton import State
from .base_engine import AxionEngineBase
from .engine_config import AxionEngineConfig
from .logging_config import LoggingConfig


class RepeatedAxionEngine(AxionEngineBase):
    def __init__(
        self,
        model: Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
        differentiable_simulation: bool = False,
    ):
        super().__init__(model, sim_steps, config, logging_config, differentiable_simulation)
        # Filled each `step()` for external logging (e.g. HDF5 dumps). CPU numpy copies.
        self._repeated_step_log: dict[str, np.ndarray] = {}

    def _snapshot_init_guess_after_warm_start(self) -> None:
        """State that becomes the initial guess for the second Newton solve."""
        wp.synchronize()
        self._repeated_step_log["init_guess_body_pose_after_warm_start"] = (self.data.body_pose.numpy().copy())
        self._repeated_step_log["init_guess_body_vel_after_warm_start"] = (self.data.body_vel.numpy().copy())
        self._repeated_step_log["init_guess_constr_force_after_warm_start"] = (self.data._constr_force.numpy().copy())

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        self._repeated_step_log = {}

        # Process Newton Contacts -> Axion Contacts
        self.load_data(state_in, control, contacts, dt)

        # Initial guess #1: use current state (zero-order warm start)
        wp.copy(dest=self.data.body_pose, src=state_in.body_q)
        wp.copy(dest=self.data.body_vel, src=state_in.body_qd)
        self.data._constr_force.zero_()
        self.data._constr_force_prev_iter.zero_()

        # Call Newton solver 1st time
        self._solve()

        # Initial guess #2:  No need to set body_pose and body_vel, just compute warm start forces
        # Warm-start constraint forces for the second NR (body_pose / body_vel = 1st solve result)
        self.compute_warm_start_forces()
        self._snapshot_init_guess_after_warm_start()

        # Call Newton solver 2nd time (`iter_count` after this is for NR #2 only; see `_solve()`)
        self._solve()

        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)
