from typing import Optional

import warp as wp
from newton import Contacts
from newton import Control
from newton import Model
from newton import State

from .base_engine import AxionEngineBase
from .engine_config import AxionEngineConfig
from .logging_config import LoggingConfig


class AxionEngine(AxionEngineBase):
    def __init__(
        self,
        model: Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
        differentiable_simulation: bool = False,
    ):
        super().__init__(model, sim_steps, config, logging_config, differentiable_simulation)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        self.load_data(state_in, control, contacts, dt)

        # Initial guess: use current state (zero-order warm start)
        wp.copy(dest=self.data.body_pose, src=state_in.body_q)
        wp.copy(dest=self.data.body_vel, src=state_in.body_qd)

        self.data._constr_force.zero_()
        self.data._constr_force_prev_iter.zero_()

        self._solve()

        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)
