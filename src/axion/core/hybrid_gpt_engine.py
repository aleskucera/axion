from typing import Optional

from newton import Contacts
from newton import Control
from newton import Model
from newton import State
import warp as wp

from .base_engine import AxionEngineBase
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig

# Neural network imports:
from pathlib import Path
import yaml
import torch
from newton import eval_fk
from axion.neural_solver.standalone.neural_predictor import NeuralPredictor
NN_BASE_PATH = Path.cwd() /"src"/"axion"/"neural_solver"/"train"/"trained_models"/"mse"/"04-24-2026-21-06-20" 
NN_PENDULUM_PT_PATH = NN_BASE_PATH/"nn"/"best_valid_valid_model.pt"
NN_PENDULUM_CFG_PATH = NN_BASE_PATH/"cfg.yaml"


@wp.kernel
def _shift_body_qd_joint_to_com_kernel(
    joint_count: int,
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    raw_body_qd: wp.array(dtype=wp.spatial_vector),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    # Newton's eval_fk writes body_qd whose linear part is the velocity at the
    # parent-side joint anchor; the Axion solver expects it at the body CoM.
    #
    # This must be applied in model topology order because child raw velocities
    # include parent-anchor velocities derived from the parent's raw FK velocity.
    # Replace that stale parent contribution with the already-corrected parent
    # CoM velocity before shifting the child velocity to its CoM.
    for i in range(joint_count):
        parent = joint_parent[i]
        child = joint_child[i]
        if child < 0:
            continue

        X_pj = joint_X_p[i]
        if parent >= 0:
            X_wpj = body_q[parent] * X_pj
        else:
            X_wpj = X_pj
        parent_anchor = wp.transform_get_translation(X_wpj)

        raw_child = raw_body_qd[child]
        child_linear_at_anchor = wp.spatial_top(raw_child)
        child_omega = wp.spatial_bottom(raw_child)

        corrected_parent_anchor_linear = wp.vec3(0.0)
        relative_linear = child_linear_at_anchor
        if parent >= 0:
            parent_com = wp.transform_point(body_q[parent], body_com[parent])

            corrected_parent = body_qd[parent]
            corrected_parent_anchor_linear = wp.spatial_top(corrected_parent) + wp.cross(
                wp.spatial_bottom(corrected_parent), parent_anchor - parent_com
            )

            raw_parent = raw_body_qd[parent]
            raw_parent_anchor_linear = wp.spatial_top(raw_parent) + wp.cross(
                wp.spatial_bottom(raw_parent), parent_anchor - parent_com
            )
            relative_linear = child_linear_at_anchor - raw_parent_anchor_linear

        child_com = wp.transform_point(body_q[child], body_com[child])
        child_linear_at_com = (
            corrected_parent_anchor_linear
            + relative_linear
            + wp.cross(child_omega, child_com - parent_anchor)
        )
        body_qd[child] = wp.spatial_vector(child_linear_at_com, child_omega)


class HybridGPTEngine(AxionEngineBase):
    """
    Engine that uses GPT to predict initial guess for the Newton-Raphson solver.
    After that, it uses AxionEngine to solve the system of equations.
    """

    def __init__(
        self,
        model: Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
        differentiable_simulation: bool = False,
    ):
        super().__init__(model, sim_steps, config, logging_config, differentiable_simulation)

        #########################################
        #  Neural network initialization
        #########################################

        print("GPTEngine is using the device = ", self.device)
        nn_model_path = NN_PENDULUM_PT_PATH
        nn_cfg_path = NN_PENDULUM_CFG_PATH

        # Load the nn .pt file and .cfg file correctly
        print(f"Loading model from: {nn_model_path}")
        loaded_nn_model, robot_name = torch.load(nn_model_path, map_location= str(self.device), weights_only= False)
        print(f"Loaded model for robot: {robot_name}")
        print(f"Loading configuration from: {nn_cfg_path}")
        with open(nn_cfg_path, 'r') as f:
            loaded_nn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # Initialize NeRDPredictor: robot config is inferred from self.model (newton.Model)
        self.nn_predictor = NeuralPredictor(
            newton_model=self.model,
            nn_model=loaded_nn_model,
            nn_cfg=loaded_nn_cfg,
            device=str(self.device),
        )
        # Exposed for external diagnostics capture (e.g., engine comparison scripts).
        self.last_predicted_next_lambdas = None
        self.last_predicted_next_body_pose = None
        self.last_predicted_next_body_vel = None

    def _shift_body_qd_to_com_frame(self, state: State) -> None:
        """Convert ``state.body_qd`` from the parent-side joint-anchor frame
        produced by :func:`eval_fk` to the CoM frame expected by the Axion solver.

        Operates in place on ``state.body_qd`` using the corresponding
        ``state.body_q`` and the model's joint topology.
        """
        raw_body_qd = wp.empty_like(state.body_qd)
        wp.copy(dest=raw_body_qd, src=state.body_qd)

        wp.launch(
            kernel=_shift_body_qd_joint_to_com_kernel,
            dim=1,
            inputs=[
                self.model.joint_count,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.body_com,
                state.body_q,
                raw_body_qd,
            ],
            outputs=[state.body_qd],
            device=self.device,
        )

    def _neural_init_state_fn(
        self,
        state_in: State,
        state_out: State,
        axion_contacts: Contacts,
        dt: float,
    ) -> None:
        """
        Perform neural network model inference to get an initial guess for the Newton method.
        For MSEModel: extracts both state and lambda predictions from the joint regression output.
        """
        # Process inputs: coordinate frame conversion, state embedding.
        self.nn_predictor.process_inputs(state_in, axion_contacts, dt)

        # Trigger neural network inference:
        next_states, next_lambdas = self.nn_predictor.predict(dt)
        if next_lambdas is not None:
            # Remove tiny neural lambda predictions so near-inactive constraints
            # warm-start from exactly zero force.
            next_lambdas = torch.where(
                torch.abs(next_lambdas) < 0.01,
                torch.zeros_like(next_lambdas),
                next_lambdas,
            )
            next_lambdas[..., 11:] = 0.0

        dof_q = self.nn_predictor.dof_q_per_env
        dof_qd = self.nn_predictor.dof_qd_per_env
        pred_joint_q = wp.from_torch(next_states[0, :dof_q].reshape(dof_q,).contiguous())
        pred_joint_qd = wp.from_torch(next_states[0, dof_q:dof_q + dof_qd].reshape(dof_qd,).contiguous())

        # Perform FK: joint_q -> body_q
        wp.copy(dest=state_out.joint_q, src=pred_joint_q)
        wp.copy(dest=state_out.joint_qd, src=pred_joint_qd)
        eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)

        # Newton's eval_fk produces body_qd in the parent-side joint-anchor frame,
        # but the Axion solver represents body_vel at the CoM. Shift in place so
        # both the diagnostic capture below and the warm-start copy into
        # self.data.body_vel are consistent with the engine's CoM-frame convention.
        self._shift_body_qd_to_com_frame(state_out)

        #-This part is only for exposing it to test_engines.py ---------------------------
        self.last_predicted_next_body_pose = state_out.body_q.numpy().copy()
        self.last_predicted_next_body_vel = state_out.body_qd.numpy().copy()
        if next_lambdas is None:
            self.last_predicted_next_lambdas = None
        else:
            self.last_predicted_next_lambdas = next_lambdas.detach().cpu().numpy().copy()
        #---------------------------------------------------------------------------------

        # Transfer neural prediction of states into solver's working arrays:
        wp.copy(dest=self.data.body_pose, src=state_out.body_q)
        wp.copy(dest=self.data.body_vel, src=state_out.body_qd)

        # Initial guess of lambda (constraint forces).
        if next_lambdas is not None and getattr(self.config, "use_neural_lambda_init", True):
            lambdas_wp = wp.from_torch(next_lambdas.squeeze(0).contiguous())
            wp.copy(dest=self.data._constr_force, src=lambdas_wp)
            wp.copy(dest=self.data._constr_force_prev_iter, src=lambdas_wp)
        elif getattr(self.config, "use_warm_start_forces", False):
            self.compute_warm_start_forces()
        else:
            self.data._constr_force.zero_()
            self.data._constr_force_prev_iter.zero_()

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        # Process Newton Contacts -> Axion Contacts
        self.load_data(state_in, control, contacts, dt)

        # Perform neural network model inference to get a initial guess for the Newton method.
        self._neural_init_state_fn(state_in, state_out, self.axion_contacts, dt)

        # Call Newton solver
        self._solve()
        wp.copy(dest=state_out.body_q, src=self.data.body_pose)
        wp.copy(dest=state_out.body_qd, src=self.data.body_vel)

