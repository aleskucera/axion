# typing and paths
from pathlib import Path
# from typing import Callable
from typing import Optional

# numpy and torch
import numpy as np

# warp and newton
import warp as wp
import newton
from newton.solvers import SolverBase
# from newton import Contacts
from newton import Model
# from newton import State

# classic axion
# from .engine_config import AxionEngineConfig
from .engine_config import NerdEngineConfig
from .engine_logger import EngineLogger
from axion.types import contact_penetration_depth_kernel, reorder_ground_contacts_kernel
# from axion.optim import JacobiPreconditioner
# from axion.optim import MatrixFreeSystemOperator
# from axion.optim import MatrixSystemOperator
# from axion.types import compute_joint_constraint_offsets
# from .control_utils import apply_control
# from .engine_data import create_engine_arrays
# from .engine_dims import EngineDimensions

# axion - nn prediction
import yaml
import torch
import sys
from axion.nn_prediction.nerd_predictor import NeRDPredictor
from axion.nn_prediction import models, utils
from axion.nn_prediction.utils.analysis_utils import write_state_to_csv

sys.modules['models'] = models
sys.modules['utils'] = utils

NERD_BASE_PATH = Path.cwd() / "src" / "axion" / "nn_prediction" / "trained_models" / "NeRD_pretrained" / "pendulum" 
NERD_PENDULUM_PT_PATH = NERD_BASE_PATH / "model.pt"
NERD_PENDULUM_CFG_PATH = NERD_BASE_PATH / "cfg.yaml"
 
class NerdEngine(SolverBase):
    """
    This class implements a neural physics solver.
    It predicts the next step of the simulation based on 
    a trained neural network.
    
    WARNING: This physics solver is robot-model-dependent.
    """

    def __init__(self, 
        model: Model,
        #init_state_fn: Callable[[State, State, Contacts, float], None],
        logger: EngineLogger,
        config: Optional[NerdEngineConfig] = NerdEngineConfig(),
        nn_model_path: Path = NERD_PENDULUM_PT_PATH ,
        nn_cfg_path: Path = NERD_PENDULUM_CFG_PATH,
        ):
        """
        Initialize the neural engine for the given robot model, neural network model and neural network configuration.

        Args:
            model: The warp.sim.Model physics model containing bodies, joints, and other physics properties.
            config: Configuration parameters for the engine of type EngineConfig.
            logger: Optional HDF5Logger or NullLogger for recording simulation data.
            nn_model_path: Filepath to the desired .pt file of the neural network.
            nn_cfg_path: Filepath to the desired .yaml configuration file of the neural network. 
        """

        # TO-DO: add some assertions if model != nn_model_path != nn_cfg_path

        #########################################
        #  Classic AxionEngine-like initialization
        #########################################

        super().__init__(model)

        #self.init_state_fn = init_state_fn
        self.logger = logger
        self.config = config
        self.model = model

        # joint_constraint_offsets, num_constraints = compute_joint_constraint_offsets(
        #     model.joint_type,
        # )

        # self.dims = EngineDimensions(
        #     body_count=self.model.body_count,
        #     contact_count=self.model.rigid_contact_max,
        #     joint_count=self.model.joint_count,
        #     linesearch_step_count=self.config.linesearch_step_count,
        #     joint_constraint_count=num_constraints,
        # )


        # self.data = create_engine_arrays(
        #     self.dims,
        #     self.config,
        #     joint_constraint_offsets,
        #     self.device,
        #     self.logger.uses_dense_matrices,
        #     self.logger.uses_pca_arrays,
        #     self.logger.config.pca_grid_res,
        # )

        # if self.config.matrixfree_representation:
        #     self.A_op = MatrixFreeSystemOperator(
        #         engine=self,
        #         regularization=self.config.regularization,
        #     )
        # else:
        #     self.A_op = MatrixSystemOperator(
        #         engine=self,
        #         regularization=self.config.regularization,
        #     )

        # self.preconditioner = JacobiPreconditioner(self)

        # self.data.set_body_M(model)
        # self.data.set_g_accel(model)

        self.step_cnt = 0

        #########################################
        #  Neural network initialization
        #########################################
        
        print("NerdEngine is using the device = ", self.device)

        # Load the nn .pt file and .cfg file correctly
        print(f"Loading model from: {nn_model_path}")
        loaded_nn_model, robot_name = torch.load(nn_model_path, map_location= str(self.device), weights_only= False)
        print(f"Loaded model for robot: {robot_name}")
        print(f"Loading configuration from: {nn_cfg_path}")
        with open(nn_cfg_path, 'r') as f:
            loaded_nn_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # Initialize NeRDPredictor object with the loaded .pt anf cfg files
        # TO-DO: make the arguments of this initialization not hardcoded
        self.nn_predictor = NeRDPredictor(
            model= loaded_nn_model,
            cfg= loaded_nn_cfg,
            device= str(self.device),
            # Robot configuration for Pendulum
            dof_q_per_env=2,      # 2 revolute joints, each with 1 angle
            dof_qd_per_env=2,     # 2 revolute joints, each with 1 angular velocity
            joint_act_dim=1,      # 1 actuator (typically on first joint)
            num_contacts_per_env=4,  # 1 contact pair (pendulum tip to ground)
            joint_types=[2, 2],             # Joint types: both are REVOLUTE    TO-DO: make them newton enums
            joint_q_start=[0, 1],               # Joint DOF start indices in q vector
            joint_q_end=[1, 2],                 # Joint DOF end indices in q vector
            is_angular_dof=[True, True, True, True],      # Which DOFs are angular (for state embedding)
            is_continuous_dof=[True, True, False, False]  # Which DOFs are continuous (unwrapped angles) Position DOFs (angles) are continuous, velocities are not
        )

        self.num_models = 1    # necessary?

        # NeRD was learned in an environment where Y axis was the up axis
        self.gravity_vector = torch.zeros((self.num_models, 3), device= str(self.device))
        self.gravity_vector[:, 1] = -1.0  # Gravity in negative Y direction

        self.csv_filename = Path(__file__).parent / 'pendulum_states_NerdEngine.csv'

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):
        state_robot_centric = torch.cat( (wp.to_torch(state_in.joint_q), wp.to_torch(state_in.joint_qd)))
        state_robot_centric = state_robot_centric.unsqueeze(0)  # shape (1,4)

        # TO-DO: Add Control from function input
        joint_acts = torch.zeros((self.num_models, 1), device= str(self.device))

        #---Preprocess contacts-------------------------------------------
        max_num_contacts_per_model = 4

        # Reshape shape_body to 2D if needed (Newton provides 1D, kernel expects 2D)
        if len(self.model.shape_body.shape) == 1:
            shape_body_2d = self.model.shape_body.reshape((1, -1))  # (shape_count,) -> (1, shape_count)
        else:
            shape_body_2d = self.model.shape_body

        # Reorder contact data so body is always in position 0, ground in position 1
        reordered_point0 = wp.zeros(
            (self.num_models, max_num_contacts_per_model), 
            dtype=wp.vec3, 
            device=str(self.device)
        )
        reordered_point1 = wp.zeros(
            (self.num_models, max_num_contacts_per_model), 
            dtype=wp.vec3, 
            device=str(self.device)
        )
        reordered_normal = wp.zeros(
            (self.num_models, max_num_contacts_per_model), 
            dtype=wp.vec3, 
            device=str(self.device)
        )
        reordered_thickness0 = wp.zeros(
            (self.num_models, max_num_contacts_per_model), 
            dtype=wp.float32, 
            device=str(self.device)
        )
        reordered_thickness1 = wp.zeros(
            (self.num_models, max_num_contacts_per_model), 
            dtype=wp.float32, 
            device=str(self.device)
        )
        reordered_body_shape = wp.zeros(
            (self.num_models, max_num_contacts_per_model), 
            dtype=wp.int32, 
            device=str(self.device)
        )

        wp.launch(
            kernel=reorder_ground_contacts_kernel,
            dim=(self.num_models, max_num_contacts_per_model),
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                shape_body_2d,
            ],
            outputs=[
                reordered_point0,  # Always body
                reordered_point1,  # Always ground
                reordered_normal,
                reordered_thickness0,  # Always body
                reordered_thickness1,  # Always ground
                reordered_body_shape,  # Body shape index for each contact
            ],
            device=str(self.device)
        )

        # Calculate Penetration depth using reordered contact data
        contact_depths_wp_array = wp.zeros(
            (self.num_models, max_num_contacts_per_model), 
            dtype=wp.float32, 
            device=str(self.device)
        )
        
        # If state_in.body_q is 1D, reshape it
        if len(state_in.body_q.shape) == 1:
            body_q_2d = state_in.body_q.reshape((1, -1))    # (body_count,) -> (1, body_count)
        else:
            body_q_2d = state_in.body_q

        wp.launch(
            kernel=contact_penetration_depth_kernel,
            dim=(self.num_models, max_num_contacts_per_model),
            inputs=[
                body_q_2d,
                shape_body_2d,
                contacts.rigid_contact_count,
                reordered_point0,  # Body points (reordered)
                reordered_point1,  # Ground points (reordered)
                reordered_normal,  # Normal from body to ground (reordered)
                reordered_thickness0,  # Body thickness (reordered)
                reordered_thickness1,  # Ground thickness (reordered)
                reordered_body_shape,  # Body shape indices
            ],
            outputs=[
                contact_depths_wp_array
            ],
            device=str(self.device)
        )

        # Convert to torch if needed for your neural network
        contact_depths = wp.to_torch(contact_depths_wp_array)
        contact_normals = wp.to_torch(reordered_normal).flatten().unsqueeze(0).clone()
        contact_thickness = wp.to_torch(reordered_thickness0).flatten().unsqueeze(0).clone()  # Body thickness
        contact_points_0 = wp.to_torch(reordered_point0).flatten().unsqueeze(0).clone()  # Body points
        contact_points_1 = wp.to_torch(reordered_point1).flatten().unsqueeze(0).clone()  # Ground points
        
        contacts = {
            "contact_normals": contact_normals,
            "contact_depths": contact_depths,
            "contact_thicknesses": contact_thickness,
            "contact_points_0": contact_points_0,
            "contact_points_1": contact_points_1
        }
        #-------------------------------------------------------------------
        
        # Edit 1: switching axis because NeRD had up_axis=Y and axion has up_axis = z
        # Edit 2: Nerd expects root_body_q to be the pos/orient of the first pendulum link, but only its rotational part for some reason? 
        root_body_q = wp.to_torch(state_in.body_q)[0, :].unsqueeze(0)
        root_body_q[0, :3] = torch.tensor([0.0, 0.0, 5.0])
        root_body_q[0, 5] = -root_body_q[0, 4]
        root_body_q[0, 4] = torch.tensor([0.0])

        # Process the inputs (NerdPredictor does this internally using process_inputs)
        self.nn_predictor.process_inputs(
            states= state_robot_centric.clone(),
            joint_acts= joint_acts,
            root_body_q= root_body_q,  # extract only body at index 0, shape = (1, 7)
            contacts= contacts,
            gravity_dir= self.gravity_vector,
        )

        # Predict using self.nn_predictor
        state_predicted = self.nn_predictor.predict(self.step_cnt)

        print(f"Step {self.step_cnt}: in: {state_robot_centric} ")
        print(f"Step {self.step_cnt}: out: {state_predicted}")
        
        if self.step_cnt < 500:
            write_state_to_csv(self.csv_filename, self.step_cnt, state_predicted)

        # Write into state_out 
        state_out.joint_q = wp.from_torch(state_predicted[0,:2].reshape(2,))
        state_out.joint_qd = wp.from_torch(state_predicted[0,2:].reshape(2,))
        
        # Edit: the newton.eval_fk has probably different convention on the sign of joint angles than NeRD, 
        # that's why I added minus here: 
        newton.eval_fk(self.model, -state_out.joint_q, -state_out.joint_qd, state_out)

        print(state_out.joint_q, state_out.joint_qd)

        # increase step counter
        self.step_cnt += 1
        