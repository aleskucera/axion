# NeRD Standalone Extraction Summary

This document summarizes what was extracted from the NeRD codebase to create a standalone model integration package.

## Files Extracted

### Model Architecture Files (Complete)

1. **`models/base_models.py`**
   - `MLPBase`: Base MLP architecture
   - `LSTMBase`: LSTM base for sequence models
   - `GRUBase`: GRU base for sequence models
   - **Source**: `models/base_models.py` (complete file)

2. **`models/model_utils.py`**
   - `get_activation_func()`: Activation function factory
   - **Source**: `models/model_utils.py` (complete file)

3. **`models/model_transformer.py`**
   - `GPTConfig`: Transformer configuration dataclass
   - `GPT`: GPT-style transformer architecture
   - `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`: Transformer components
   - **Source**: `models/model_transformer.py` (complete file, truncated at line 367)

4. **`models/models.py`**
   - `MLPDeterministic`: Deterministic MLP output layer
   - `ModelMixedInput`: Main model wrapper that handles mixed inputs
   - **Source**: `models/models.py` (complete file)

### Utility Files (Complete)

5. **`utils/torch_utils.py`**
   - Quaternion operations: `quat_mul`, `quat_inv`, `quat_rotate`, `quat_rotate_inverse`
   - Coordinate transformations: `transform_point`, `transform_point_inverse`
   - State conversions: `convert_states_w2b`, `convert_states_b2w`
   - Exponential map: `quat_to_exponential_coord`, `exponential_coord_to_quat`
   - **Source**: `utils/torch_utils.py` (complete file)

6. **`utils/running_mean_std.py`**
   - `RunningMeanStd`: Running mean/std normalization
   - `RunningMeanStdDict`: Dictionary-based normalization
   - **Source**: `utils/running_mean_std.py` (complete file)

7. **`utils/commons.py`**
   - Constants: `CONTACT_DEPTH_UPPER_RATIO`, `MIN_CONTACT_EVENT_THRESHOLD`
   - **Source**: `utils/commons.py` (extracted constants only)

### State Processing (Extracted Methods)

8. **`integrators/state_processor.py`**
   - Extracted from `integrators/integrator_neural.py`:
     - `get_contact_masks()` (lines 395-410)
     - `wrap2PI()` (lines 522-531)
     - `convert_coordinate_frame()` (lines 963-1023)
     - `convert_states_back_to_world()` (lines 1025-1068)
     - `embed_states()` (lines 1084-1112)
     - `convert_prediction_to_next_states()` (lines 573-629)
     - `convert_prediction_to_next_states_regular_dofs()` (lines 646-661)
     - `convert_prediction_to_next_states_orientation_dofs()` (lines 678-716)
     - `_convert_states_w2b()` (lines 906-940)
     - `_convert_contacts_w2b()` (lines 870-900)
     - `_convert_gravity_w2b()` (lines 942-961)
   - **Note**: Adapted to work without Warp dependencies

### Standalone Wrapper

9. **`nerd_predictor.py`**
   - `NeRDPredictor`: Main interface class
   - Combines all extracted components into a simple API
   - Handles model loading, input preparation, and prediction
   - **New file**: Created specifically for standalone usage

## Key Adaptations Made

1. **Removed Warp.sim Dependencies**:
   - Can use `warp` (import warp as wp) but NOT `warp.sim`
   - Joint type constants try to use `wp.JOINT_*` if available, otherwise use numeric constants
   - Removed dependency on Warp's `Model` and `State` objects from `warp.sim`
   - Replaced `wp.to_torch()` and `wp.from_torch()` with direct tensor operations where needed

2. **Contact Information**:
   - Original code extracts contacts from Warp's `model.rigid_contact_*` arrays
   - Standalone version expects contacts to be provided as tensors in the same format
   - User must provide contact data from their own simulation/physics engine

3. **Robot Configuration**:
   - Original code extracts robot configuration from Warp model
   - Standalone version requires explicit robot configuration parameters
   - User must provide: DOF counts, joint types, joint indices, angular/continuous DOF flags

4. **Import Paths**:
   - Changed from absolute imports (`from models.base_models import ...`) to relative imports (`from .base_models import ...`)
   - Updated cross-module imports to work within standalone package

## Dependencies

### Required Python Packages
- `torch` (PyTorch)
- `numpy`
- `yaml` (for loading configuration files)

### Removed Dependencies
- `warp.sim` (NVIDIA Warp simulator module - but `warp` itself can be used)
- All RL training infrastructure
- Environment wrappers
- Dataset generation code

## Usage Pattern

The standalone implementation follows this pattern:

1. **Load Model**: `model, robot_name = torch.load(model_path)`
2. **Load Config**: `cfg = yaml.load(cfg_path)`
3. **Create Predictor**: `predictor = NeRDPredictor(model, cfg, device, robot_config)`
4. **Predict**: `next_states = predictor.predict(states, joint_acts, root_body_q, contacts, gravity_dir)`

## What Was NOT Extracted

The following components were intentionally excluded as they're not needed for standalone prediction:

- Training code (`algorithms/`, `train/`)
- Dataset generation (`generate/`)
- Environment wrappers (`envs/`)
- Evaluation scripts (`eval/`)
- Warp-specific utilities (`utils/warp_utils.py`)
- RL integration code

## Testing

To test the standalone implementation:

1. Load a pretrained model (e.g., Pendulum)
2. Create a `NeRDPredictor` instance with correct robot configuration
3. Provide test inputs (states, actions, contacts, gravity)
4. Verify that predictions are reasonable

## Notes

- The standalone code maintains the same mathematical operations and coordinate frame conversions as the original
- All state processing logic is preserved
- The model architecture is identical to the original
- The only changes are removal of Warp dependencies and simplification of the interface

