# Contact Processing Pipeline: Newton → NerdEngine → NeRDPredictor → Neural Model

This document describes how contact data flows from the Newton physics package through the Axion codebase and into the pretrained NeRD neural dynamics model.

---

## Overview

```
newton.Contacts (raw collision detection output)
       │
       ▼
NerdEngine.step()                          [nerd_engine.py]
  ├─ reorder_ground_contacts_kernel        [contact_interaction.py]
  ├─ contact_penetration_depth_kernel      [contact_interaction.py]
  ├─ Warp → Torch conversion
  ├─ Axis swap (Z-up → Y-up)
  └─ Build contacts dict
       │
       ▼
NeRDPredictor.process_inputs()             [nerd_predictor.py]
  ├─ get_contact_masks()                   [state_processor.py]
  ├─ History stacking (deque)
  ├─ convert_coordinate_frame()            [state_processor.py]
  │     └─ _convert_contacts_w2b()
  └─ Apply contact masks (zero inactive)
       │
       ▼
ModelMixedInput.evaluate()                 [models.py]
  └─ low_dim inputs include:
       contact_normals, contact_points_1,
       contact_depths (masked)
```

---

## Step-by-step Pipeline

### Step 0: Newton Collision Detection

Newton's collision detection system populates a `newton.Contacts` object (defined in `third_party/newton/newton/_src/sim/contacts.py`). The relevant rigid contact buffers are:

| Buffer | Type | Description |
|--------|------|-------------|
| `rigid_contact_count` | `int32[1]` | Number of active contacts this step |
| `rigid_contact_shape0` | `int32[max]` | Shape index of first body in pair |
| `rigid_contact_shape1` | `int32[max]` | Shape index of second body in pair |
| `rigid_contact_point0` | `vec3[max]` | Contact point on shape0 (in shape0's body-local frame) |
| `rigid_contact_point1` | `vec3[max]` | Contact point on shape1 (in shape1's body-local frame) |
| `rigid_contact_normal` | `vec3[max]` | Contact normal (direction depends on which shape is 0 vs 1) |
| `rigid_contact_thickness0` | `float32[max]` | Collision margin/thickness of shape0 |
| `rigid_contact_thickness1` | `float32[max]` | Collision margin/thickness of shape1 |

Newton does **not** guarantee a consistent ordering of which body is shape0 vs shape1. The ground plane (body index = -1) can appear in either position.

### Step 1: Reordering Contacts (`reorder_ground_contacts_kernel`)

**File:** `src/axion/types/contact_interaction.py`

**Purpose:** Newton returns contacts in arbitrary order with no guarantee about which shape is the "body" vs the "ground". This kernel normalizes the data so that:

- `point0` / `thickness0` always belong to the **body** (body index >= 0)
- `point1` / `thickness1` always belong to the **ground** (body index == -1)
- The contact normal always points **from body to ground**
- If Newton had them swapped (`body_a == -1`), all fields are swapped and the normal is negated

**Slot allocation scheme:**

Contacts are scattered into fixed output slots organized by body/link index:

```
slot = link_index * MAX_CONTACTS_PER_BODY + contact_index_within_body
```

- `MAX_CONTACTS_PER_BODY = 2` (up to 2 contacts per link)
- `BODY_INDEX_OFFSET = 0` (link_index = body_idx directly)
- For a 2-link pendulum with 4 total slots:
  - Slots 0, 1 → body 0 (first link) contacts
  - Slots 2, 3 → body 1 (second link) contacts

Non-ground contacts (body-body) are skipped. An atomic counter per body (`body_contact_count`) assigns the sub-slot within each body.

**Output shapes:** All outputs are 2D arrays `(num_models, max_num_contacts_per_model)` where `max_num_contacts_per_model = 4` for the pendulum.

### Step 2: Penetration Depth Computation (`contact_penetration_depth_kernel`)

**File:** `src/axion/types/contact_interaction.py`

**Purpose:** Compute signed penetration depth for each reordered contact slot.

**Algorithm:**

1. Look up the body index from the reordered `body_shape` array
2. Transform the body-local contact point to world space using `body_q` (body transform)
3. The ground contact point is already in world space (ground has identity transform)
4. Apply thickness offsets along the normal direction:
   - Body point moves outward: `p_body_adj = p_body_world - thickness0 * n`
   - Ground point moves outward: `p_ground_adj = p_ground_world + thickness1 * n`
5. Compute raw depth: `raw = dot(n, p_ground_adj - p_body_adj)`
6. Negate: `depth = -raw`

**Sign convention:**

- **Negative depth** → bodies are penetrating (overlap)
- **Positive depth** → bodies are separated (not touching)
- **Inactive/invalid slots** → set to `NON_TOUCHING_DEPTH = 1000.0`

### Step 3: Warp → Torch Conversion and Axis Adjustments (`NerdEngine.step()`)

**File:** `src/axion/core/nerd_engine.py`

After the Warp kernels run, the results are converted to PyTorch tensors and adjusted for the NeRD model's coordinate conventions.

**Contact normals — axis swap:**

The NeRD model was trained with Y-up convention, but Axion uses Z-up. The Y and Z components are swapped for all 4 contact slots:

```python
contact_normals = contact_normals.view(1, 4, 3)
contact_normals[:, :, [1, 2]] = contact_normals[:, :, [2, 1]]
contact_normals = contact_normals.view(1, 12)
```

**Contact points — scaling adjustment:**

Some body-local contact point components are scaled by 2x. This is a calibration correction because Newton returns contact points relative to the body's center of mass, while NeRD expects them relative to the joint/link origin (whose offset differs by a factor related to link length):

```python
contact_points_0[0, 6] *= 2.0   # x-component of slot 2 (link2, contact1)
contact_points_0[0, 10] *= 2.0  # y-component of slot 3 (link2, contact2)
```

**Assembled contacts dictionary:**

```python
contacts = {
    "contact_normals":     # (1, 12) — 4 contacts × 3 components, Y/Z swapped
    "contact_depths":      # (1, 4)  — signed depth per slot
    "contact_thicknesses": # (1, 4)  — body-side thickness per slot
    "contact_points_0":    # (1, 12) — body contact points (body-local), selectively scaled
    "contact_points_1":    # (1, 12) — ground contact points (world space)
}
```

### Step 4: Contact Mask Computation (`get_contact_masks()`)

**File:** `src/axion/nn_prediction/integrators/state_processor.py`

**Purpose:** Determine which contact slots represent active (near/touching) contacts vs inactive ones.

**Algorithm:**

```python
contact_event_threshold = CONTACT_DEPTH_UPPER_RATIO * contact_thickness
contact_event_threshold = max(contact_event_threshold, MIN_CONTACT_EVENT_THRESHOLD)
contact_masks = (contact_depths < contact_event_threshold)
```

**Constants** (from `utils/commons.py`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `CONTACT_DEPTH_UPPER_RATIO` | 4.0 | Multiplied by thickness to get activation threshold |
| `MIN_CONTACT_EVENT_THRESHOLD` | 0.12 | Minimum threshold (ensures contacts activate even with tiny thickness) |
| `CONTACT_FREE_DEPTH` | 10000.0 | Depth value indicating no contact |

A contact is **active** (mask = True) when `depth < 4 × thickness` (or `depth < 0.12`, whichever is larger). Since inactive slots have `depth = 1000.0`, they will always be masked out.

### Step 5: History Stacking and Input Assembly (`NeRDPredictor.process_inputs()`)

**File:** `src/axion/nn_prediction/nerd_predictor.py`

The predictor maintains a sliding window of past states using a `deque(maxlen=num_states_history)`. For the pretrained pendulum model, `num_states_history = 10`.

Each history entry stores:

```python
{
    "root_body_q", "states", "joint_acts", "gravity_dir",
    "contact_normals", "contact_depths", "contact_thicknesses",
    "contact_points_0", "contact_points_1", "contact_masks"
}
```

These are stacked along a time dimension:

```python
# Result shape: (num_envs, T, dim) where T grows up to num_states_history
model_inputs[key] = torch.stack([entry[key] for entry in history], dim=1)
```

### Step 6: Coordinate Frame Conversion (`convert_coordinate_frame()`)

**File:** `src/axion/nn_prediction/integrators/state_processor.py`

With `states_frame = "body"` and `anchor_frame_step = "every"`, contacts are converted from world frame to body frame at each timestep:

**Contact points** (`contact_points_1`, the ground points) and **contact normals** are transformed via `_convert_contacts_w2b()`:

1. Extract the root body's position and quaternion from `root_body_q`
2. Apply inverse transform to contact_points_1 (ground points):
   ```
   contact_points_1_body = transform_point_inverse(body_pos, body_quat, contact_points_1)
   ```
3. Rotate contact normals into body frame:
   ```
   contact_normals_body = quat_rotate_inverse(body_quat, contact_normals)
   ```

Note: `contact_points_0` (body-local points) are **not** frame-converted — they are already in body-local space from Newton.

### Step 7: Apply Contact Masks

**File:** `src/axion/nn_prediction/nerd_predictor.py`

All contact fields (except `contact_masks` itself) are zeroed out where the mask is inactive:

```python
for key in model_inputs.keys():
    if key.startswith('contact_') and key != 'contact_masks':
        masked = torch.where(mask < 1e-5, 0.0, data)
```

This is done per-contact-slot, with proper reshaping:
- `contact_depths`, `contact_thicknesses`: reshaped to `(B, T, num_contacts, 1)`
- `contact_normals`, `contact_points_0`, `contact_points_1`: reshaped to `(B, T, num_contacts, 3)`

### Step 8: Neural Model Consumption (`ModelMixedInput.evaluate()`)

**File:** `src/axion/nn_prediction/models/models.py`

The model's input configuration (from `cfg.yaml`) specifies which fields form the `low_dim` input vector:

```yaml
inputs:
  low_dim:
    - states_embedding
    - contact_normals
    - contact_points_1
    - contact_depths
    - joint_acts
    - gravity_dir
```

These are concatenated along the last dimension and passed through:

1. **Input normalization** (`normalize_input: true`) — running mean/std normalization per input field
2. **Low-dim encoder** (identity in this config, `layer_sizes: []`)
3. **Transformer** (6 layers, 192 embedding dim, 12 heads) over the time window
4. **MLP head** (64 hidden units) → prediction output

The contact fields that directly reach the neural model are:
- `contact_normals` — (B, T, 12) body-frame normals, masked
- `contact_points_1` — (B, T, 12) body-frame ground points, masked
- `contact_depths` — (B, T, 4) signed depths, masked

Note: `contact_points_0`, `contact_thicknesses`, and `contact_masks` are used during processing but are **not** part of the model's input features.

---

## Additional Notes

### Contact Data in the Classic Axion Solver

The file `src/axion/types/contact_interaction.py` also contains `contact_interaction_kernel` and the `ContactInteraction` struct. These are used by the classical (non-neural) Axion constraint solver, not the NeRD engine. The `ContactInteraction` struct includes additional physical properties like friction coefficient, restitution, and spatial constraint bases (Jacobians) that are not needed for the neural approach.

### Pendulum-Specific Configuration

The current implementation is hardcoded for the 2-link pendulum:

- `num_contacts_per_env = 4` (2 links × 2 contacts per link)
- `dof_q_per_env = 2` (2 revolute joints)
- `dof_qd_per_env = 2` (2 angular velocities)
- `joint_types = [2, 2]` (both REVOLUTE)

### Frame Convention Mismatch

The NeRD model was trained with **Y-up** gravity (negative Y), while Axion/Newton uses **Z-up**. This requires:

1. Y/Z axis swap on contact normals (Step 3)
2. Manual construction of `root_body_q` with the height placed in the Y component
3. Negation of `joint_q` / `joint_qd` signs when calling `newton.eval_fk()` after prediction (different angle sign convention)

### Contact Points Reference Frame

Newton returns `rigid_contact_point0` and `rigid_contact_point1` in **body-local** space (relative to each body's transform). For ground contacts, the ground body has identity transform, so `contact_point1` (ground side) is effectively in world space. The body-side point (`contact_point0`) must be transformed to world space using `body_q` when computing penetration depth.
