# Neural Pendulum: NeRD Module in Axion

This document describes how the neural pendulum runs inside Axion: which scripts do what, how they connect, and why the `step()` implementation in `nerd_engine.py` is structured so that a Torch module trained in a different environment (Warp + Neural Robot Dynamics with its own contact handling) works correctly inside Axion (Newton + Axion contact pipeline).

---

## Overview

The neural pendulum uses a **pretrained NeRD model** (`.pt` + `cfg.yaml`) as the dynamics: instead of solving Newton’s equations, each step is a forward pass of the network. Axion provides the scene (Newton model, contacts, state) and adapts its data into the exact format the NeRD model was trained on.

**Three main pieces:**

| File | Role |
|------|------|
| `examples/pendulum_NerdEngine.py` | Builds the pendulum scene (Newton model), runs the simulator, uses `NerdEngine` as the solver. |
| `src/axion/core/nerd_engine.py` | Newton-compatible solver that turns state + contacts into NeRD inputs, calls the predictor, and writes predicted state back into Newton. |
| `src/axion/nn_prediction/nerd_predictor.py` | NeRD predictor: normalizes inputs (frame, embedding, contact masks), runs the `.pt` model, and converts predictions back to world-frame joint state. |

Data flow in one step: **Newton state + contacts** → **NerdEngine.step()** (contact reorder + depth + frame/contact adapters) → **NeRDPredictor** (process_inputs + predict) → **state_out** for Newton.

---

## What Each File Does

### `examples/pendulum_NerdEngine.py`

- **Builds the pendulum** with Newton’s `ModelBuilder`: two links (capsules), two revolute joints, ground plane. Root joint is at `(0, 0, PENDULUM_HEIGHT)` (Z-up in Axion).
- **Chooses the engine**: Hydra config instantiates `NerdEngine` (e.g. via `nerdPendulum` config), so the simulation loop uses the neural solver instead of a classical one.
- **Control**: `control_policy` sets joint torques (e.g. `[0.0, 800.0]`); these could be passed into the predictor later (currently joint acts are zeros in the engine).
- **Rendering**: Optional world and anchor axes; otherwise standard Axion viewer and contact logging.

So this file is “the scene + run loop”; it does not know about contact reordering or NeRD input format—that is all in `nerd_engine.py`.

### `src/axion/core/nerd_engine.py`

- **NerdEngine** subclasses Newton’s `SolverBase`. It loads the NeRD `.pt` and `cfg.yaml`, constructs a `NeRDPredictor` with robot-specific sizes (dofs, joints, contact slots), and infers the root joint height from the model for `root_body_q`.
- **Each step** it:
  1. Takes Newton’s `state_in`, `contacts`, and (optionally) control.
  2. Reorders and preprocesses contacts into fixed-size, link-ordered slots and computes penetration depths.
  3. Builds NeRD-style inputs (states, root_body_q, contact dict, gravity).
  4. Calls `NeRDPredictor.process_inputs(...)` then `NeRDPredictor.predict(step)`.
  5. Writes the predicted state into `state_out` and runs `newton.eval_fk` so body poses are consistent.

The critical compatibility work lives here: making Axion/Newton’s contact and frame conventions match what the NeRD model was trained on.

### `src/axion/nn_prediction/nerd_predictor.py`

- **NeRDPredictor** is the interface to the pretrained NeRD model. It does not run Warp or NeRD’s simulator; it only runs the Torch model and the same input pipeline (frame conversion, embedding, contact masking) that was used at training time.
- **process_inputs**: Moves data to the right device, builds contact masks from depths/thickness, pushes current state/contacts/root into a history buffer, converts to the anchor frame (e.g. body frame at root), wraps angles, embeds state, and applies contact masks to contact tensors. The result is `model_inputs` (states_embedding, contact_*, gravity_dir, etc.) in the format expected by `model.evaluate(...)`.
- **predict**: Uses `model_inputs` (filled by `process_inputs`), runs `model.evaluate(self.model_inputs)`, then converts the raw prediction (e.g. relative pose / velocity deltas) to next joint state and converts back to world frame; optional angle wrapping is applied.

So the predictor assumes inputs already match the training convention (slot layout, frame, gravity, contact semantics). The engine is responsible for that.

---

## How They Work Together

1. **Startup**: `pendulum_NerdEngine.py` builds the model and creates a simulator with `NerdEngine`. The engine loads the NeRD `.pt`/`cfg` and creates `NeRDPredictor` with dof sizes, joint layout, and contact count matching the pendulum (e.g. 2 DOF position, 2 DOF velocity, 4 contact slots).
2. **Every step**:
   - Newton calls `NerdEngine.step(state_in, state_out, control, contacts, dt)`.
   - The engine converts `state_in` and Newton `contacts` into the exact tensor shapes and semantics the predictor expects (see “Step() in detail” below).
   - The engine calls `nn_predictor.process_inputs(...)` then `nn_predictor.predict(step)` and gets back the next state in world-frame joint space.
   - The engine writes that into `state_out` and runs `newton.eval_fk(...)` so the rest of Axion (e.g. rendering, contact detection) sees consistent body poses.
3. **Rendering / logging**: The simulator and viewer use `state_out` and Newton’s contact list as usual; they do not need to know that the state came from a neural network.

So: **pendulum_NerdEngine.py** = scene + loop, **nerd_engine.py** = Newton ↔ NeRD adapter, **nerd_predictor.py** = NeRD model + input/output processing.

---

## The `step()` Function in `nerd_engine.py` in Detail

The NeRD model was trained in an environment that differs from Axion in:

- **Contact representation**: NeRD uses a fixed number of contact slots per link, body-always-first (point0 = body, point1 = ground), and a fixed slot order (e.g. link0 contact0, link0 contact1, link1 contact0, link1 contact1). Newton reports contacts in arbitrary order and does not guarantee body-first or link ordering.
- **Coordinate frame**: NeRD training often uses Y-up and “root body” frame for part of the input (e.g. contact_points_1 in root frame). Axion/Newton use Z-up and world frame.
- **Penetration depth**: The network was trained with a specific depth convention and with inactive slots marked (e.g. large constant depth) so contact masks are correct.
- **Root body pose**: The network expects `root_body_q` to describe the first link’s pose in a specific way (position and quaternion convention) so that transforming contacts and gravity into the anchor frame matches training.

If any of these are wrong, the network gets out-of-distribution inputs and the trajectory degrades. Below is what each part of `step()` does and why it matters.

---

### 1. State and control (lines ~171–176)

- `state_robot_centric`: Joint positions and velocities from `state_in` are concatenated and unsqueezed to `(1, 4)` for a single environment. This matches the NeRD state vector `[joint_q, joint_qd]`.
- `joint_acts`: Currently zeros; can be wired from `control` later. The model was trained with joint actions; shape must be `(num_models, joint_act_dim)`.

**Why it matters**: The network expects one state vector per env with a fixed layout; wrong size or order would misalign with the model’s expectations.

---

### 2. Contact preprocessing — reorder kernel (lines ~178–224)

Newton’s contact list is unordered: contact pairs can be (body, ground) or (ground, body), and order of contacts is not by link. NeRD instead expects:

- **Body always in “position 0”**: point0/thickness0 = body, point1/thickness1 = ground; normal from body to ground.
- **Fixed slot order**: e.g. link0 contact0, link0 contact1, link1 contact0, link1 contact1 (slots 0–3 for two links, two contacts per link).

So the engine:

- Allocates reordered arrays (point0, point1, normal, thickness0, thickness1, body_shape, plus per-body contact count) via `_allocate_reordered_contact_arrays(max_num_contacts_per_model)`.
- Runs `reorder_ground_contacts_kernel`: for each Newton contact, it determines which shape is body (body index ≥ 0) and which is ground (body index -1), swaps if needed so body is always in slot 0, flips the normal to “body → ground”, and writes into **output slots by link**. Slot index is `(body_idx - BODY_INDEX_OFFSET) * MAX_CONTACTS_PER_BODY + contact_index_within_body`, with `body_contact_count` used so contacts of the same link go into consecutive slots. Unwritten slots get `reordered_body_shape = -1` so the depth kernel can mark them inactive.

**Why it matters**: Without this, the same physical contact could appear in different input indices each step, or with body/ground swapped. The network was trained on a fixed slot semantics (e.g. slot 0,1 = link0, slot 2,3 = link1); reordering makes Axion’s contacts match that.

---

### 3. Contact preprocessing — penetration depth kernel (lines ~226–257)

NeRD uses penetration depth and thickness to build **contact masks** (in `get_contact_masks`): a slot is “in contact” if depth is below a threshold derived from thickness; otherwise it is masked out. So depth must:

- Be defined for the **same** reordered slots (body point, ground point, normal, thicknesses).
- Use the same sign/convention as in training (e.g. negative = penetrating).
- Mark inactive slots with a large positive value (e.g. 1000) so they are not considered in contact.

The engine runs `contact_penetration_depth_kernel` on the **reordered** contact arrays and body poses. For each slot it: checks `body_shape` (skip if &lt; 0), transforms body point to world, applies thickness offsets, and computes signed depth. Inactive slots get a constant `NON_TOUCHING_DEPTH` (e.g. 1000). Result is `contact_depths_wp_array`, then converted to Torch.

**Why it matters**: Wrong depth convention or wrong slot layout would produce wrong contact masks; the network would then “see” contact where there is none or miss real contact, and predictions would diverge.

---

### 4. Converting contacts to Torch and axis/frame fixes (lines ~256–281)

- **contact_depths**: Warp array → Torch; already in reordered slot order.
- **contact_normals**: Reordered normals are flattened to `(1, 12)`. NeRD was trained with **Y-up**; Axion/Newton use **Z-up**. So the engine swaps the y and z components for all four slots: `contact_normals[:, :, [1, 2]] = contact_normals[:, :, [2, 1]]`, so that what the network sees matches the training frame.
- **contact_thickness**: From reordered body thickness, flattened to `(1, num_contacts)`.
- **contact_points_0**: Body contact points (reordered). Optional per-slot scaling (e.g. `contact_points_0[0, 6] *= 2.0`, `contact_points_0[0, 10] *= 2.0`) can be used to match training if the training pipeline applied a different scale or convention for certain slots.
- **contact_points_1**: Ground contact points (reordered), world frame; the predictor will transform them into the root body frame in `convert_coordinate_frame`.

**Why it matters**: Normals and points are in the network’s expected frame (Y-up) and slot order; otherwise gravity direction and contact positions would be inconsistent with training and the learned dynamics would be wrong.

---

### 5. Root body pose `root_body_q` (lines ~286–296)

NeRD expects `root_body_q` as the pose of the “root” link (first link) in a specific format: position (x, y, z) and quaternion (qx, qy, qz, qw). Training used **Y-up** and a fixed anchor (e.g. pendulum pivot at `(0, height, 0)` in NeRD’s frame). So the engine:

- Takes the first body’s pose from `state_in.body_q`.
- Replaces the position with `[0.0, self._root_joint_height, 0.0]`: x=0, z=0, y=height. This matches NeRD’s world frame (Y-up, pivot on Y-axis). `_root_joint_height` is inferred in `__init__` from the root joint’s world position (joint with parent -1) so it matches the scene (e.g. `PENDULUM_HEIGHT`).
- Adjusts the quaternion: e.g. `root_body_q[0, 5] = -root_body_q[0, 4]` and `root_body_q[0, 4] = 0` so the orientation matches the convention the NeRD integrator used (e.g. first-link orientation in Y-up world).

**Why it matters**: `root_body_q` is used in the predictor to transform `contact_points_1` and normals/gravity into the anchor frame. If position or orientation convention differed from training, those transformed inputs would be wrong and the network would receive out-of-distribution inputs.

---

### 6. Gravity vector (initialized in `__init__`)

`self.gravity_vector` is set to `(0, -1, 0)` (negative Y) for all envs. NeRD was trained with Y-up and gravity along -Y; the predictor uses this for `gravity_dir` and possibly frame conversions.

---

### 7. Call to the predictor (lines ~298–307)

- `process_inputs(states, joint_acts, root_body_q, contacts, gravity_dir)`: Fills `model_inputs` (contact masks, history, coordinate conversion, state embedding, masked contact tensors).
- `predict(step)`: Runs `model.evaluate(self.model_inputs)`, converts prediction to next state in world frame, wraps angles.

---

### 8. Writing back to Newton and FK (lines ~314–318)

- `state_out.joint_q` and `state_out.joint_qd` are filled from the predicted state (first two components = joint_q, last two = joint_qd).
- `newton.eval_fk(self.model, -state_out.joint_q, -state_out.joint_qd, state_out)` updates body poses in `state_out`. The sign flip (`-joint_q`, `-joint_qd`) compensates for a different angle sign convention between Newton and the NeRD training (e.g. same physical angle stored with opposite sign).

---

## Summary

- **pendulum_NerdEngine.py**: Builds the pendulum and runs the simulator with `NerdEngine`.
- **nerd_engine.py**: Implements the solver; adapts Newton state and contacts to NeRD’s expected format (reordered contacts, depths, Y-up normals/root, root_body_q convention) and writes predicted state back with the correct joint sign convention.
- **nerd_predictor.py**: Runs the NeRD model and its input/output pipeline (frames, embedding, contact masks).