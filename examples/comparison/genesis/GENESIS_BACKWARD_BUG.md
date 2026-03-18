# Genesis Differentiable Simulation — Backward Bug Report

Filed against: https://github.com/Genesis-Embodied-AI/Genesis

---

## Bug Description

`loss.backward()` hangs indefinitely for any articulated robot where a free-floating root body (`freejoint`) has child joints (revolute, prismatic, etc.). The backward pass never returns — it blocks inside the ABD backward kernel.

**Single free-floating bodies (freejoint only, no children) work correctly.** Single fixed-base joints (hinge/slide with no parent freejoint) also work. The hang is triggered exclusively by kinematic trees of depth ≥ 2 with a free root.

---

## Steps to Reproduce

### Working case (single free body — for reference)

```python
import os, tempfile
import genesis as gs
import torch

MJCF_FREE_BODY = """
<mujoco model="free_body">
  <worldbody>
    <body name="chassis" pos="0 0 0">
      <freejoint name="root"/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

gs.init(backend=gs.gpu, logging_level="warning")
fd, path = tempfile.mkstemp(suffix=".xml")
with os.fdopen(fd, "w") as f:
    f.write(MJCF_FREE_BODY)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0), requires_grad=True),
    rigid_options=gs.options.RigidOptions(enable_collision=False),
    show_viewer=False,
)
robot = scene.add_entity(gs.morphs.MJCF(file=path))
scene.build()

# NOTE: must use gs.tensor (not torch.tensor) for gradient to flow
ctrl = gs.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
target = torch.tensor([0.05, 0.0, 0.0], device=gs.device)

scene.reset()
for _ in range(5):
    robot.set_dofs_velocity(ctrl)
    scene.step()

# NOTE: must use robot.get_state().pos (not get_pos() or get_links_pos())
# get_state() registers the state in _queried_states so backward can seed gradients
state = robot.get_state()
loss = torch.nn.functional.mse_loss(state.pos.squeeze(), target)
loss.backward()  # completes in ~11s (JIT), ctrl.grad is non-zero ✓
print(f"ctrl.grad = {ctrl.grad}")
```

### Hanging case (freejoint + one child hinge — minimal repro)

```python
import os, tempfile
import genesis as gs
import torch

MJCF_ARTICULATED = """
<mujoco model="free_plus_hinge">
  <worldbody>
    <body name="chassis" pos="0 0 0">
      <freejoint name="root"/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
      <body name="wheel" pos="0.2 0 0">
        <joint name="hinge_y" type="hinge" axis="0 1 0"/>
        <inertial mass="0.5" pos="0 0 0" diaginertia="0.05 0.05 0.05"/>
        <geom type="cylinder" fromto="0 -0.05 0 0 0.05 0" size="0.1"
              contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

gs.init(backend=gs.gpu, logging_level="warning")
fd, path = tempfile.mkstemp(suffix=".xml")
with os.fdopen(fd, "w") as f:
    f.write(MJCF_ARTICULATED)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0), requires_grad=True),
    rigid_options=gs.options.RigidOptions(enable_collision=False),
    show_viewer=False,
)
robot = scene.add_entity(gs.morphs.MJCF(file=path))
scene.build()

ctrl = gs.tensor([0.0] * 7, requires_grad=True)  # 6 free DOFs + 1 hinge
target = torch.tensor([0.05, 0.0, 0.0], device=gs.device)

scene.reset()
for _ in range(5):
    robot.set_dofs_velocity(ctrl)
    scene.step()

state = robot.get_state()
loss = torch.nn.functional.mse_loss(state.pos.squeeze(), target)
loss.backward()  # <-- hangs indefinitely
print("Never reached")
```

Same hang occurs with `slide` (prismatic) child joints and with 3+ child joints.
Replacing the hinge with a second `freejoint` (separate free body, no parent-child
relationship) does not hang.

---

## Expected Behavior

`loss.backward()` completes and `ctrl.grad` is populated with the gradient of the
loss w.r.t. the control velocities, as it does for the single free-body case.

---

## Environment

| | |
|---|---|
| OS | Arch Linux (kernel 6.18.9) |
| GPU | NVIDIA RTX A500 Laptop GPU |
| GPU driver | 590.48.01 |
| CUDA | 12.8 |
| PyTorch | 2.9.1+cu128 |
| Python | 3.12.12 |

---

## Release versions tested

Tested on **v0.3.8**, **v0.3.9**, and **v0.4.1** — all hang for the articulated case.

---

## Additional Context

### What works vs. what hangs

| Configuration | Backward |
|---|---|
| Single freejoint (free-floating body, no children) | ✅ completes (~11s JIT) |
| Single fixed-base hinge (no parent freejoint) | ✅ completes (~4s JIT) |
| Single fixed-base slide / prismatic | ✅ completes |
| freejoint root + one hinge child | ❌ hangs |
| freejoint root + one slide child | ❌ hangs |
| freejoint root + three hinge children (e.g. wheeled robot) | ❌ hangs |

The hang is unaffected by: `enable_collision`, `disable_constraint`, `gravity`,
number of simulation steps, or joint type of the child.

### Correct gradient API (not clearly documented)

The gradient API requires:

1. **`gs.tensor` for the control variable** — `torch.tensor(..., requires_grad=True)`
   does not register with the Genesis AD bridge; gradients will be `None`.

2. **`robot.get_state().pos` as output** — `robot.get_pos()` and
   `robot.get_links_pos()` return detached tensors (`grad_fn=None`).
   `robot.get_state()` registers the state in `_queried_states` so that
   `scene._backward()` (called internally by `loss.backward()` via the
   `gs.Tensor.backward()` override) can seed gradients via `add_grad_from_state()`.

3. **`loss.backward()` triggers `scene._backward()` automatically** — the
   `gs.Tensor.backward()` override calls `super().backward()` then
   `self.scene._backward()`. Calling `scene._backward()` manually after
   `set_pos_grad()` is the old undocumented approach and does not work correctly.

### Additional bug in v0.3.8

`set_dofs_velocity_grad` in `rigid_entity.py` calls `self._get_idx(...)` but the
method is named `self._get_global_idx(...)` — a one-character typo that raises
`AttributeError: 'RigidEntity' object has no attribute '_get_idx'` when the
backward path reaches `process_input_grad()`.

**Fix:** in `genesis/engine/entities/rigid_entity/rigid_entity.py`, line ~2333:
```python
# wrong (v0.3.8):
dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)

# correct:
dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
```
This bug is fixed in v0.3.9+.
