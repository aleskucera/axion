"""
Standalone Fast Neural Predictor (CUDA-graph friendly).

This module is a capture-safe rewrite of `NeuralPredictor` that drives the
TensorRT MSE engine through preallocated buffers only:

- No `torch.cat` / `torch.stack` / `.clone()` per step.
- No `deque` of dict entries — the engine's input tensor IS the history.
- New row enters at slot T-1 via a fixed-shape "shift-left" pattern using
  `T-1` disjoint `copy_` ops (all kernel launches; safe under capture).
- All torch ops are pinned to Warp's current stream during capture.

It targets the Pendulum example (1 world, ≤ MAX_CONTACTS contacts,
revolute joints only, "relative" state prediction, "identical" state
embedding) but the layout is general enough that other small robots can
be supported with minor changes.

Use this in place of `NeuralPredictor` when `USE_TENSORRT_ENGINE = True`
inside `gpt_engine.py` / `hybrid_gpt_engine.py`. The construction-time
allocations (warp scratch buffers, AxionContacts, torch raw rows) happen
eagerly; from then on every step is shape-static and kernel-only.
"""

from typing import Optional

import numpy as np
import torch
import warp as wp
import newton

try:
    from src.axion.types import (
        reorder_ground_contacts_kernel,
        contact_penetration_depth_kernel,
    )
    from src.axion.core.contacts import AxionContacts
    from src.axion.neural_solver.fast_inference.tensorrt_mse_engine import (
        TensorRTMSEEngine,
    )
except ModuleNotFoundError:
    from axion.types import (
        reorder_ground_contacts_kernel,
        contact_penetration_depth_kernel,
    )
    from axion.core.contacts import AxionContacts
    from axion.neural_solver.fast_inference.tensorrt_mse_engine import (
        TensorRTMSEEngine,
    )


PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL = 4
DT_FROM_TRAINING = 0.01
# Mirrors neural_predictor_helpers.CONTACT_DEPTH_UPPER_RATIO /
# MIN_CONTACT_EVENT_THRESHOLD. Kept local because we don't import the
# torch-based mask helper.
CONTACT_DEPTH_UPPER_RATIO = 4.0
MIN_CONTACT_EVENT_THRESHOLD = 0.12

JOINT_REVOLUTE = newton.JointType.REVOLUTE


# ─────────────────────────────────────────────────────────────────────────────
# Warp kernels (all capture-safe; one launch each)
# ─────────────────────────────────────────────────────────────────────────────


@wp.kernel
def _write_states_embedding_kernel(
    joint_q: wp.array(dtype=wp.float32, ndim=1),
    joint_qd: wp.array(dtype=wp.float32, ndim=1),
    is_continuous: wp.array(dtype=wp.bool, ndim=1),
    dof_q: wp.int32,
    states_embedding: wp.array(dtype=wp.float32, ndim=2),  # (1, state_dim) — out
):
    """Pack joint_q (positions, optionally wrapped to (-pi, pi]) and joint_qd
    (velocities) into a single (1, state_dim) row. Pendulum-friendly:
    `is_continuous` controls which entries are angles that need wrapping."""
    i = wp.tid()
    pi = wp.float32(3.14159265358979)
    two_pi = wp.float32(2.0) * pi

    if i < dof_q:
        val = joint_q[i]
    else:
        val = joint_qd[i - dof_q]

    if is_continuous[i]:
        wrap_delta = wp.floor((val + pi) / two_pi) * two_pi
        val = val - wrap_delta

    states_embedding[0, i] = val


@wp.kernel
def _world_to_body_and_mask_kernel(
    body_q_2d: wp.array(dtype=wp.transform, ndim=2),  # (num_worlds, bodies)
    reordered_point1: wp.array(dtype=wp.vec3, ndim=2),  # ground point (world)
    reordered_normal: wp.array(dtype=wp.vec3, ndim=2),  # contact normal (world)
    contact_depths_in: wp.array(dtype=wp.float32, ndim=2),
    contact_thicknesses_body: wp.array(dtype=wp.float32, ndim=2),
    pivot_offset: wp.vec3,
    contact_depth_upper_ratio: wp.float32,
    min_contact_event_threshold: wp.float32,
    # Outputs
    contact_points_1_body_masked: wp.array(dtype=wp.vec3, ndim=2),
    contact_normals_body_masked: wp.array(dtype=wp.vec3, ndim=2),
    contact_depths_masked: wp.array(dtype=wp.float32, ndim=2),
):
    """Fused world→body conversion + contact-active masking.

    For each (world, contact) slot:
      - compute the active mask from `depth < max(MIN, ratio * thickness)`,
      - if inactive, write zeros and return,
      - otherwise rotate the ground point/normal into the root body's frame
        (and subtract the body→pivot offset for the point).
    """
    world_idx, contact_idx = wp.tid()

    depth = contact_depths_in[world_idx, contact_idx]
    thickness = contact_thicknesses_body[world_idx, contact_idx]
    threshold = contact_depth_upper_ratio * thickness
    if threshold < min_contact_event_threshold:
        threshold = min_contact_event_threshold

    is_active = depth < threshold

    if not is_active:
        contact_points_1_body_masked[world_idx, contact_idx] = wp.vec3(0.0, 0.0, 0.0)
        contact_normals_body_masked[world_idx, contact_idx] = wp.vec3(0.0, 0.0, 0.0)
        contact_depths_masked[world_idx, contact_idx] = 0.0
        return

    root_X = body_q_2d[world_idx, 0]
    root_X_inv = wp.transform_inverse(root_X)

    p_world = reordered_point1[world_idx, contact_idx]
    p_body = wp.transform_point(root_X_inv, p_world) - pivot_offset
    contact_points_1_body_masked[world_idx, contact_idx] = p_body

    root_q = wp.transform_get_rotation(root_X)
    n_world = reordered_normal[world_idx, contact_idx]
    n_body = wp.quat_rotate_inv(root_q, n_world)
    contact_normals_body_masked[world_idx, contact_idx] = n_body

    contact_depths_masked[world_idx, contact_idx] = depth


@wp.kernel
def _gravity_world_to_body_kernel(
    body_q_2d: wp.array(dtype=wp.transform, ndim=2),
    gravity_world: wp.array(dtype=wp.vec3, ndim=1),  # (num_worlds,)
    gravity_body: wp.array(dtype=wp.vec3, ndim=1),  # (num_worlds,)  — out
):
    world_idx = wp.tid()
    root_X = body_q_2d[world_idx, 0]
    root_q = wp.transform_get_rotation(root_X)
    gravity_body[world_idx] = wp.quat_rotate_inv(root_q, gravity_world[world_idx])


@wp.kernel
def _convert_relative_prediction_kernel(
    states_current: wp.array(dtype=wp.float32, ndim=2),  # (1, state_dim)
    prediction: wp.array(dtype=wp.float32, ndim=2),  # (1, state_dim)
    is_continuous: wp.array(dtype=wp.bool, ndim=1),
    next_states: wp.array(dtype=wp.float32, ndim=2),  # (1, state_dim) — out
):
    """next = current + prediction, with angle wrapping on continuous DOFs.
    Assumes pendulum-style robots: regular revolute joints only, so the
    relative-prediction math collapses to a single element-wise add + wrap."""
    i = wp.tid()
    pi = wp.float32(3.14159265358979)
    two_pi = wp.float32(2.0) * pi

    val = states_current[0, i] + prediction[0, i]
    if is_continuous[i]:
        wrap_delta = wp.floor((val + pi) / two_pi) * two_pi
        val = val - wrap_delta
    next_states[0, i] = val


@wp.kernel
def _clip_small_lambdas_kernel(
    lambdas: wp.array(dtype=wp.float32, ndim=2),
    threshold: wp.float32,
    zero_from: wp.int32,
):
    """In-place: zero entries where |λ| < threshold, and zero entries with
    index >= zero_from (used by HybridGPTEngine to silence trailing
    constraints that are not relevant for the pendulum)."""
    b, i = wp.tid()
    if i >= zero_from:
        lambdas[b, i] = 0.0
        return
    if wp.abs(lambdas[b, i]) < threshold:
        lambdas[b, i] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────────────────────────────────────


class FastNeuralPredictor:
    """Capture-safe neural predictor over a `TensorRTMSEEngine`.

    Drop-in replacement for `NeuralPredictor` when the underlying nn_model is
    a `TensorRTMSEEngine`. Exposes the same public surface (`reset()`,
    `create_axion_contacts()`, `process_inputs()`, `predict()`) plus a new
    `prewarm()` method that the engine calls once eagerly before
    `wp.ScopedCapture()` to seed the engine-input buffer.

    See module docstring for the rewrite rules.
    """

    def __init__(
        self,
        newton_model: newton.Model,
        nn_model: TensorRTMSEEngine,
        nn_cfg: dict,
        device: str = "cuda:0",
        clip_small_lambdas: bool = False,
        small_lambda_threshold: float = 0.01,
        lambda_zero_from: int = -1,
        joint_q_end=(1, 2),
        is_angular_dof=(True, True, True, True),
        is_continuous_dof=(True, True, False, False),
    ):
        if not isinstance(nn_model, TensorRTMSEEngine):
            raise TypeError(
                "FastNeuralPredictor only supports TensorRTMSEEngine. "
                f"Got {type(nn_model).__name__}."
            )

        self.device = device
        self.robot_model = newton_model
        self.nn_model = nn_model

        # Robot configuration (matches NeuralPredictor behaviour)
        self.num_worlds = 1
        self.dof_q_per_env = int(newton_model.joint_coord_count) // self.num_worlds
        self.dof_qd_per_env = int(newton_model.joint_dof_count) // self.num_worlds
        self.state_dim = self.dof_q_per_env + self.dof_qd_per_env
        self.num_joints_per_env = int(newton_model.joint_count) // self.num_worlds
        self.bodies_per_world = int(newton_model.body_count) // self.num_worlds

        joint_type_np = newton_model.joint_type.numpy()
        joint_q_start_global = newton_model.joint_q_start.numpy()
        self.joint_types = joint_type_np[: self.num_joints_per_env].copy()
        self.joint_q_start = (
            joint_q_start_global[: self.num_joints_per_env] % self.dof_q_per_env
        ).tolist()
        self.joint_q_end = list(joint_q_end)
        self.is_angular_dof = np.array(is_angular_dof)
        self.is_continuous_dof = np.array(is_continuous_dof)

        # Gravity vector in world frame, packed as (num_worlds,) wp.vec3.
        gravity_axis = int(newton_model.up_axis)
        gravity_np = np.zeros((self.num_worlds, 3), dtype=np.float32)
        gravity_np[:, gravity_axis] = -1.0
        self._gravity_world_wp = wp.from_numpy(
            gravity_np.reshape(-1, 3),
            dtype=wp.vec3,
            device=str(self.device),
        )
        self._gravity_body_wp = wp.zeros(
            self.num_worlds, dtype=wp.vec3, device=str(self.device)
        )
        # Torch-facing gravity vector (only used by external diagnostics).
        self.gravity_vector = torch.zeros((self.num_worlds, 3), device=str(self.device))
        self.gravity_vector[:, gravity_axis] = -1.0

        # Root joint pivot offset in the first link's CoM frame.
        joint_X_c = newton_model.joint_X_c.numpy()
        pivot_np = joint_X_c[0, :3].astype(np.float32)
        self._com_to_pivot_offset = torch.as_tensor(
            pivot_np, dtype=torch.float32, device=self.device
        )
        self._pivot_offset_wp = wp.vec3(
            float(pivot_np[0]), float(pivot_np[1]), float(pivot_np[2])
        )

        # NN-config-derived dims.
        env_cfg = nn_cfg.get("env", {})
        self.neural_integrator_cfg = env_cfg.get("utils_provider_cfg", env_cfg.get("neural_integrator_cfg", {}))
        self.num_states_history = int(self.neural_integrator_cfg.get("num_states_history", 1)       )
        self.states_frame = self.neural_integrator_cfg.get("states_frame", "body")
        self.anchor_frame_step = self.neural_integrator_cfg.get("anchor_frame_step", "every")
        self.state_prediction_type = self.neural_integrator_cfg.get("state_prediction_type", self.neural_integrator_cfg.get("prediction_type", "relative"),)
        self.lambda_prediction_type = self.neural_integrator_cfg.get("lambda_prediction_type", "absolute")
        self.prediction_type = self.state_prediction_type
        self.prediction_quantity_type = self.neural_integrator_cfg.get("prediction_quantity_type", "full_state")
        self.orientation_prediction_parameterization = self.neural_integrator_cfg.get("orientation_prediction_parameterization", "quaternion")
        self.states_embedding_type = self.neural_integrator_cfg.get("states_embedding_type", None)
        if self.states_embedding_type not in (None, "identical"):
            raise NotImplementedError(
                f"FastNeuralPredictor only supports states_embedding_type "
                f"in (None, 'identical'); got {self.states_embedding_type!r}."
            )
        # Pendulum-style joints only.
        if self.state_prediction_type != "relative":
            raise NotImplementedError(
                f"FastNeuralPredictor only supports state_prediction_type='relative'; "
                f"got {self.state_prediction_type!r}."
            )
        if self.prediction_quantity_type != "full_state":
            raise NotImplementedError(
                f"FastNeuralPredictor only supports prediction_quantity_type="
                f"'full_state'; got {self.prediction_quantity_type!r}."
            )
        for jt in self.joint_types:
            if int(jt) != int(JOINT_REVOLUTE):
                raise NotImplementedError(
                    "FastNeuralPredictor currently only supports revolute joints "
                    "(pendulum-style robots)."
                )
        self.state_embedding_dim = self.state_dim

        # Engine dims (authoritative).
        self.T = int(nn_model.block_size)
        self.D_total = int(nn_model.total_low_dim)
        self.B = int(nn_model.batch_size)
        if self.T != self.num_states_history:
            raise RuntimeError(
                f"TRT engine T ({self.T}) must equal num_states_history "
                f"({self.num_states_history}). Re-export the ONNX with "
                f"T_OVERRIDE={self.num_states_history} and rebuild the .plan."
            )
        if self.B != self.num_worlds:
            raise RuntimeError(
                f"TRT engine batch_size ({self.B}) must equal num_worlds "
                f"({self.num_worlds})."
            )

        # Lambdas config.
        self.lambda_dim = int(nn_model.lambda_output_dim)
        self.state_output_dim = int(nn_model.state_output_dim)
        self.has_lambda_prediction_module = self.lambda_dim > 0
        self.clip_small_lambdas = bool(clip_small_lambdas)
        self.small_lambda_threshold = float(small_lambda_threshold)
        self.lambda_zero_from = (
            int(lambda_zero_from) if lambda_zero_from >= 0 else self.lambda_dim
        )
        self.lambdas = (
            torch.zeros(
                (self.num_worlds, self.lambda_dim),
                device=device,
                dtype=torch.float32,
            )
            if self.lambda_dim > 0
            else None
        )

        # Per-key column ranges in the engine input row.
        low_dim_keys = list(nn_model.low_dim_input_names)
        low_dim_dims_meta = getattr(nn_model, "low_dim_dims", None) or {}
        self._low_dim_keys = low_dim_keys
        self._key_ranges: dict[str, tuple[int, int]] = {}
        offset = 0
        for key in low_dim_keys:
            if key in low_dim_dims_meta and int(low_dim_dims_meta[key]) > 0:
                d_key = int(low_dim_dims_meta[key])
            else:
                d_key = self._infer_low_dim_per_key(key)
            self._key_ranges[key] = (offset, offset + d_key)
            offset += d_key
        if offset != self.D_total:
            raise RuntimeError(
                f"Sum of low_dim per-key dims ({offset}) does not match engine "
                f"total_low_dim ({self.D_total}). Re-run build_tensorrt_engine.py "
                "to refresh engine_meta.pt with proper per-key dims."
            )

        # === Engine input buffer alias + per-key views into the LAST slot ===
        self._engine_input_buf = nn_model._input_buf  # (B, T, D_total)
        # Per-key sliced views on slot T-1 (the new-row target). These are NOT
        # contiguous (the parent has stride T*D_total), but copy_() into a
        # strided destination is supported and runs as a single kernel.
        self._slot_views: dict[str, torch.Tensor] = {
            key: self._engine_input_buf[:, -1, s:e]
            for key, (s, e) in self._key_ranges.items()
        }

        # === Per-key contiguous "raw" buffers (one row each) ===
        self._raw: dict[str, torch.Tensor] = {
            key: torch.empty(
                self.num_worlds,
                e - s,
                device=device,
                dtype=torch.float32,
            )
            for key, (s, e) in self._key_ranges.items()
        }

        # WP view of the states_embedding raw row (kernel writes here directly).
        self._raw_states_emb_wp = wp.from_torch(
            self._raw["states_embedding"], dtype=wp.float32
        )

        # === Per-key 1-D normalization slices (contiguous on the device) ===
        # nn_model._norm_mean / _norm_inv_std are (D_total,) tensors.
        self._norm_mean_per_key: dict[str, torch.Tensor] = {
            key: nn_model._norm_mean[s:e].contiguous()
            for key, (s, e) in self._key_ranges.items()
        }
        self._norm_inv_std_per_key: dict[str, torch.Tensor] = {
            key: nn_model._norm_inv_std[s:e].contiguous()
            for key, (s, e) in self._key_ranges.items()
        }

        # === Contact processing scratch (Warp arrays, allocated once) ===
        shape_contacts = (
            self.num_worlds,
            PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL,
        )
        self._reordered_point0 = wp.zeros(
            shape_contacts, dtype=wp.vec3, device=str(self.device)
        )
        self._reordered_point1 = wp.zeros(
            shape_contacts, dtype=wp.vec3, device=str(self.device)
        )
        self._reordered_normal = wp.zeros(
            shape_contacts, dtype=wp.vec3, device=str(self.device)
        )
        self._reordered_thickness0 = wp.zeros(
            shape_contacts, dtype=wp.float32, device=str(self.device)
        )
        self._reordered_thickness1 = wp.zeros(
            shape_contacts, dtype=wp.float32, device=str(self.device)
        )
        self._reordered_body_shape = wp.full(
            shape_contacts, -1, dtype=wp.int32, device=str(self.device)
        )
        self._body_contact_count = wp.zeros(
            (self.num_worlds, self.bodies_per_world),
            dtype=wp.int32,
            device=str(self.device),
        )
        self._contact_depths_wp = wp.zeros(
            shape_contacts, dtype=wp.float32, device=str(self.device)
        )
        # World→body output buffers (consumed by the per-key raw copies).
        self._contact_points_1_body_wp = wp.zeros(
            shape_contacts, dtype=wp.vec3, device=str(self.device)
        )
        self._contact_normals_body_wp = wp.zeros(
            shape_contacts, dtype=wp.vec3, device=str(self.device)
        )
        self._contact_depths_masked_wp = wp.zeros(
            shape_contacts, dtype=wp.float32, device=str(self.device)
        )

        # Reusable AxionContacts (so engine.step doesn't allocate one per call).
        self._axion_contacts = AxionContacts(
            model=newton_model,
            max_contacts_per_world=PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL,
        )
        self._num_shapes_per_world = newton_model.shape_count // self.num_worlds
        self._shape_body_2d = newton_model.shape_body.reshape(
            (self.num_worlds, self._num_shapes_per_world)
        )

        # is_continuous as a warp bool array (for the states_embedding /
        # prediction-conversion kernels).
        self._is_continuous_wp = wp.from_numpy(
            np.asarray(is_continuous_dof, dtype=np.bool_),
            dtype=wp.bool,
            device=str(self.device),
        )

        # === Prediction conversion scratch ===
        # next_state buffer (torch) + warp view for the convert kernel.
        self._next_state_buf = torch.empty(
            self.num_worlds,
            self.state_dim,
            device=device,
            dtype=torch.float32,
        )
        self._next_state_wp = wp.from_torch(self._next_state_buf, dtype=wp.float32)
        # Current state for the convert kernel is the (already-wrapped) raw
        # states_embedding row.
        self._states_emb_wp_for_predict = wp.from_torch(
            self._raw["states_embedding"], dtype=wp.float32
        )

        # Engine output views (static; the TRT wrapper registers the
        # _output_buf address once at construction). We slice the last
        # timestep state / lambda portions for prediction decoding.
        #
        # Note on contiguity: _output_buf has shape (B, T, R). The view
        # _output_buf[:, -1, :state_output_dim] is non-contiguous by torch's
        # definition (stride[0] = T*R != shape[1] = state_output_dim) even
        # though the data is contiguous in memory. wp.from_torch rejects
        # non-contiguous tensors. Workaround: squeeze the batch dim first to
        # get a 1-D contiguous view (stride (1,)), then unsqueeze back. The
        # resulting tensor has stride (state_output_dim, 1) which IS
        # contiguous and aliases the live engine output.
        self._output_buf = nn_model._output_buf
        self._state_prediction = (
            self._output_buf[0, -1, : self.state_output_dim].unsqueeze(0)
        )
        self._state_prediction_wp = wp.from_torch(
            self._state_prediction, dtype=wp.float32
        )
        if self.lambda_dim > 0:
            self._lambda_prediction = (
                self._output_buf[0, -1, self.state_output_dim:].unsqueeze(0)
            )
        else:
            self._lambda_prediction = None

        # Pre-wrap self.lambdas as a warp array for the clip kernel.
        if self.lambdas is not None:
            self._lambdas_wp = wp.from_torch(self.lambdas, dtype=wp.float32)
        else:
            self._lambdas_wp = None

        # Cached body_q reshape view (the underlying state_in.body_q pointer is
        # expected to be stable across calls; we re-create the view if the
        # ID changes).
        self._cached_body_q_2d = None
        self._cached_body_q_id: int = 0

        # Stream caching.
        self._cached_torch_stream: Optional[torch.cuda.Stream] = None
        self._cached_warp_stream_handle: int = 0

        # Public dict view of the engine input buffer (in nn_model_inputs
        # shape). Each value is a (B, T, key_dim) view of the engine buffer.
        self.nn_model_inputs: dict[str, torch.Tensor] = {
            key: self._engine_input_buf[:, :, s:e]
            for key, (s, e) in self._key_ranges.items()
        }
        # Synthetic "states" key for compatibility (alias of states_embedding
        # under identical-embedding cfg).
        if "states_embedding" in self.nn_model_inputs:
            self.nn_model_inputs["states"] = self.nn_model_inputs["states_embedding"]

        self._prewarmed = False

    # ─────────────────────────────────────────────────────────────
    # Construction helpers
    # ─────────────────────────────────────────────────────────────

    def _infer_low_dim_per_key(self, key: str) -> int:
        """Per-key feature dim fallback when engine_meta.low_dim_dims is
        empty/wrong. Pendulum defaults."""
        mapping = {
            "states_embedding": self.state_dim,
            "contact_normals": PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL * 3,
            "contact_points_1": PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL * 3,
            "contact_depths": PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL,
            "gravity_dir": 3,
        }
        if key not in mapping:
            raise KeyError(
                f"Unknown low_dim key '{key}'; cannot infer per-key dim. "
                "Add it to engine_meta.low_dim_dims via build_tensorrt_engine.py."
            )
        return mapping[key]

    # ─────────────────────────────────────────────────────────────
    # Stream helper
    # ─────────────────────────────────────────────────────────────

    def _torch_stream_from_warp(self) -> torch.cuda.Stream:
        """Return a torch ExternalStream wrapping Warp's current CUDA stream.
        During `wp.ScopedCapture()` this is the capture stream, so torch ops
        wrapped in `with torch.cuda.stream(...)` land in the captured graph."""
        wp_stream = wp.get_stream()
        cur_handle = int(wp_stream.cuda_stream)
        if (
            self._cached_torch_stream is None
            or self._cached_warp_stream_handle != cur_handle
        ):
            self._cached_torch_stream = torch.cuda.ExternalStream(cur_handle)
            self._cached_warp_stream_handle = cur_handle
        return self._cached_torch_stream

    def _get_body_q_2d(self, state_in) -> wp.array:
        """Return a cached (num_worlds, bodies_per_world) reshape of
        state_in.body_q. Reuses the view if state_in.body_q's pointer is
        unchanged across calls (the common case)."""
        body_q = state_in.body_q
        bq_id = id(body_q)
        if self._cached_body_q_2d is None or self._cached_body_q_id != bq_id:
            self._cached_body_q_2d = body_q.reshape(
                (self.num_worlds, self.bodies_per_world)
            )
            self._cached_body_q_id = bq_id
        return self._cached_body_q_2d

    # ─────────────────────────────────────────────────────────────
    # Public API (matches NeuralPredictor)
    # ─────────────────────────────────────────────────────────────

    def reset(self):
        """Reset the engine input buffer (call at start of a new trajectory)."""
        self._engine_input_buf.zero_()
        if self.lambdas is not None:
            self.lambdas.zero_()
        self._prewarmed = False

    def create_axion_contacts(self, newton_contacts) -> AxionContacts:
        """Reuse the preallocated AxionContacts; just refresh contact data.
        Mirrors the kernel-launch pattern in HybridGPTEngine.load_data so this
        call is capturable."""
        self._axion_contacts.load_contact_data(newton_contacts, self.robot_model)
        return self._axion_contacts

    def prewarm(self, state_in, axion_contacts, dt: float):
        """Seed the engine's input ring buffer with the current state in
        every slot. Equivalent to the original "history of length 1, pad with
        first entry" behaviour on step 0. Must be called once eagerly before
        `wp.ScopedCapture()`."""
        self._compute_new_row(state_in, axion_contacts, dt)
        self._normalize_raw_keys()
        with torch.cuda.stream(self._torch_stream_from_warp()):
            for t in range(self.T):
                for key, (s, e) in self._key_ranges.items():
                    self._engine_input_buf[:, t, s:e].copy_(self._raw[key])
        self._prewarmed = True

    def process_inputs(self, state_in, axion_contacts, dt: float) -> dict:
        """Compute the new row, shift-left the engine input ring buffer, and
        write the new row at slot T-1. Shape-static; uses only kernel
        launches and (T-1) + len(keys) disjoint `copy_` ops."""
        self._compute_new_row(state_in, axion_contacts, dt)
        self._normalize_raw_keys()
        with torch.cuda.stream(self._torch_stream_from_warp()):
            for t in range(self.T - 1):
                self._engine_input_buf[:, t, :].copy_(
                    self._engine_input_buf[:, t + 1, :]
                )
            for key in self._low_dim_keys:
                self._slot_views[key].copy_(self._raw[key])
        return self.nn_model_inputs

    @torch.no_grad()
    def predict(self, dt: float):
        """Run the TRT engine on the populated input buffer and decode the
        prediction into `_next_state_buf` (and `lambdas` if applicable)."""
        torch_stream = self._torch_stream_from_warp()
        with torch.cuda.stream(torch_stream):
            self.nn_model.evaluate_prepopulated()

            wp.launch(
                kernel=_convert_relative_prediction_kernel,
                dim=(self.state_dim,),
                inputs=[
                    self._states_emb_wp_for_predict,
                    self._state_prediction_wp,
                    self._is_continuous_wp,
                ],
                outputs=[self._next_state_wp],
                device=str(self.device),
            )

            next_lambdas = None
            if self.lambda_dim > 0:
                self.lambdas.copy_(self._lambda_prediction)
                if self.clip_small_lambdas:
                    wp.launch(
                        kernel=_clip_small_lambdas_kernel,
                        dim=(self.num_worlds, self.lambda_dim),
                        inputs=[
                            self._lambdas_wp,
                            self.small_lambda_threshold,
                            self.lambda_zero_from,
                        ],
                        device=str(self.device),
                    )
                next_lambdas = self.lambdas

        return self._next_state_buf, next_lambdas

    @torch.no_grad()
    def predict_lambdas_only(self, dt: float) -> torch.Tensor:
        """Run the engine and return next_lambdas only (no state decoding)."""
        if self.lambda_dim == 0:
            raise RuntimeError(
                "FastNeuralPredictor.predict_lambdas_only: model has no lambda head."
            )
        torch_stream = self._torch_stream_from_warp()
        with torch.cuda.stream(torch_stream):
            self.nn_model.evaluate_prepopulated()
            self.lambdas.copy_(self._lambda_prediction)
            if self.clip_small_lambdas:
                wp.launch(
                    kernel=_clip_small_lambdas_kernel,
                    dim=(self.num_worlds, self.lambda_dim),
                    inputs=[
                        self._lambdas_wp,
                        self.small_lambda_threshold,
                        self.lambda_zero_from,
                    ],
                    device=str(self.device),
                )
        return self.lambdas

    # ─────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────

    def _compute_new_row(self, state_in, axion_contacts, dt: float):
        """Populate every entry in `self._raw[...]` with the new (unnormalized)
        input row. All work is kernel launches on the active Warp stream."""
        torch_stream = self._torch_stream_from_warp()
        with torch.cuda.stream(torch_stream):
            # 1. Sync joint coords from body coords (Axion stores max-coord).
            newton.eval_ik(
                self.robot_model,
                state_in,
                state_in.joint_q,
                state_in.joint_qd,
            )

            # 2. States embedding (writes into _raw["states_embedding"]).
            wp.launch(
                kernel=_write_states_embedding_kernel,
                dim=(self.state_dim,),
                inputs=[
                    state_in.joint_q,
                    state_in.joint_qd,
                    self._is_continuous_wp,
                    self.dof_q_per_env,
                ],
                outputs=[self._raw_states_emb_wp],
                device=str(self.device),
            )

            # 3. Reorder contacts (body always on side 0, ground on side 1).
            self._body_contact_count.zero_()
            wp.launch(
                kernel=reorder_ground_contacts_kernel,
                dim=(self.num_worlds, axion_contacts.max_contacts),
                inputs=[
                    axion_contacts.contact_count,
                    axion_contacts.contact_shape0,
                    axion_contacts.contact_shape1,
                    axion_contacts.contact_point0,
                    axion_contacts.contact_point1,
                    axion_contacts.contact_normal,
                    axion_contacts.contact_thickness0,
                    axion_contacts.contact_thickness1,
                    self._shape_body_2d,
                    self.bodies_per_world,
                    self._body_contact_count,
                ],
                outputs=[
                    self._reordered_point0,
                    self._reordered_point1,
                    self._reordered_normal,
                    self._reordered_thickness0,
                    self._reordered_thickness1,
                    self._reordered_body_shape,
                ],
                device=str(self.device),
            )

            # 4. Contact penetration depth (writes into _contact_depths_wp).
            body_q_2d = self._get_body_q_2d(state_in)
            wp.launch(
                kernel=contact_penetration_depth_kernel,
                dim=(self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
                inputs=[
                    body_q_2d,
                    self._shape_body_2d,
                    self.bodies_per_world,
                    self._reordered_point0,
                    self._reordered_point1,
                    self._reordered_normal,
                    self._reordered_thickness0,
                    self._reordered_thickness1,
                    self._reordered_body_shape,
                ],
                outputs=[self._contact_depths_wp],
                device=str(self.device),
            )

            # 5. World→body for contact points/normals + active-mask in one kernel.
            wp.launch(
                kernel=_world_to_body_and_mask_kernel,
                dim=(self.num_worlds, PENDULUM_MAX_NUM_CONTACTS_PER_ROBOT_MODEL),
                inputs=[
                    body_q_2d,
                    self._reordered_point1,
                    self._reordered_normal,
                    self._contact_depths_wp,
                    self._reordered_thickness0,
                    self._pivot_offset_wp,
                    CONTACT_DEPTH_UPPER_RATIO,
                    MIN_CONTACT_EVENT_THRESHOLD,
                ],
                outputs=[
                    self._contact_points_1_body_wp,
                    self._contact_normals_body_wp,
                    self._contact_depths_masked_wp,
                ],
                device=str(self.device),
            )

            # 6. Gravity world→body.
            wp.launch(
                kernel=_gravity_world_to_body_kernel,
                dim=(self.num_worlds,),
                inputs=[body_q_2d, self._gravity_world_wp],
                outputs=[self._gravity_body_wp],
                device=str(self.device),
            )

            # 7. Copy Warp outputs into the per-key torch raw buffers (single
            #    GPU memcpy per key — no host work).
            cn_view = wp.to_torch(self._contact_normals_body_wp).reshape(
                self.num_worlds, -1
            )
            self._raw["contact_normals"].copy_(cn_view)

            cp_view = wp.to_torch(self._contact_points_1_body_wp).reshape(
                self.num_worlds, -1
            )
            self._raw["contact_points_1"].copy_(cp_view)

            self._raw["contact_depths"].copy_(
                wp.to_torch(self._contact_depths_masked_wp)
            )
            self._raw["gravity_dir"].copy_(wp.to_torch(self._gravity_body_wp))

    def _normalize_raw_keys(self):
        """Apply per-key (x - mean) * inv_std in place on each raw buffer."""
        with torch.cuda.stream(self._torch_stream_from_warp()):
            for key in self._low_dim_keys:
                self._raw[key].sub_(self._norm_mean_per_key[key]).mul_(
                    self._norm_inv_std_per_key[key]
                )
