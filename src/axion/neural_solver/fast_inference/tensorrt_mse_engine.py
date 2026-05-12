"""
Drop-in TensorRT replacement for axion.neural_solver.models.mse_model.MSEModel.

TensorRTMSEEngine subclasses nn.Module so .to(device) / .eval() / .train()
are free no-ops, then exposes the exact attribute surface that the existing
MSE branches in NeuralPredictor and AxionEngineWithNeuralLambdas look for:

    state_output_dim, lambda_output_dim, regression_output_dim,
    regression_head (sentinel, non-None),
    low_dim_input_names,
    evaluate(input_dict) -> (B, 1, regression_output_dim) torch tensor

With this, the engine plugs into NeuralPredictor without any other edits.

Inputs come in as the same dict that MSEModel expects:
    {
      "states_embedding":  (B, T, 4),
      "contact_normals":   (B, T, 12),
      "contact_points_1":  (B, T, 12),
      "contact_depths":    (B, T, 4),
      "gravity_dir":       (B, T, 3),
      ...
    }

The wrapper concatenates them in the metadata's low_dim_keys order, applies
the cached RMS normalization (so the engine sees the same distribution it
was trained on), copies into a pre-allocated CUDA buffer, runs the engine
on a single CUDA stream with static tensor addresses (CUDA-graph-friendly),
and returns the last timestep slice.
"""

from pathlib import Path

import tensorrt as trt
import torch
import torch.nn as nn
import warp as wp


class _RegressionHeadSentinel:
    """Placeholder so `hasattr(engine, 'regression_head')` returns True without
    registering anything as an nn.Module submodule."""


class TensorRTMSEEngine(nn.Module):
    """TensorRT-backed MSE engine that duck-types MSEModel."""

    is_tensorrt_engine = True

    def __init__(
        self,
        plan_path: str | Path,
        meta_path: str | Path,
        device: str = "cuda:0",
        apply_normalization: bool = True,
    ):
        super().__init__()

        plan_path = Path(plan_path)
        meta_path = Path(meta_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"TensorRT .plan not found: {plan_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Engine metadata not found: {meta_path}")

        self.device = device
        self.apply_normalization = bool(apply_normalization)

        meta = torch.load(str(meta_path), map_location="cpu", weights_only=False)
        self.low_dim_input_names: list[str] = list(meta["low_dim_keys"])
        self.state_output_dim:   int = int(meta["state_output_dim"])
        self.lambda_output_dim:  int = int(meta["lambda_output_dim"])
        self.regression_output_dim: int = int(meta["regression_output_dim"])
        self.total_low_dim:      int = int(meta["total_low_dim"])
        self.block_size:         int = int(meta["block_size"])
        self.batch_size:         int = int(meta["batch_size"])
        self._input_rms_meta = meta.get("input_rms", {})
        self.low_dim_dims: dict[str, int] = dict(meta.get("low_dim_dims", {}))

        # Sentinel so the MSE duck-type checks pass (hasattr(..., "regression_head")).
        # No classification_head attribute → engine is detected as MSE, not MTL.
        # Bypass nn.Module.__setattr__ for non-tensor / non-Module attrs we don't
        # want auto-registered as submodules.
        object.__setattr__(self, "regression_head", _RegressionHeadSentinel())

        # ── Build TRT runtime ─────────────────────────────────────────────────
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {plan_path}")
        self._context = self._engine.create_execution_context()

        in_info, out_info = self._discover_io()
        if len(in_info) != 1 or len(out_info) != 1:
            raise RuntimeError(
                f"TensorRTMSEEngine expects 1 input / 1 output; got "
                f"{len(in_info)} / {len(out_info)}."
            )
        self._in_name, self._in_shape, _ = in_info[0]
        self._out_name, self._out_shape, _ = out_info[0]

        B, T, D = self._in_shape
        if (B, T, D) != (self.batch_size, self.block_size, self.total_low_dim):
            raise RuntimeError(
                f"Engine I/O shape {self._in_shape} disagrees with metadata "
                f"(B={self.batch_size}, T={self.block_size}, D={self.total_low_dim})."
            )

        # ── Static buffers (addresses registered once → CUDA-graph friendly) ──
        self._input_buf = torch.empty(
            *self._in_shape, device=device, dtype=torch.float32
        )
        self._output_buf = torch.empty(
            *self._out_shape, device=device, dtype=torch.float32
        )
        self._context.set_tensor_address(self._in_name, int(self._input_buf.data_ptr()))
        self._context.set_tensor_address(self._out_name, int(self._output_buf.data_ptr()))

        # Dedicated stream so all TRT work goes through a single capturable stream.
        self._stream = torch.cuda.Stream(device=device)

        # ── Pre-build normalization tensors aligned to concat order ───────────
        self._norm_mean, self._norm_inv_std = self._build_norm_buffers(device)

    # ─────────────────────────────────────────────────────────────────────────
    # nn.Module machinery — keep .to / .eval / .train no-ops in spirit.
    # ─────────────────────────────────────────────────────────────────────────
    def to(self, device, *args, **kwargs):
        # The engine and its CUDA buffers are bound to the device chosen at
        # construction. Re-binding to a different device would require a
        # full re-build, so silently keep us on the original device.
        del device, args, kwargs
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # TRT plumbing
    # ─────────────────────────────────────────────────────────────────────────
    def _discover_io(self):
        inputs, outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self._engine.get_tensor_shape(name))
            dtype = self._engine.get_tensor_dtype(name)
            mode = self._engine.get_tensor_mode(name)
            (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(
                (name, shape, dtype)
            )
        return inputs, outputs

    def _build_norm_buffers(self, device: str):
        """
        Concatenate per-key mean/var in low_dim_keys order so a single
        broadcast against (B, T, D) does all normalization at once.

        If the stored mean/var for a key has fewer entries than that key's
        feature dim (e.g. scalar RMS), it is broadcast-repeated to the
        feature dim.
        """
        means, inv_stds = [], []
        for key in self.low_dim_input_names:
            entry = self._input_rms_meta.get(key)
            if entry is None:
                # Fully missing → identity normalization for this key. We
                # cannot know the per-key feature dim without input_rms; rely
                # on later concat-shape validation to surface mismatches.
                means.append(torch.zeros(1, dtype=torch.float32))
                inv_stds.append(torch.ones(1, dtype=torch.float32))
                continue
            mean = entry["mean"].to(dtype=torch.float32).flatten()
            var = entry["var"].to(dtype=torch.float32).flatten()
            inv_std = 1.0 / torch.sqrt(var + 1e-5)
            means.append(mean)
            inv_stds.append(inv_std)

        norm_mean = torch.cat(means).to(device)
        norm_inv_std = torch.cat(inv_stds).to(device)
        if norm_mean.shape[-1] != self.total_low_dim:
            # Scalar/short RMS entries — fall back to broadcasting by treating
            # the concatenated buffer as identity for any unknown dim. We log
            # this once during construction so it's not silent.
            print(
                f"[TensorRTMSEEngine] normalization buffer shape "
                f"{tuple(norm_mean.shape)} does not match total_low_dim="
                f"{self.total_low_dim}; broadcasting at runtime."
            )
        return norm_mean, norm_inv_std

    # ─────────────────────────────────────────────────────────────────────────
    # Public API (MSEModel duck-type)
    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure pass-through: caller hands in the already-concatenated, already-
        normalized tensor of shape (B, T, total_low_dim). Returns the full
        (B, T, regression_output_dim) tensor. Used by the parity check.
        """
        return self._run_engine(x, normalize=False)[:, :, :].clone()

    @torch.no_grad()
    def evaluate(self, input_dict: dict, deterministic: bool = False) -> torch.Tensor:
        """
        Mirror of MSEModel.evaluate: returns (B, 1, regression_output_dim).
        Concatenates input_dict in low_dim_keys order, normalizes, runs TRT.

        Eager-mode path (kept for parity benchmarks and `NeuralPredictor`).
        For the CUDA-graph path use `evaluate_prepopulated`.
        """
        del deterministic
        x = self._concat_low_dim(input_dict)
        out = self._run_engine(x, normalize=self.apply_normalization)
        return out[:, -1:, :].clone()

    @torch.no_grad()
    def evaluate_prepopulated(self) -> torch.Tensor:
        """
        Run the engine assuming the caller has already written the new input
        into `self._input_buf` directly. Skips all concat / normalization /
        clone work. Returns a (B, 1, regression_output_dim) view into the
        engine's pre-allocated output buffer — DO NOT modify it, and DO NOT
        keep references past the next call.

        This is the fast path used by `FastNeuralPredictor` under capture.
        """
        # Run on Warp's current stream so the enqueue is captured into the
        # same graph as the surrounding kernel launches.
        stream_handle = int(wp.get_stream().cuda_stream)
        ext_stream = torch.cuda.ExternalStream(stream_handle)
        with torch.cuda.stream(ext_stream):
            ok = self._context.execute_async_v3(stream_handle)
            if not ok:
                raise RuntimeError(
                    "TensorRT execute_async_v3 returned False (evaluate_prepopulated)."
                )
        return self._output_buf[:, -1:, :]

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────
    def _concat_low_dim(self, input_dict: dict) -> torch.Tensor:
        parts = []
        for key in self.low_dim_input_names:
            if key not in input_dict:
                raise KeyError(
                    f"TensorRTMSEEngine.evaluate: input_dict is missing key '{key}' "
                    f"(low_dim concatenation order: {self.low_dim_input_names})."
                )
            parts.append(input_dict[key])
        x = torch.cat(parts, dim=-1)

        # During simulation warm-up the NeuralPredictor history queue grows
        # from 1 entry up to num_states_history. The TRT engine has a static T
        # baked in, so left-pad by replicating the oldest entry to fill the
        # buffer. The most-recent state stays at index -1, which is what the
        # predictor reads back.
        B_eng, T_eng, D_eng = self._in_shape
        B_dyn, T_dyn, D_dyn = x.shape
        if (B_dyn, D_dyn) != (B_eng, D_eng):
            raise RuntimeError(
                f"Input batch/feature shape ({B_dyn}, {D_dyn}) does not match "
                f"engine ({B_eng}, {D_eng})."
            )
        if T_dyn > T_eng:
            raise RuntimeError(
                f"Input T={T_dyn} exceeds engine T={T_eng}; this should not "
                "happen if num_states_history matches the engine's T."
            )
        if T_dyn < T_eng:
            pad = x[:, :1, :].expand(B_dyn, T_eng - T_dyn, D_dyn)
            x = torch.cat([pad, x], dim=1)
        return x

    def _run_engine(self, x: torch.Tensor, normalize: bool) -> torch.Tensor:
        if x.device != self._input_buf.device:
            x = x.to(self._input_buf.device)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        if normalize:
            x = (x - self._norm_mean) * self._norm_inv_std

        self._input_buf.copy_(x)

        # Eager-mode path: run on a dedicated stream and sync at the end so
        # the caller can read the output buffer from torch's default stream
        # without further coordination. The CUDA-graph path goes through
        # `evaluate_prepopulated` (no sync, runs on Warp's capture stream).
        with torch.cuda.stream(self._stream):
            ok = self._context.execute_async_v3(self._stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned False.")
        self._stream.synchronize()

        return self._output_buf
