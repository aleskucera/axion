"""
Export-friendly mirror of MSEModel for ONNX -> TensorRT.

Differences vs. axion.neural_solver.models.mse_model.MSEModel:
    - Single concatenated tensor input (B, T, total_low_dim); no dict iteration.
    - No input/output RMS normalization (caller handles it externally).
    - No constructor-time input probing; dims are passed explicitly.
    - Uses the export-friendly GPT from fast_model_transformer.

Submodule names match the original MSEModel so a trained checkpoint loads
cleanly via FastMSEModel.from_mse_model(...) or load_state_dict(..., strict=False).
"""

import torch
import torch.nn as nn

from axion.neural_solver.models.base_models import MLPBase
from axion.neural_solver.models.fast_model_transformer import GPT, GPTConfig


class RegressionHead(nn.Module):
    """Mirror of MSEModel.RegressionHead with identical param names."""

    def __init__(self, input_dim, output_dim, mlp_cfg=None, device="cuda:0"):
        super().__init__()
        self.device = device
        mlp_cfg = mlp_cfg or {}
        hidden_cfg = {
            "layer_sizes": list(mlp_cfg.get("layer_sizes", [])),
            "activation": mlp_cfg.get("activation", "relu"),
            "layernorm": bool(mlp_cfg.get("layernorm", False)),
        }
        self.feature_net = MLPBase(input_dim, hidden_cfg, device=device)
        self.output_net = nn.Linear(self.feature_net.out_features, output_dim)
        self.to(device)

    def forward(self, inputs):
        features = self.feature_net(inputs)
        return self.output_net(features)

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class FastMSEModel(nn.Module):
    """
    Inference-only MSE model that consumes a single concatenated tensor.

    Forward input:  x of shape (B, T, total_low_dim)
    Forward output: tensor of shape (B, T, state_output_dim + lambda_output_dim)
    """

    def __init__(
        self,
        total_low_dim,
        state_output_dim,
        lambda_output_dim,
        encoder_cfg,
        transformer_cfg,
        head_mlp_cfg=None,
        states_only=False,
        device="cuda:0",
    ):
        super().__init__()

        self.device = device
        self.total_low_dim = int(total_low_dim)
        self.state_output_dim = int(state_output_dim)
        self.states_only = bool(states_only)
        self.lambda_output_dim = 0 if self.states_only else int(lambda_output_dim)
        self.regression_output_dim = self.state_output_dim + self.lambda_output_dim

        self.encoders = nn.ModuleDict()
        self.encoders["low_dim"] = MLPBase(
            self.total_low_dim, encoder_cfg, device=device
        )
        self.feature_dim = self.encoders["low_dim"].out_features

        gptconf = GPTConfig(
            n_layer=transformer_cfg["n_layer"],
            n_head=transformer_cfg["n_head"],
            n_embd=transformer_cfg["n_embd"],
            block_size=transformer_cfg["block_size"],
            bias=transformer_cfg["bias"],
            vocab_size=self.feature_dim,
            dropout=transformer_cfg.get("dropout", 0.0),
        )
        self.transformer_model = GPT(gptconf)
        self.transformer_model.to(self.device)
        self.feature_dim = self.transformer_model.config.n_embd

        self.regression_head = RegressionHead(
            self.feature_dim,
            self.regression_output_dim,
            mlp_cfg=head_mlp_cfg,
            device=device,
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, total_low_dim) concatenated low_dim inputs, already normalized.

        Returns:
            (B, T, state_output_dim + lambda_output_dim)
        """
        features = self.encoders["low_dim"](x)
        features = self.transformer_model(features)

        bsz, seq_len, feature_dim = features.shape
        features_flatten = features.reshape(bsz * seq_len, feature_dim)
        regression_flatten = self.regression_head(features_flatten)
        return regression_flatten.reshape(bsz, seq_len, -1)

    def evaluate_last(self, x):
        """Last-timestep slice, mirroring MSEModel.evaluate's return shape (B, 1, D)."""
        return self.forward(x)[:, -1:, :]

    @classmethod
    def from_mse_model(cls, mse_model, device=None):
        """
        Instantiate a FastMSEModel and copy weights from a trained MSEModel.

        The source model must have a transformer backbone and a single 'low_dim'
        encoder. RMS normalization buffers on the source are intentionally
        dropped (load_state_dict is called with strict=False).
        """
        if "low_dim" not in mse_model.encoders:
            raise ValueError(
                "FastMSEModel.from_mse_model requires a 'low_dim' encoder on the source model."
            )

        low_dim_encoder = mse_model.encoders["low_dim"]
        if any(isinstance(m, nn.Linear) for m in low_dim_encoder.body):
            first_linear = next(m for m in low_dim_encoder.body if isinstance(m, nn.Linear))
            total_low_dim = int(first_linear.in_features)
        else:
            # Identity encoder (layer_sizes: []) — out_features equals in_features
            total_low_dim = int(low_dim_encoder.out_features)

        layer_sizes = []
        activation = "relu"
        layernorm = False
        for module in low_dim_encoder.body:
            if isinstance(module, nn.Linear):
                layer_sizes.append(int(module.out_features))
            elif isinstance(module, nn.LayerNorm):
                layernorm = True
            elif isinstance(module, nn.Tanh):
                activation = "tanh"
            elif isinstance(module, nn.ReLU):
                activation = "relu"
            elif isinstance(module, nn.ELU):
                activation = "elu"
            elif isinstance(module, nn.SiLU):
                activation = "silu"
            elif isinstance(module, nn.Identity):
                activation = "identity"
        encoder_cfg = {
            "layer_sizes": layer_sizes,
            "activation": activation,
            "layernorm": layernorm,
        }

        src_cfg = mse_model.transformer_model.config
        transformer_cfg = {
            "n_layer": src_cfg.n_layer,
            "n_head": src_cfg.n_head,
            "n_embd": src_cfg.n_embd,
            "block_size": src_cfg.block_size,
            "bias": src_cfg.bias,
            "dropout": src_cfg.dropout,
        }

        head_mlp_cfg = None
        head = mse_model.regression_head
        if hasattr(head, "feature_net"):
            head_layer_sizes = []
            head_activation = "relu"
            head_layernorm = False
            for module in head.feature_net.body:
                if isinstance(module, nn.Linear):
                    head_layer_sizes.append(int(module.out_features))
                elif isinstance(module, nn.LayerNorm):
                    head_layernorm = True
                elif isinstance(module, nn.Tanh):
                    head_activation = "tanh"
                elif isinstance(module, nn.ReLU):
                    head_activation = "relu"
                elif isinstance(module, nn.ELU):
                    head_activation = "elu"
                elif isinstance(module, nn.SiLU):
                    head_activation = "silu"
                elif isinstance(module, nn.Identity):
                    head_activation = "identity"
            head_mlp_cfg = {
                "layer_sizes": head_layer_sizes,
                "activation": head_activation,
                "layernorm": head_layernorm,
            }

        target_device = device if device is not None else getattr(mse_model, "device", "cuda:0")

        fast = cls(
            total_low_dim=total_low_dim,
            state_output_dim=int(mse_model.state_output_dim),
            lambda_output_dim=int(mse_model.lambda_output_dim),
            encoder_cfg=encoder_cfg,
            transformer_cfg=transformer_cfg,
            head_mlp_cfg=head_mlp_cfg,
            states_only=bool(getattr(mse_model, "states_only", False)),
            device=target_device,
        )

        # strict=False because the source carries RMS normalization buffers
        # (input_rms / regression_output_rms) and possibly an `is_transformer`
        # flag that the fast model intentionally drops.
        missing, unexpected = fast.load_state_dict(mse_model.state_dict(), strict=False)
        del unexpected

        # attn.bias is a causal-mask buffer (lower-triangular constant). When the
        # source model was trained with flash attention enabled it never registers
        # this buffer, so it is absent from the checkpoint state_dict. The fast
        # model always registers it (flash is disabled for ONNX compatibility) and
        # initialises it correctly from block_size — no data from the checkpoint
        # is needed. Any other missing key is a real mismatch and should fail.
        truly_missing = [k for k in missing if not k.endswith(".attn.bias")]
        if truly_missing:
            raise RuntimeError(
                f"FastMSEModel.from_mse_model: missing keys when loading state_dict: {truly_missing}"
            )

        fast.eval()
        return fast

    def to(self, device):
        self.device = device
        for (_, encoder) in self.encoders.items():
            encoder.to(device)
        self.transformer_model.to(device)
        self.regression_head.to(device)
        return self
