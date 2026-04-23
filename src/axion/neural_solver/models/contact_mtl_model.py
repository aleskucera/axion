# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from axion.neural_solver.models.base_models import MLPBase
from axion.neural_solver.models.model_transformer import GPT, GPTConfig


class ClassificationHead(nn.Module):
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
        # Backward compatibility for checkpoints created before configurable
        # head depth/width, where ClassificationHead only had a linear layer.
        if not hasattr(self, "feature_net"):
            return self.output_net(inputs)
        features = self.feature_net(inputs)
        return self.output_net(features)

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class ContactLambdaPredictionHead(nn.Module):
    """Regression head that predicts the contact-related channels of the lambda vector."""

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
        # Backward compatibility for checkpoints created before configurable
        # head depth/width, where ContactLambdaPredictionHead only had linear output.
        if not hasattr(self, "feature_net"):
            return self.output_net(inputs)
        features = self.feature_net(inputs)
        return self.output_net(features)

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class ContactMTLModel(nn.Module):
    def __init__(
        self,
        input_sample,
        contact_lambda_dim,
        input_cfg,
        network_cfg,
        device="cuda:0",
    ):
        super().__init__()

        self.device = device

        self.input_rms = None
        self.normalize_input = network_cfg.get("normalize_input", False)
        self.normalize_output = network_cfg.get("normalize_output", False)
        self.use_asinh_transform = bool(network_cfg.get("use_asinh_transform", True))

        self.encoders, self.feature_dim = self.construct_input_encoders(
            input_cfg, network_cfg["encoder"], input_sample, device=device
        )

        if "transformer" in network_cfg:
            model_args = dict(
                n_layer=network_cfg["transformer"]["n_layer"],
                n_head=network_cfg["transformer"]["n_head"],
                n_embd=network_cfg["transformer"]["n_embd"],
                block_size=network_cfg["transformer"]["block_size"],
                bias=network_cfg["transformer"]["bias"],
                vocab_size=self.feature_dim,
                dropout=network_cfg["transformer"]["dropout"],
            )
            gptconf = GPTConfig(**model_args)
            self.transformer_model = GPT(gptconf)
            self.transformer_model.to(self.device)
            self.is_transformer = True
            self.feature_dim = self.transformer_model.config.n_embd
        else:
            raise NotImplementedError

        self.contact_lambda_dim = int(contact_lambda_dim)
        self.contact_lambda_transform_rms = None

        head_mlp_cfg = (
            network_cfg.get("model", {}).get("mlp", {})
            if isinstance(network_cfg.get("model", {}), dict)
            else {}
        )
        self.cls_head = ClassificationHead(
            self.feature_dim,
            self.contact_lambda_dim,
            mlp_cfg=head_mlp_cfg,
            device=device,
        )
        self.contact_lambda_head = ContactLambdaPredictionHead(
            self.feature_dim,
            self.contact_lambda_dim,
            mlp_cfg=head_mlp_cfg,
            device=device,
        )

    def construct_input_encoders(
        self, input_cfg, encoder_cfg, input_sample, device="cuda:0"
    ):
        encoders = nn.ModuleDict()

        if len(input_cfg.get("low_dim", [])) > 0:
            low_dim_size = 0
            self.low_dim_input_names = input_cfg.get("low_dim")
            for low_dim_input_name in self.low_dim_input_names:
                assert len(input_sample[low_dim_input_name].shape) in [2, 3]
                low_dim_size += input_sample[low_dim_input_name].shape[-1]

            assert "low_dim" in encoder_cfg
            low_dim_encoder = MLPBase(
                low_dim_size, encoder_cfg["low_dim"], device=device
            )
            encoders["low_dim"] = low_dim_encoder

        feature_dim = 0
        for input_name in encoders:
            feature_dim += encoders[input_name].out_features

        return encoders, feature_dim

    def __setstate__(self, state):
        # Migrate checkpoint saved before attribute rename (log_contact_lambda_rms →
        # contact_lambda_transform_rms, introduced in af3f7e7) and backfill any
        # other fields added to __init__ after the checkpoint was saved.
        if "log_contact_lambda_rms" in state and "contact_lambda_transform_rms" not in state:
            state["contact_lambda_transform_rms"] = state.pop("log_contact_lambda_rms")
        state.setdefault("contact_lambda_transform_rms", None)
        state.setdefault("input_rms", None)
        state.setdefault("normalize_input", False)
        state.setdefault("normalize_output", False)
        state.setdefault("use_asinh_transform", True)
        super().__setstate__(state)

    def set_input_rms(self, data_rms):
        self.input_rms = {}
        for input_name in self.encoders:
            if input_name == "low_dim":
                for low_dim_input_name in self.low_dim_input_names:
                    if low_dim_input_name in data_rms:
                        self.input_rms[low_dim_input_name] = data_rms[low_dim_input_name]
            else:
                self.input_rms[input_name] = data_rms[input_name]

    def set_output_rms(self, output_rms=None, lambda_output_rms=None, contact_lambda_transform_rms=None):
        self.contact_lambda_transform_rms = contact_lambda_transform_rms

    def extract_input_features(self, input_dict):
        features = []
        for input_name in self.encoders:
            if input_name == "low_dim":
                low_dim_input_list = []
                for low_dim_input_name in self.low_dim_input_names:
                    low_dim_input_list.append(input_dict[low_dim_input_name])
                cur_input = torch.cat(low_dim_input_list, dim=-1)
            else:
                cur_input = input_dict[input_name]
            features.append(self.encoders[input_name](cur_input))
        return torch.cat(features, dim=-1)

    def _run_heads(self, features):
        """Run both heads on transformer features and return the output dict."""
        bsz, seq_len, feature_dim = features.shape
        features_flat = features.contiguous().view(-1, feature_dim)

        logits = self.cls_head(features_flat).view(bsz, seq_len, -1)
        contact_lambda_hat = self.contact_lambda_head(features_flat).view(bsz, seq_len, -1)

        return {
            "logits": logits,
            "contact_lambda_hat": contact_lambda_hat,
        }

    def evaluate(self, input_dict, deterministic=False):
        del deterministic
        if self.normalize_input:
            for obs_key in self.input_rms.keys():
                input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[obs_key])

        features = self.extract_input_features(input_dict)
        if self.is_transformer:
            features = self.transformer_model(features)

        out = self._run_heads(features)
        out = {k: v[:, -1:, ...] for k, v in out.items()}

        # Invert the configured target transform applied during training to recover actual lambda values.
        p = out["contact_lambda_hat"]
        if self.normalize_output and self.contact_lambda_transform_rms is not None:
            p = self.contact_lambda_transform_rms.normalize(p, un_norm=True)
        if self.use_asinh_transform:
            p = torch.sinh(p)
        out["contact_lambda_hat"] = p
        return out

    def forward(self, input_dict, deterministic=False, inject_noise=False):
        del deterministic
        if self.normalize_input:
            for obs_key in self.input_rms.keys():
                input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[obs_key])

        if inject_noise:
            for obs_key in input_dict.keys():
                input_dict[obs_key] = input_dict[obs_key] + torch.randn_like(
                    input_dict[obs_key]
                ) * 0.01

        features = self.extract_input_features(input_dict)
        if self.is_transformer:
            features = self.transformer_model(features)

        return self._run_heads(features)

    def to(self, device):
        self.device = device
        for (_, encoder) in self.encoders.items():
            encoder.to(device)
        if self.transformer_model is not None:
            self.transformer_model.to(device)

        self.cls_head.to(device)
        self.contact_lambda_head.to(device)

        if self.input_rms is not None:
            for key in self.input_rms:
                self.input_rms[key] = self.input_rms[key].to(device)
        if self.contact_lambda_transform_rms is not None:
            self.contact_lambda_transform_rms = self.contact_lambda_transform_rms.to(device)
        return self
