# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class VelAndLambdaPredictionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        device='cuda:0'
    ):
        super().__init__()
        self.device = device

        self.output_net = nn.Linear(
            input_dim,
            output_dim
        )

        self.to(device)

    def forward(self, inputs):
        return self.output_net(inputs)

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class VelAndLambdaModel(nn.Module):
    def __init__(
        self,
        input_sample,
        vel_ouput_dim,
        lambda_output_dim,
        input_cfg,
        network_cfg,
        device='cuda:0'
    ):
        super().__init__()

        self.device = device
        self.has_state_head = True
        self.has_lambda_head = True

        self.input_rms = None
        self.normalize_input = network_cfg.get('normalize_input', False)
        self.output_rms = None
        self.lambda_output_rms = None
        self.normalize_output = network_cfg.get('normalize_output', False)

        self.encoders, self.feature_dim = self.construct_input_encoders(
            input_cfg,
            network_cfg['encoder'],
            input_sample,
            device=device
        )

        if "transformer" in network_cfg:
            model_args = dict(
                n_layer=network_cfg['transformer']['n_layer'],
                n_head=network_cfg['transformer']['n_head'],
                n_embd=network_cfg['transformer']['n_embd'],
                block_size=network_cfg['transformer']['block_size'],
                bias=network_cfg['transformer']['bias'],
                vocab_size=self.feature_dim,
                dropout=network_cfg['transformer']['dropout'],
            )
            gptconf = GPTConfig(**model_args)

            self.transformer_model = GPT(gptconf)
            self.transformer_model.to(self.device)

            self.is_transformer = True
            self.feature_dim = self.transformer_model.config.n_embd

        else:
            raise NotImplementedError

        self.vel_output_dim = int(vel_ouput_dim)
        self.lambda_output_dim = int(lambda_output_dim)
        self.total_output_dim = self.vel_output_dim + self.lambda_output_dim

        self.model = VelAndLambdaPredictionHead(
            self.feature_dim,
            self.total_output_dim,
            device=device
        )
        # Keep this alias for compatibility with older trainer logic/checkpoints.
        self.lambda_model = self.model

        self.output_tanh = network_cfg.get('output_tanh', False)
        self.lambda_output_tanh = network_cfg.get('lambda_output_tanh', False)

    def construct_input_encoders(
        self,
        input_cfg,
        encoder_cfg,
        input_sample,
        device='cuda:0'
    ):
        encoders = nn.ModuleDict()

        if len(input_cfg.get('low_dim', [])) > 0:
            low_dim_size = 0
            self.low_dim_input_names = input_cfg.get('low_dim')
            for low_dim_input_name in self.low_dim_input_names:
                assert len(input_sample[low_dim_input_name].shape) in [2, 3]
                low_dim_size += input_sample[low_dim_input_name].shape[-1]

            assert 'low_dim' in encoder_cfg
            low_dim_encoder = MLPBase(
                low_dim_size,
                encoder_cfg['low_dim'],
                device=device
            )
            encoders['low_dim'] = low_dim_encoder

        feature_dim = 0
        for input_name in encoders:
            feature_dim += encoders[input_name].out_features

        return encoders, feature_dim

    def set_input_rms(self, data_rms):
        self.input_rms = {}
        for input_name in self.encoders:
            if input_name == 'low_dim':
                for low_dim_input_name in self.low_dim_input_names:
                    if low_dim_input_name in data_rms:
                        self.input_rms[low_dim_input_name] = data_rms[low_dim_input_name]
            else:
                self.input_rms[input_name] = data_rms[input_name]

    def set_output_rms(self, output_rms=None, lambda_output_rms=None):
        self.output_rms = output_rms
        self.lambda_output_rms = lambda_output_rms

    def extract_input_features(self, input_dict):
        features = []
        for input_name in self.encoders:
            if input_name == 'low_dim':
                low_dim_input_list = []
                for low_dim_input_name in self.low_dim_input_names:
                    low_dim_input_list.append(input_dict[low_dim_input_name])
                cur_input = torch.cat(low_dim_input_list, dim=-1)
            else:
                cur_input = input_dict[input_name]
            features.append(self.encoders[input_name](cur_input))
        features = torch.cat(features, dim=-1)
        return features

    def _split_outputs(self, output):
        state_output = output[..., :self.vel_output_dim]
        lambda_output = output[..., self.vel_output_dim:self.total_output_dim]

        if self.output_tanh:
            state_output = torch.tanh(state_output)
        if self.lambda_output_tanh:
            lambda_output = torch.tanh(lambda_output)

        if self.normalize_output:
            if self.output_rms is not None:
                state_output = self.output_rms.normalize(state_output, un_norm=True)
            if self.lambda_output_rms is not None:
                lambda_output = self.lambda_output_rms.normalize(lambda_output, un_norm=True)

        return state_output, lambda_output

    def evaluate(self, input_dict, deterministic=False):
        del deterministic
        if self.normalize_input:
            for obs_key in self.input_rms.keys():
                input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[obs_key])

        features = self.extract_input_features(input_dict)

        if self.is_transformer:
            features = self.transformer_model(features)
        output = self.model(features)
        state_output, lambda_output = self._split_outputs(output)

        return {
            'state': state_output[:, -1:, :],
            'lambda': lambda_output[:, -1:, :],
        }

    def forward(self, input_dict, deterministic=False, inject_noise=False):
        del deterministic
        if self.normalize_input:
            for obs_key in self.input_rms.keys():
                input_dict[obs_key] = self.input_rms[obs_key].normalize(input_dict[obs_key])

        if inject_noise:
            for obs_key in input_dict.keys():
                input_dict[obs_key] = (
                    input_dict[obs_key] +
                    torch.randn_like(input_dict[obs_key]) * 0.01
                )

        features = self.extract_input_features(input_dict)

        if self.is_transformer:
            features = self.transformer_model(features)

        B, T, feature_dim = features.shape
        features_flatten = features.contiguous().view(-1, feature_dim)
        output_flatten = self.model(features_flatten)
        output = output_flatten.view(B, T, -1)
        state_output, lambda_output = self._split_outputs(output)
        return {'state': state_output, 'lambda': lambda_output}

    def to(self, device):
        self.device = device
        for (_, encoder) in self.encoders.items():
            encoder.to(device)
        if self.transformer_model is not None:
            self.transformer_model.to(device)

        self.model.to(device)
        if self.input_rms is not None:
            for k in self.input_rms:
                self.input_rms[k] = self.input_rms[k].to(device)
        if self.output_rms is not None:
            self.output_rms = self.output_rms.to(device)
        if self.lambda_output_rms is not None:
            self.lambda_output_rms = self.lambda_output_rms.to(device)