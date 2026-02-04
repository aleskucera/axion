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
from src.axion.neural_solver.models import model_utils
import numpy as np

class MLPBase(nn.Module):
    def __init__(self, in_features, network_cfg, device='cuda:0'):
        super(MLPBase, self).__init__()
        
        self.device = device

        layer_sizes = network_cfg['layer_sizes']
        modules = []
        for i in range(len(layer_sizes)):
            modules.append(nn.Linear(in_features, layer_sizes[i]))
            modules.append(model_utils.get_activation_func(network_cfg['activation']))
            if network_cfg.get('layernorm', False):
                modules.append(torch.nn.LayerNorm(layer_sizes[i]))
            in_features = layer_sizes[i]

        self.body = nn.Sequential(*modules).to(device)
        self.out_features = in_features

    def forward(self, inputs):
        shape = inputs.shape
        inputs_flatten = inputs.view((-1, shape[-1]))
        out = self.body(inputs_flatten).view((*shape[:-1], self.out_features))
        return out

    def to(self, device):
        self.device = device
        self.body.to(device)