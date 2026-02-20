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

import wandb

project = "neural-solver-transformer"


class Logger:
    def __init__(self):
        self.wandb = None
        self.wandb_logs = {}

    def init_wandb(self, wandb_project=project, wandb_name=None):
        self.wandb = wandb.init(
            project=wandb_project,
            name=wandb_name
        )

    def init_epoch(self, epoch):
        self.wandb_logs = {"step": epoch}

    def add_scalar(self, name, value, step):
        if self.wandb:
            self.wandb_logs[name] = value

    def flush(self):
        if self.wandb:
            wandb.log(self.wandb_logs)

    def finish(self):
        if self.wandb:
            wandb.finish()
