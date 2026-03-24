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

import numpy as np
import random
import torch
import sys
import csv
from pathlib import Path

# if there's overlap between args_list and commandline input, use commandline input
def solve_argv_conflict(args_list):
    arguments_to_be_removed = []
    arguments_size = []

    for argv in sys.argv[1:]:
        if argv.startswith('-'):
            size_count = 1
            for i, args in enumerate(args_list):
                if args == argv:
                    arguments_to_be_removed.append(args)
                    for more_args in args_list[i+1:]:
                        if not more_args.startswith('-'):
                            size_count += 1
                        else:
                            break
                    arguments_size.append(size_count)
                    break

    for args, size in zip(arguments_to_be_removed, arguments_size):
        args_index = args_list.index(args)
        for _ in range(size):
            args_list.pop(args_index)

def convert_dict_to_hydra_str_list(cfg, prefix=""):
    hydra_str_list = []
    for key in cfg.keys():
        if type(cfg[key]) is dict:
            hydra_str_list.extend(
                convert_dict_to_hydra_str_list(cfg[key], prefix + "." + key)
            )
        else:
            hydra_str_list.append(prefix + "." + key + "=" + str(cfg[key]))
    return hydra_str_list

def override_cfg_entry(key, value, cfg):
    idx = key.find('.')
    if idx == -1:
        if key not in cfg:
            return False
        if value.replace('.', '').isnumeric(): # TODO: to support other types
            cfg[key] = eval(value)
        elif value == 'True' or value == 'False':
            cfg[key] = eval(value)
        else:
            cfg[key] = value
        return True
    else:
        if key[:idx] not in cfg:
            return False
        return override_cfg_entry(key[idx + 1:], value, cfg[key[:idx]])
    
def handle_cfg_overrides(cfg_overrides, cfg):
    overrides = cfg_overrides.split()
    assert len(overrides) % 2 == 0
    for i in range(0, len(overrides), 2):
        key = overrides[i]
        value = overrides[i + 1]
        if not override_cfg_entry(key, value, cfg):
            print_error(f'No key {key} in config to be override')

def format_number(value, precision = 8, scientific_threshold = 1e4):
    value = float(value)
    if not np.isfinite(value):
        return str(value)

    abs_value = abs(value)
    if abs_value != 0.0 and (
        abs_value >= scientific_threshold or abs_value < 10 ** (-precision)
    ):
        return "{val:.{precision}e}".format(val=value, precision=precision)
    return "{val:.{precision}f}".format(val=value, precision=precision)

def format_value(value, precision = 8, scientific_threshold = 1e4):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return format_number(
                value.detach().cpu().item(),
                precision = precision,
                scientific_threshold = scientific_threshold,
            )
        array = value.detach().cpu().numpy()
        return np.array2string(
            array,
            precision = precision,
            separator = ', ',
            formatter = {
                'float_kind': lambda x: format_number(
                    x,
                    precision = precision,
                    scientific_threshold = scientific_threshold,
                )
            },
        )

    if isinstance(value, np.ndarray):
        return np.array2string(
            value,
            precision = precision,
            separator = ', ',
            formatter = {
                'float_kind': lambda x: format_number(
                    x,
                    precision = precision,
                    scientific_threshold = scientific_threshold,
                )
            },
        )

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (float, int)):
        return format_number(
            value,
            precision = precision,
            scientific_threshold = scientific_threshold,
        )

    return str(value)

def format_dict(dict, precision = 8):
    dict_str = "{"
    items_cnt = 0
    for k, v in dict.items():
        k_str = format_value(k, precision = precision)
        v_str = format_value(v, precision = precision)
        dict_str += "{}: {}".format(k_str, v_str)
        items_cnt += 1
        if items_cnt < len(dict):
            dict_str += ", "
    dict_str += "}"

    return dict_str

def format_model_name(checkpoint_path: str) -> str:
    if checkpoint_path is None:
        return ""
    p = Path(checkpoint_path)
    parts = p.parts
    if "nn" in parts:
        nn_idx = parts.index("nn")
        if nn_idx > 0:
            return str(Path(*parts[nn_idx - 1:]))
    return p.name

def append_test_results_csv(csv_path: Path, row: dict):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if (not file_exists) or csv_path.stat().st_size == 0:
            writer.writerow([
                "model_name",
                "validation_dataset_name",
                "valid_loss_total",
                "state_eval_error_total",
                "lambda_eval_error_total",
            ])
        writer.writerow([
            row["model_name"],
            row.get("validation_dataset_name", ""),
            row["valid_loss_total"],
            row["state_eval_error_total"],
            row["lambda_eval_error_total"],
        ])

def get_validation_dataset_name(cfg: dict) -> str:
    valid_cfg = cfg.get("algorithm", {}).get("dataset", {}).get("valid_datasets", {})
    dataset_paths = []
    for _, dataset_path in valid_cfg.items():
        dataset_paths.append(Path(dataset_path).name)
    if len(dataset_paths) == 0:
        return ""
    return ";".join(dataset_paths)

from datetime import datetime

def get_time_stamp():
    now = datetime.now()
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    hour = now.strftime('%H')
    minute = now.strftime('%M')
    second = now.strftime('%S')
    return '{}-{}-{}-{}-{}-{}'.format(month, day, year, hour, minute, second)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError


def print_ok(*message):
    print('\033[92m', *message, '\033[0m')


def print_warning(*message):
    print('\033[91m', *message, '\033[0m')


def print_info(*message):
    print('\033[96m', *message, '\033[0m')


def print_white(*message):
    print('\033[37m', *message, '\033[0m')