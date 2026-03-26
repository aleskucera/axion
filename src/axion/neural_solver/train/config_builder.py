from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml


def load_default_cfg(cfg_path: str | Path) -> dict:
    cfg_path = Path(cfg_path)
    with cfg_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {cfg_path} did not load to a dict.")
    return cfg


def validate_cfg(cfg: Mapping[str, Any]) -> None:
    if "network" not in cfg or not isinstance(cfg["network"], Mapping):
        raise ValueError("Config must define 'network' as a mapping.")
    if "transformer" not in cfg["network"]:
        raise ValueError(
            "Only transformer model is supported; config must define network.transformer"
        )
    utils_provider_name = cfg["env"]["utils_provider_cfg"]["name"]
    if utils_provider_name != "TransformerNeuralModelUtilsProvider":
        raise ValueError(
            "Only TransformerNeuralModelUtilsProvider is supported. Got: "
            + str(utils_provider_name)
        )
    if cfg["env"]["utils_provider_cfg"].get("num_states_history") != cfg["algorithm"].get(
        "sample_sequence_length"
    ):
        raise ValueError(
            "'num_states_history' must equal 'sample_sequence_length' for the transformer."
        )


def build_cli_cfg(
    *,
    logdir: str | Path,
    train: bool,
    cfg: Mapping[str, Any],
    save_interval: Optional[int] = None,
    log_interval: Optional[int] = None,
    eval_interval: Optional[int] = None,
    skip_check_log_override: bool = False,
) -> dict:
    algo = cfg.get("algorithm", {}) if isinstance(cfg.get("algorithm", {}), Mapping) else {}
    env = cfg.get("env", {}) if isinstance(cfg.get("env", {}), Mapping) else {}
    return {
        "logdir": str(logdir),
        "train": bool(train),
        "render": bool(env.get("render", False)),
        "save_interval": int(save_interval if save_interval is not None else algo.get("save_interval", 50)),
        "log_interval": int(log_interval if log_interval is not None else algo.get("log_interval", 1)),
        "eval_interval": int(eval_interval if eval_interval is not None else algo.get("eval_interval", 1)),
        "skip_check_log_override": bool(skip_check_log_override),
    }


def _ensure_path(d: MutableMapping[str, Any], path: list[str]) -> MutableMapping[str, Any]:
    cur: MutableMapping[str, Any] = d
    for key in path:
        nxt = cur.get(key)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    return cur


def _set_by_path(d: MutableMapping[str, Any], path: list[str], value: Any) -> None:
    if not path:
        raise ValueError("Empty path")
    parent = _ensure_path(d, path[:-1])
    parent[path[-1]] = value


def apply_sweep_overrides(cfg: Mapping[str, Any], sweep: Mapping[str, Any]) -> dict:
    """Return a new cfg with sweep params applied.

    Only keys present in `sweep` are overridden. Everything else stays at YAML defaults.
    """
    out = deepcopy(dict(cfg))

    # Normalize sweep config into a plain dict (wandb.config is dict-like).
    sweep_dict = dict(sweep)

    mapping: dict[str, list[str]] = {
        "lr_start": ["algorithm", "optimizer", "lr_start"],
        "lr_schedule": ["algorithm", "optimizer", "lr_schedule"],
        "optimizer": ["algorithm", "optimizer", "name"],
        "weight_decay": ["algorithm", "optimizer", "weight_decay"],
        "batch_size": ["algorithm", "batch_size"],
        "dropout": ["network", "transformer", "dropout"],
        "normalize_input": ["network", "normalize_input"],
        "normalize_output": ["network", "normalize_output"],
        "huber_delta": ["algorithm", "loss", "huber_delta"],
        "kinematics_loss_weight": ["algorithm", "loss", "kinematics_loss_weight"],
    }

    for sweep_key, cfg_path in mapping.items():
        if sweep_key in sweep_dict and sweep_dict[sweep_key] is not None:
            _set_by_path(out, cfg_path, sweep_dict[sweep_key])

    return out

