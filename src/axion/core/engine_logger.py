from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import warp as wp
from axion.core.logging_config import LoggingConfig
from axion.logging import HDF5Logger
from axion.logging import NullLogger


class EngineMode(Enum):
    PRODUCTION = 0  # capture_while, zero python overhead
    TIMING = 1  # capture unrolled graph, baked-in warp events
    DEBUG = 2  # python loop, full signals and hdf5 logging


class Signal:
    """A lightweight event dispatcher for the DEBUG path."""

    def __init__(self):
        self._observers: List[Callable] = []

    def connect(self, callback: Callable):
        self._observers.append(callback)

    def emit(self, **kwargs):
        for callback in self._observers:
            callback(**kwargs)


class PolymorphicScope:
    """
    A scope that adapts its behavior based on the active EngineMode.

    - PRODUCTION: Effectively a 'pass'. Zero runtime cost during graph capture.
    - TIMING: Records start/end Warp events into the graph (GPU timestamps).
    - DEBUG: Emits Python signals with data payloads (CPU timestamps + Data).
    """

    def __init__(self, name: str, events: "EngineEvents"):
        self.name = name
        self.events = events
        self.on_enter_signal = Signal()
        self.on_exit_signal = Signal()

    @contextmanager
    def scope(self, iter_idx: int = 0, **kwargs):
        mode = self.events.current_mode

        # --- ENTER ---
        if mode == EngineMode.DEBUG:
            # CPU timestamp (optional) + Signal
            self.on_enter_signal.emit(iter_idx=iter_idx, **kwargs)

        elif mode == EngineMode.TIMING:
            # GPU Timestamp baked into graph
            evt = self.events.get_timing_event(self.name, iter_idx, start=True)
            if evt:
                wp.record_event(evt)

        # --- YIELD (Run Physics) ---
        yield

        # --- EXIT ---
        if mode == EngineMode.DEBUG:
            self.on_exit_signal.emit(iter_idx=iter_idx, **kwargs)

        elif mode == EngineMode.TIMING:
            evt = self.events.get_timing_event(self.name, iter_idx, start=False)
            if evt:
                wp.record_event(evt)


@dataclass
class EngineEvents:
    """
    The central nervous system of the engine.
    """

    current_mode: EngineMode = EngineMode.PRODUCTION

    # 1. Signals (Used in Debug Mode)
    step_start: Signal = field(default_factory=Signal)
    step_end: Signal = field(default_factory=Signal)
    newton_iteration_end: Signal = field(default_factory=Signal)

    # 2. Scopes (Polymorphic)
    step: PolymorphicScope = field(init=False)
    control: PolymorphicScope = field(init=False)
    initial_guess: PolymorphicScope = field(init=False)
    linearization: PolymorphicScope = field(init=False)
    linear_solve: PolymorphicScope = field(init=False)
    linesearch: PolymorphicScope = field(init=False)

    # 3. Timing Pool (Used in Timing Mode)
    # Structure: _timing_pool[iteration][scope_name] = (StartEvent, EndEvent)
    _timing_pool: List[Dict[str, tuple[wp.Event, wp.Event]]] = field(default_factory=list)
    _step_timing_pool: Dict[str, tuple[wp.Event, wp.Event]] = field(default_factory=dict)

    # 4. Aggregated Timing Data
    _timing_history: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        # Initialize scopes
        self.step = PolymorphicScope("step", self)
        self.control = PolymorphicScope("control", self)
        self.initial_guess = PolymorphicScope("initial_guess", self)
        self.linearization = PolymorphicScope("linearization", self)
        self.linear_solve = PolymorphicScope("linear_solve", self)
        self.linesearch = PolymorphicScope("linesearch", self)

    def allocate_timing_events(self, max_iters: int):
        """Pre-allocates events for the 'Unrolled Graph' timing mode."""
        self._step_timing_pool = {
            "step": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
            "control": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
            "initial_guess": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
        }

        self._timing_pool = []
        for _ in range(max_iters):
            self._timing_pool.append(
                {
                    "linearization": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
                    "linear_solve": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
                    "linesearch": (wp.Event(enable_timing=True), wp.Event(enable_timing=True)),
                }
            )

    def get_timing_event(self, name: str, iter_idx: int, start: bool) -> Optional[wp.Event]:
        # Check Step-level events first
        if name in self._step_timing_pool:
            pair = self._step_timing_pool[name]
            return pair[0] if start else pair[1]

        # Check Iteration-level events
        if 0 <= iter_idx < len(self._timing_pool):
            if name in self._timing_pool[iter_idx]:
                pair = self._timing_pool[iter_idx][name]
                return pair[0] if start else pair[1]

        return None

    def record_timings(self):
        """Captures current event values and appends to history."""
        if self.current_mode != EngineMode.TIMING:
            return

        record = {}

        # Step-level timings
        for name, (start, end) in self._step_timing_pool.items():
            record[name] = wp.get_event_elapsed_time(start, end)

        # Iteration-level timings
        for i, iter_events in enumerate(self._timing_pool):
            for name, (start, end) in iter_events.items():
                key = f"iter_{i:02d}_{name}"
                record[key] = wp.get_event_elapsed_time(start, end)

        self._timing_history.append(record)

    def print_timings(self):
        """Rudimentary report printer for TIMING mode."""
        if self.current_mode != EngineMode.TIMING:
            print("Timings not available (Mode is not TIMING)")
            return

        self.print_stats()

    def print_stats(self):
        """Prints aggregated statistics using Pandas if available."""
        if not self._timing_history:
            print("No timing data collected.")
            return

        # Flatten/Aggregate the per-iteration keys
        aggregated_history = []
        for record in self._timing_history:
            new_record = {}
            # Initialize accumulators for Newton steps
            newton_metrics = {"linearization": [], "linear_solve": [], "linesearch": []}

            for k, v in record.items():
                if k.startswith("iter_"):
                    # k format: iter_00_linearization
                    parts = k.split("_", 2)
                    if len(parts) == 3:
                        metric_name = parts[2]
                        if metric_name in newton_metrics:
                            newton_metrics[metric_name].append(v)
                else:
                    new_record[k] = v

            # Add averaged Newton metrics to the record
            # We treat "newton_linearization" as the average time of ONE linearization call
            for name, values in newton_metrics.items():
                if values:
                    new_record[f"newton_{name}"] = sum(values) / len(values)

            aggregated_history.append(new_record)

        try:
            import pandas as pd

            df = pd.DataFrame(aggregated_history)

            # Reorder columns for readability if possible
            preferred_order = [
                "step",
                "control",
                "initial_guess",
                "newton_linearization",
                "newton_linear_solve",
                "newton_linesearch",
            ]
            cols = [c for c in preferred_order if c in df.columns] + [
                c for c in df.columns if c not in preferred_order
            ]
            df = df[cols]

            print("\n=== GPU TIMING STATISTICS (ms) ===")
            print(df.describe().T[["mean", "std", "min", "max"]])
        except ImportError:
            print("\n=== GPU TIMING STATISTICS (ms) (Install pandas for better formatting) ===")
            if not aggregated_history:
                return
            keys = aggregated_history[0].keys()
            # Simple sorting
            preferred_order = [
                "step",
                "control",
                "initial_guess",
                "newton_linearization",
                "newton_linear_solve",
                "newton_linesearch",
            ]
            sorted_keys = [k for k in preferred_order if k in keys] + [
                k for k in keys if k not in preferred_order
            ]

            print(f"{'Metric':<30} | {'Mean':<10} | {'Min':<10} | {'Max':<10}")
            print("-" * 70)
            for k in sorted_keys:
                values = [r[k] for r in aggregated_history if k in r]
                if not values:
                    continue
                avg = sum(values) / len(values)
                print(f"{k:<30} | {avg:<10.3f} | {min(values):<10.3f} | {max(values):<10.3f}")


class HDF5Observer:
    """Listens to Data Signals and writes to HDF5 (Debug Mode only)."""

    def __init__(self, events: EngineEvents, config: LoggingConfig):
        self.config = config
        self.logger = NullLogger()
        if config.enable_hdf5_logging:
            self.logger = HDF5Logger(filepath=config.hdf5_log_file)
            self.logger.open()

            # Connect
            events.step.on_enter_signal.connect(self._on_step_start)
            events.step.on_exit_signal.connect(self._on_step_end)
            events.newton_iteration_end.connect(self._log_iteration)

    def close(self):
        if self.config.enable_hdf5_logging:
            self.logger.close()

    def _on_step_start(self, iter_idx: int, time: float = None, **kwargs):
        self._step_scope = self.logger.scope(f"timestep_{iter_idx:04d}")
        self._step_scope.__enter__()
        if time is not None:
            self.logger.log_scalar("time", time)

    def _on_step_end(self, **kwargs):
        if hasattr(self, "_step_scope"):
            self._step_scope.__exit__(None, None, None)

    def _log_iteration(self, iter_idx: int, snapshot: Dict, **kwargs):
        with self.logger.scope(f"newton_iteration_{iter_idx:02d}"):
            self._recursive_log(snapshot)

    def _recursive_log(self, data: Dict[str, Any]):
        """Recursively logs a dictionary of data."""
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, dict):
                with self.logger.scope(k):
                    self._recursive_log(v)
            elif isinstance(v, np.ndarray):
                self.logger.log_np_dataset(k, v)
            else:
                self.logger.log_scalar(k, v)
