"""Persistent-state logging for the Axion engine.

Three domain-specific HDF5 loggers, each consumed by AxionEngineBase
when the corresponding sub-config (``LoggingConfig.{hdf5,dataset,adjoint}``)
is enabled. The configs themselves live in ``axion.core.logging_config``
because they are part of the engine config tree.

``HDF5Logger`` / ``NullLogger`` are a scoped, signal-driven API used by
GPTEngine and ``engine_logger.HDF5Observer`` (debug-mode event logging).
"""
from .adjoint_logger import AdjointHDF5Logger
from .dataset_logger import DatasetHDF5Logger
from .hdf5_logger import HDF5Logger
from .null_logger import NullLogger
from .sim_logger import SimulationHDF5Logger


__all__ = [
    "AdjointHDF5Logger",
    "DatasetHDF5Logger",
    "HDF5Logger",
    "NullLogger",
    "SimulationHDF5Logger",
]
