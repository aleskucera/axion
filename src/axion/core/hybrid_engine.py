from typing import Optional

from newton import Model

from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig


class HybridEngine(AxionEngine):
    """
    Compatibility wrapper around :class:`axion.core.engine.AxionEngine`.

    This keeps Hydra configs and examples that select `engine: hybrid` runnable
    after internal refactors.
    """

    def __init__(
        self,
        model: Model,
        sim_steps: int,
        config: Optional[AxionEngineConfig] = AxionEngineConfig(),
        logging_config: Optional[LoggingConfig] = LoggingConfig(),
        differentiable_simulation: bool = False,
    ):
        super().__init__(
            model=model,
            sim_steps=sim_steps,
            config=config,
            logging_config=logging_config,
            differentiable_simulation=differentiable_simulation,
        )
