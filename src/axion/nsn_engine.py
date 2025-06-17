import warp.sim
from warp.sim import Control
from warp.sim import integrate_bodies
from warp.sim import Integrator
from warp.sim import Model
from warp.sim import State


class NSNEngine(Integrator):
    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        regularization: float = 1e-4,
    ):
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.regularization = regularization

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        control: Control | None = None,
    ):

        if state_in.particle_count > 0:
            raise ValueError("NSNEngine does not support particles.")

        if not state_in.body_count:
            raise ValueError("State must contain at least one body.")

        if control is None:
            control = model.control(clone_variables=False)

        # Get the initial guess for the output state. This will be used as the starting point for the iterative solver.

        return


def main():
    print(f"Hello world")


if __name__ == "__main__":
    main()
