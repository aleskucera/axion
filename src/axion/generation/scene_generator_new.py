import math
import random

import newton
import numpy as np
import warp as wp


class SceneGenerator:
    def __init__(self, builder: newton.ModelBuilder, seed=42):
        """
        Initialize the scene generator.

        Args:
            builder: The newton.ModelBuilder to add objects to.
            seed: Random seed for reproducibility.
        """
        self.builder = builder

        random.seed(seed)
        np.random.seed(seed)

    def _random_params(self, shape_type: str, size_bounds: tuple):
        if shape_type == "box":
            return {
                "hx": random.uniform(size_bounds[0], size_bounds[1]),
                "hy": random.uniform(size_bounds[0], size_bounds[1]),
                "hz": random.uniform(size_bounds[0], size_bounds[1]),
            }
        elif shape_type == "sphere":
            return {"radius": random.uniform(size_bounds[0], size_bounds[1])}
        elif shape_type == "capsule":
            return {
                "radius": random.uniform(size_bounds[0], size_bounds[1]),
                "half_height": random.uniform(size_bounds[0], size_bounds[1]),
            }
        return {}

    def _random_xform(self, pos_bounds: tuple):
        p_min, p_max = pos_bounds
        pos = wp.vec3(
            random.uniform(p_min[0], p_max[0]),
            random.uniform(p_min[1], p_max[1]),
            random.uniform(p_min[2], p_max[2]),
        )

        # Random rotation
        axis_vec = [random.uniform(-1, 1) for _ in range(3)]
        if all(v == 0 for v in axis_vec):
            axis_vec = [0, 0, 1]
        axis = wp.normalize(wp.vec3(*axis_vec))
        angle = random.uniform(0, 2 * math.pi)
        rot = wp.quat_from_axis_angle(axis, angle)

        return wp.transform(pos, rot)

    def _random_mass(self, mass_bounds: tuple):
        return random.uniform(mass_bounds[0], mass_bounds[1])

    def _add_to_builder(
        self, shape_type: str, params: dict, xform: wp.transform, mass: float
    ) -> int:
        body = self.builder.add_body(xform=xform, mass=mass)
        if shape_type == "box":
            self.builder.add_shape_box(body, hx=params["hx"], hy=params["hy"], hz=params["hz"])
        elif shape_type == "sphere":
            self.builder.add_shape_sphere(body, radius=params["radius"])
        elif shape_type == "capsule":
            self.builder.add_shape_capsule(
                body, radius=params["radius"], half_height=params["half_height"]
            )

        return body

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate_random_object(
        self,
        pos_bounds: tuple,
        mass_bounds: tuple,
        size_bounds: tuple,
        shape_type: str = None,
    ) -> int:
        """Generates a random object that doesn't touch anything."""
        if shape_type is None:
            shape_type = random.choice(["box", "sphere", "capsule"])

        params = self._random_params(shape_type, size_bounds)
        xform = self._random_xform(pos_bounds)
        mass = self._random_mass(mass_bounds)
        self._add_to_builder(shape_type, params, xform, mass)
