import newton

class Model:
    """Adapter for wp.sim.Model -> newton.Model"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.Model(*args, **kwargs)

class State:
    """Adapter for wp.sim.State -> newton.State"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.State(*args, **kwargs)

class Control:
    """Adapter for wp.sim.Control -> newton.Control"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.Control(*args, **kwargs)

class ModelBuilder:
    """Adapter for wp.sim.ModelBuilder -> newton.ModelBuilder"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.ModelBuilder(*args, **kwargs)

class Mesh:
    """Adapter for wp.sim.Mesh -> newton.Mesh"""
    def __init__(
        self,
        vertices,
        indices,
        normals=None,
        uvs=None,
        compute_inertia=True,
        is_solid=True,
        maxhullvert=32,   # or MESH_MAXHULLVERT constant
        color=None,
        **kwargs
    ):
        # store Newton mesh internally
        self.inner = newton.Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
            is_solid=is_solid,
            maxhullvert=maxhullvert,
            color=color,
        )