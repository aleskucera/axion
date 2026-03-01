# Newton Contacts

As of commit id: 08296988

```
class Contacts:

    def __init__(
        self,
        rigid_contact_max: int,
        soft_contact_max: int,
        requires_grad: bool = False,
        device: Devicelike = None,
        per_contact_shape_properties: bool = False,
    ):
        self.per_contact_shape_properties = per_contact_shape_properties
        with wp.ScopedDevice(device):
            # rigid contacts
            self.rigid_contact_count = wp.zeros(1, dtype=wp.int32)
            self.rigid_contact_point_id = wp.zeros(rigid_contact_max, dtype=wp.int32)
            self.rigid_contact_shape0 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_shape1 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_point0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_point1 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_offset0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_offset1 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_normal = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_thickness0 = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            self.rigid_contact_thickness1 = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            self.rigid_contact_tids = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            # to be filled by the solver (currently unused)
            self.rigid_contact_force = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)

            # contact stiffness/damping/friction (only allocated if per_contact_shape_properties is enabled)
            if self.per_contact_shape_properties:
                self.rigid_contact_stiffness = wp.zeros(
                    rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad
                )
                self.rigid_contact_damping = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
                self.rigid_contact_friction = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            else:
                self.rigid_contact_stiffness = None
                self.rigid_contact_damping = None
                self.rigid_contact_friction = None

            # soft contacts ....
```

## NeRD paper
- used warp.sim contacts but edited them
- used:
    - contact points 0, contacts points 1
    - contacts normals
    - contact thickness (singular)
    - contact depths
    - contacts masks