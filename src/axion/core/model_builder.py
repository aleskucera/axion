import newton
import warp as wp
from axion.core.types import JointMode


class AxionModelBuilder(newton.ModelBuilder):
    """
    A custom ModelBuilder for Axion that adds necessary attributes for PID control and joint modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_axion_custom_attributes()

    def _add_axion_custom_attributes(self):
        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_dof_mode",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.int32,
                default=JointMode.NONE,
                assignment=newton.ModelAttributeAssignment.MODEL,
            )
        )

        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_compliance",
                frequency=newton.ModelAttributeFrequency.JOINT,
                dtype=wp.float32,
                default=-1.0,
                assignment=newton.ModelAttributeAssignment.MODEL,
            )
        )

        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="track_u_offset",
                frequency=newton.ModelAttributeFrequency.JOINT,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.ModelAttributeAssignment.MODEL,
            )
        )

        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="is_track_joint",
                frequency=newton.ModelAttributeFrequency.JOINT,
                dtype=wp.int32,
                default=0,
                assignment=newton.ModelAttributeAssignment.MODEL,
            )
        )

        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="track_velocity",
                frequency=newton.ModelAttributeFrequency.JOINT,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.ModelAttributeAssignment.CONTROL,
            )
        )

    def add_track(
        self,
        parent_body: int,
        num_elements: int,
        element_radius: float,
        element_half_width: float,
        shape_config: newton.ModelBuilder.ShapeConfig,
        track_helper,
        track_center: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
        track_rotation: wp.quat = wp.quat_identity(),
        parent_world_xform: wp.transform = wp.transform_identity(),
        name_prefix: str = "track",
    ):
        """
        Adds a sequence of capsules constrained to a track path.

        Args:
            parent_body: The body index to attach the track elements to (e.g., base/world).
            num_elements: Number of elements to place on the track.
            element_radius: Radius of the capsule elements.
            element_half_width: Half-width (half-height along Z) of the capsule elements.
            shape_config: Configuration for the visual/collision shape.
            track_helper: An object with properties `total_len` and method `get_frame(u)`.
            track_center: Offset for the entire track system.
            track_rotation: Rotation for the entire track system.
            parent_world_xform: The initial world transform of the parent body.
                                Used to initialize track links at the correct world location.
            name_prefix: Prefix for the track link keys.
        """
        import numpy as np

        spacing = track_helper.total_len / num_elements

        # Update shape config to use negative collision group so tracks don't collide with themselves
        track_shape_config = shape_config.copy()
        track_shape_config.collision_group = -1

        # Transform for the track base
        X_track = wp.transform(track_center, track_rotation)

        created_joints = []
        for i in range(num_elements):
            u = i * spacing

            # Get track frame (assuming 2D track in XY plane for now)
            # track_helper.get_frame(u) returns pos (2D), tan (2D)
            pos_2d, tan_2d = track_helper.get_frame(u)

            # Convert to 3D local frame
            tangent = np.array([tan_2d[0], tan_2d[1], 0.0])
            normal = np.array([-tan_2d[1], tan_2d[0], 0.0])
            binormal = np.array([0.0, 0.0, 1.0])

            pos_local = np.array([pos_2d[0], pos_2d[1], 0.0])

            # Orientation matrix to quaternion
            # Frame: X=Tangent, Y=Normal, Z=Binormal
            rot_matrix = np.column_stack((tangent, normal, binormal))
            q_local = wp.quat_from_matrix(
                wp.matrix_from_cols(wp.vec3(tangent), wp.vec3(normal), wp.vec3(binormal))
            )

            # Compute world transform of the anchor point on the track
            X_anchor_local = wp.transform(wp.vec3(pos_local), q_local)

            # X_anchor_relative is the pose of the link relative to the parent body
            X_anchor_relative = wp.transform_multiply(X_track, X_anchor_local)

            # X_link_world is the initial global pose of the link
            X_link_world = wp.transform_multiply(parent_world_xform, X_anchor_relative)

            # Create the link body
            # We position it exactly at the anchor point initially
            link = self.add_link(
                key=f"{name_prefix}_link_{i}",
                mass=0.0,  # Kinematic / infinite mass effectively if fixed?
                # Actually, if it's attached via FIXED joint to parent, its mass matters less for statics,
                # but for dynamics, if parent is static, this is static.
                xform=X_link_world,
            )

            self.add_shape_capsule(
                body=link,
                radius=element_radius,
                half_height=element_half_width,
                cfg=track_shape_config,
            )

            # Add FIXED Joint
            # Connect parent to link.
            # parent_xform is the location of the joint on the parent body (track path).
            # child_xform is identity (joint is at the center of the link).

            joint_idx = self.add_joint(
                newton.JointType.FIXED,
                parent_body,
                link,
                parent_xform=X_anchor_relative,
                child_xform=wp.transform_identity(),
                custom_attributes={"track_u_offset": u, "is_track_joint": 1},
            )
            created_joints.append(joint_idx)

        return created_joints

    def finalize_replicated(
        self,
        num_worlds: int,
        gravity: float = -9.81,
        requires_grad: bool = False,
        **kwargs,
    ) -> newton.Model:
        """
        Creates a new newton.ModelBuilder, replicates the content of this builder into it
        for the specified number of worlds, and finalizes it to return the Model.
        """
        final_builder = newton.ModelBuilder(gravity=gravity)
        for k, v in kwargs.items():
            setattr(final_builder, k, v)
        final_builder.replicate(self, num_worlds=num_worlds)
        return final_builder.finalize(requires_grad=requires_grad)
