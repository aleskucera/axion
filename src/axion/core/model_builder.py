import newton
import warp as wp
from axion.core.control_utils import JointMode


class AxionModelBuilder(newton.ModelBuilder):
    """
    A custom ModelBuilder for Axion that adds necessary attributes for PID control and joint modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_axion_custom_attributes()

    def _add_axion_custom_attributes(self):
        # integral constant (PID control)
        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_target_ki",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.ModelAttributeAssignment.MODEL,
            )
        )

        # previous instance of the control error (PID control)
        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_err_prev",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.ModelAttributeAssignment.CONTROL,
            )
        )

        # cumulative error of the integral part (PID control)
        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_err_i",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.ModelAttributeAssignment.CONTROL,
            )
        )

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
                name="joint_target",
                frequency=newton.ModelAttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.ModelAttributeAssignment.CONTROL,
            )
        )

    def finalize_replicated(self, num_worlds: int, gravity: float = -9.81, **kwargs) -> newton.Model:
        """
        Creates a new newton.ModelBuilder, replicates the content of this builder into it
        for the specified number of worlds, and finalizes it to return the Model.
        """
        final_builder = newton.ModelBuilder(gravity=gravity)
        for k, v in kwargs.items():
            setattr(final_builder, k, v)
        final_builder.replicate(self, num_worlds=num_worlds)
        return final_builder.finalize()
