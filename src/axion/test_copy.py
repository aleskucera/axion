import numpy as np
import warp as wp


@wp.kernel
def add_one(x: wp.array(dtype=wp.float32)):
    wp.atomic_add(x, wp.tid(), wp.float32(1.0))


B = 3

# Create body_qd array which consists of multiple spatial vectors
body_qd = wp.array(np.arange(B * 6), dtype=wp.spatial_vector)


x = wp.zeros((B * 6 + 2), dtype=wp.float32)

# Test if I can copy the body_qd into the x
wp.copy(dest=x, src=body_qd, dest_offset=0, src_offset=0, count=B)

print(x.numpy())

wp.launch(
    kernel=add_one,
    dim=B * 6 + 2,
    inputs=[x],
)

wp.synchronize()

print(x.numpy())
print(body_qd.numpy())
