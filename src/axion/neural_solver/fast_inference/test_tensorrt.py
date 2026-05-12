import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)

network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

input_tensor = network.add_input("input", trt.float32, (1, 3))

identity = network.add_identity(input_tensor)

network.mark_output(identity.get_output(0))

config = builder.create_builder_config()

engine = builder.build_serialized_network(network, config)

print("Engine built:", engine is not None)