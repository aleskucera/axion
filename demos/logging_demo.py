import numpy as np
from axion import HDF5Logger
from axion import HDF5Reader

DEMO_FILENAME = "demo_log_with_scalars.h5"


def run_logging_demo():
    print("--- Running Logging Demo ---")
    with HDF5Logger(DEMO_FILENAME) as logger:
        # Set top-level attributes for the whole file
        logger.log_attribute("simulation_name", "Helhest Demo")
        logger.log_attribute("version", "1.1")

        for t in range(2):
            with logger.scope(f"timestep_{t:02d}"):
                # Log an attribute specific to this timestep
                logger.log_attribute("dt", 0.016)

                # Log a single float and a single string as SCALAR DATASETS
                logger.log_scalar("residual_norm", 1.23e-4 * (t + 1))
                logger.log_scalar("status_message", f"Timestep {t} completed.")

                fake_J = np.random.rand(20, 60)
                logger.log_np_dataset("J", fake_J)

                # Attach an attribute DIRECTLY to the 'J' dataset
                logger.log_attribute("description", "This is the Jacobian matrix.", target_path="J")

    print("--- Logging Demo Finished ---")


def run_reading_demo():
    print("\n--- Running Reading Demo ---")
    with HDF5Reader(DEMO_FILENAME) as reader:
        # The enhanced tree view will now show attributes
        reader.print_tree()

        # Retrieve different types of data
        print("\n--- Retrieving Data ---")

        # 1. Get a top-level attribute
        sim_name = reader.get_attribute("/", "simulation_name")
        print(f"Retrieved root attribute 'simulation_name': {sim_name}")

        # 2. Get an attribute from a group
        dt_val = reader.get_attribute("timestep_01", "dt")
        print(f"Retrieved group attribute 'dt' from timestep_01: {dt_val}")

        # 3. Get an attribute from a dataset
        desc = reader.get_attribute("timestep_01/J", "description")
        print(f"Retrieved dataset attribute 'description' from J: {desc}")

        # 4. Get a scalar dataset
        status = reader.get_scalar("timestep_01/status_message")
        res_norm = reader.get_scalar("timestep_01/residual_norm")
        print(f"Retrieved scalar 'status_message': {status}")
        print(f"Retrieved scalar 'residual_norm': {res_norm:.6f}")

        # 5. List attributes on an object
        attrs_on_j = reader.list_attributes("timestep_01/J")
        print(f"Attributes on timestep_01/J: {attrs_on_j}")

    print("--- Reading Demo Finished ---")


def main():

    run_logging_demo()
    run_reading_demo()


if __name__ == "__main__":
    main()
